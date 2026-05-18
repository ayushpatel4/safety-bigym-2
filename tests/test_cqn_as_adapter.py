"""Unit tests for the CQN-AS env adapter.

The adapter (``safety_bigym.agents.cqn_as.env_adapter``) glues
``SafetyBiGymEnvFactory``'s gym env to CQN-AS's TimeStep interface. These
tests exercise the pure-Python translation logic (obs shape, action
normalisation, TimeStep step/reset typing, info forwarding) without
booting MuJoCo: ``SafetyBiGymEnvFactory._create_env`` is monkeypatched to
return a stub gym env that scripts the obs/info schema the adapter
expects.

Covers A7 deliverables from the revised forward queue:
- obs shape with/without ``human_pos_estimate`` (bodyslam off vs on)
- action [-1, 1] roundtrip through ``_convert_action_to_raw`` /
  ``_convert_action_from_raw`` (including gripper-tail handling)
- ``step()`` returns a TimeStep with ``info["safety"]`` populated
- ``first()`` / ``mid()`` / ``last()`` on reset / mid / terminal
- ``ExtendedTimeStepWrapper`` attaches the action field
- frame stacking widens low_dim_obs by ``frame_stack``
- pixels=False yields a zero-shaped rgb placeholder
- missing required obs keys raise cleanly
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf

from safety_bigym.agents.cqn_as import env_adapter
from safety_bigym.agents.cqn_as.env_adapter import (
    ExtendedTimeStep,
    ExtendedTimeStepWrapper,
    SafetyBiGymCQNAdapter,
    TimeStep,
)
from safety_bigym.perception.bodyslam_wrapper import OBS_KEY as BODYSLAM_OBS_KEY

try:
    from dm_env import StepType
except ImportError:  # pragma: no cover - dm_env is required by adapter
    pytest.skip("dm_env not installed", allow_module_level=True)


# Proprioception widths roughly match RoboBase's 4-dof H1 wrapping order.
PROPRIO_WIDTH = 60
GRIPPER_WIDTH = 2
FLOATING_BASE_WIDTH = 4
ACTION_DIM = 16  # 14 body joints + 2 grippers, matches the gripper-tail handling


def _episode_safety(ssm_violation: bool = False, pfl_violation: bool = False) -> dict:
    return {
        "ep_ssm_violation_rate": 0.42 if ssm_violation else 0.0,
        "ep_pfl_violation_rate": 0.0,
        "ep_min_ssm_margin": -0.1 if ssm_violation else 0.5,
    }


def _step_safety(ssm_margin: float = 0.5, pfl_force_ratio: float = 0.0) -> dict:
    return {
        "ssm_margin": ssm_margin,
        "ssm_violation": ssm_margin < 0.0,
        "pfl_force_ratio": pfl_force_ratio,
        "pfl_violation": False,
        "min_separation": max(ssm_margin, 0.05),
    }


class _StubSafetyBiGymEnv(gym.Env):
    """Gym env shaped like ``SafetyBiGymEnvFactory._create_env`` output.

    Honours the same observation-space contract the adapter probes:
    ``proprioception``, ``proprioception_grippers``,
    ``proprioception_floating_base`` (always present) plus
    ``human_pos_estimate`` and ``rgb_<cam>`` keys when configured.
    Each step emits ``info["safety"]``; the terminal step also emits
    ``info["episode_safety"]``.
    """

    metadata: dict = {}

    def __init__(
        self,
        *,
        include_human_pos: bool,
        cameras: Tuple[str, ...] = (),
        camera_shape: Tuple[int, int] = (84, 84),
        steps_to_terminal: int = 10,
        action_dim: int = ACTION_DIM,
    ):
        self._include_human_pos = include_human_pos
        self._cameras = tuple(cameras)
        self._camera_shape = camera_shape
        self._steps_to_terminal = steps_to_terminal
        self._t = 0

        obs_dict = {
            "proprioception": spaces.Box(
                -np.inf, np.inf, shape=(PROPRIO_WIDTH,), dtype=np.float32
            ),
            "proprioception_grippers": spaces.Box(
                0.0, 1.0, shape=(GRIPPER_WIDTH,), dtype=np.float32
            ),
            "proprioception_floating_base": spaces.Box(
                -np.inf, np.inf, shape=(FLOATING_BASE_WIDTH,), dtype=np.float32
            ),
        }
        if include_human_pos:
            obs_dict[BODYSLAM_OBS_KEY] = spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float32
            )
        for cam in self._cameras:
            obs_dict[f"rgb_{cam}"] = spaces.Box(
                0, 255, shape=(3, *self._camera_shape), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_dict)

        # Mirrors the real action range: body dims in [-pi, pi]-ish,
        # gripper dims in [0, 1]. The adapter's default action_stats only
        # special-cases the gripper tail, so test against that contract.
        low = -np.ones(action_dim, dtype=np.float32)
        high = np.ones(action_dim, dtype=np.float32)
        low[-2:] = 0.0
        high[-2:] = 1.0
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.last_action: Optional[np.ndarray] = None

    def _make_obs(self) -> dict:
        # Deterministic in t so tests can assert specific values flow through.
        proprio = np.full(
            (PROPRIO_WIDTH,), float(self._t) * 0.01, dtype=np.float32
        )
        grippers = np.full(
            (GRIPPER_WIDTH,), 0.5, dtype=np.float32
        )
        floating = np.full(
            (FLOATING_BASE_WIDTH,), 0.1 * (self._t + 1), dtype=np.float32
        )
        obs = {
            "proprioception": proprio,
            "proprioception_grippers": grippers,
            "proprioception_floating_base": floating,
        }
        if self._include_human_pos:
            obs[BODYSLAM_OBS_KEY] = np.array(
                [1.0, 2.0, 3.0, 0.0, 0.0, 1.0], dtype=np.float32
            )
        for cam in self._cameras:
            obs[f"rgb_{cam}"] = np.full(
                (3, *self._camera_shape), 17, dtype=np.uint8
            )
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        info = {"safety": _step_safety()}
        return self._make_obs(), info

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self._t += 1
        terminated = self._t >= self._steps_to_terminal
        info: dict = {"safety": _step_safety(ssm_margin=0.5 - 0.05 * self._t)}
        if terminated:
            info["episode_safety"] = _episode_safety(ssm_violation=True)
        return self._make_obs(), float(self._t), terminated, False, info

    def render(self):
        return np.zeros((self._camera_shape[0], self._camera_shape[1], 3), dtype=np.uint8)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    bodyslam_mode: str = "off",
    pixels: bool = False,
    cameras: Tuple[str, ...] = (),
    episode_length: int = 200,
    demo_down_sample_rate: int = 20,
):
    cfg = OmegaConf.create(
        {
            "pixels": pixels,
            "visual_observation_shape": [84, 84],
            "env": {
                "task_name": "reach_target_single",
                "bodyslam": {"mode": bodyslam_mode},
                "cameras": list(cameras),
                "episode_length": episode_length,
                "demo_down_sample_rate": demo_down_sample_rate,
            },
        }
    )
    return cfg


@pytest.fixture
def patched_factory(monkeypatch):
    """Replaces ``SafetyBiGymEnvFactory._create_env`` with a stub builder.

    The fixture returns a function the test can call to configure the
    stub env (bodyslam channel on/off, cameras, terminal step). The
    stub is what the adapter will see when it calls ``factory._create_env``.
    """

    state: dict = {"env": None, "kwargs": {}}

    def install(
        *,
        include_human_pos: bool,
        cameras: Tuple[str, ...] = (),
        camera_shape: Tuple[int, int] = (84, 84),
        steps_to_terminal: int = 10,
        action_dim: int = ACTION_DIM,
    ):
        state["kwargs"] = dict(
            include_human_pos=include_human_pos,
            cameras=cameras,
            camera_shape=camera_shape,
            steps_to_terminal=steps_to_terminal,
            action_dim=action_dim,
        )

        def _fake_create_env(self, cfg):
            env = _StubSafetyBiGymEnv(**state["kwargs"])
            state["env"] = env
            return env

        monkeypatch.setattr(
            "safety_bigym.agents.cqn_as.env_adapter.SafetyBiGymEnvFactory."
            "_create_env",
            _fake_create_env,
            raising=True,
        )
        return state

    return install


# ---------------------------------------------------------------------------
# Observation shape
# ---------------------------------------------------------------------------


def test_low_dim_obs_shape_without_bodyslam(patched_factory):
    patched_factory(include_human_pos=False)
    cfg = _make_cfg(bodyslam_mode="off")
    adapter = SafetyBiGymCQNAdapter(cfg)

    expected = PROPRIO_WIDTH + GRIPPER_WIDTH + FLOATING_BASE_WIDTH
    spec = adapter.low_dim_raw_observation_spec()
    assert spec.shape == (expected,), spec.shape

    ts = adapter.reset()
    assert ts.low_dim_obs.shape == (expected,), ts.low_dim_obs.shape
    assert ts.low_dim_obs.dtype == np.float32


def test_low_dim_obs_shape_with_bodyslam(patched_factory):
    patched_factory(include_human_pos=True)
    cfg = _make_cfg(bodyslam_mode="noisy")
    adapter = SafetyBiGymCQNAdapter(cfg)

    expected = PROPRIO_WIDTH + GRIPPER_WIDTH + FLOATING_BASE_WIDTH + 6
    spec = adapter.low_dim_raw_observation_spec()
    assert spec.shape == (expected,), spec.shape

    ts = adapter.reset()
    assert ts.low_dim_obs.shape == (expected,), ts.low_dim_obs.shape
    # Tail of low_dim_obs must equal human_pos_estimate
    assert np.allclose(ts.low_dim_obs[-6:], [1.0, 2.0, 3.0, 0.0, 0.0, 1.0])


def test_bodyslam_oracle_also_injects_human_pos(patched_factory):
    """Adapter gates on ``bodyslam.mode != 'off'``, so ``oracle`` injects too."""
    patched_factory(include_human_pos=True)
    cfg = _make_cfg(bodyslam_mode="oracle")
    adapter = SafetyBiGymCQNAdapter(cfg)
    assert adapter._inject_human_pos is True
    assert adapter.low_dim_raw_observation_space.shape == (
        PROPRIO_WIDTH + GRIPPER_WIDTH + FLOATING_BASE_WIDTH + 6,
    )


def test_missing_state_key_raises(patched_factory):
    """If the env doesn't expose a required proprioception key, fail loudly."""

    def _bad_create_env(self, cfg):
        env = _StubSafetyBiGymEnv(include_human_pos=False)
        # Drop a required key after construction
        env.observation_space = spaces.Dict(
            {
                k: v
                for k, v in env.observation_space.spaces.items()
                if k != "proprioception_grippers"
            }
        )
        return env

    import safety_bigym.agents.cqn_as.env_adapter as ea

    orig = ea.SafetyBiGymEnvFactory._create_env
    ea.SafetyBiGymEnvFactory._create_env = _bad_create_env
    try:
        cfg = _make_cfg(bodyslam_mode="off")
        with pytest.raises(KeyError, match="proprioception_grippers"):
            SafetyBiGymCQNAdapter(cfg)
    finally:
        ea.SafetyBiGymEnvFactory._create_env = orig


def test_missing_bodyslam_key_raises(patched_factory):
    """bodyslam.mode != 'off' but env has no human_pos_estimate -> KeyError."""
    patched_factory(include_human_pos=False)
    cfg = _make_cfg(bodyslam_mode="noisy")
    with pytest.raises(KeyError, match=BODYSLAM_OBS_KEY):
        SafetyBiGymCQNAdapter(cfg)


# ---------------------------------------------------------------------------
# Action normalisation roundtrip
# ---------------------------------------------------------------------------


def test_action_roundtrip_identity(patched_factory):
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())

    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, size=adapter.action_space.shape).astype(np.float32)
    raw = adapter._convert_action_to_raw(a)
    back = adapter._convert_action_from_raw(raw)
    np.testing.assert_allclose(back, a, atol=1e-5)


def test_action_normalisation_maps_gripper_tail(patched_factory):
    """[-1, 1] gripper input must land in [0, 1] raw range."""
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())

    minus_one = -np.ones(adapter.action_space.shape, dtype=np.float32)
    plus_one = np.ones(adapter.action_space.shape, dtype=np.float32)

    raw_lo = adapter._convert_action_to_raw(minus_one)
    raw_hi = adapter._convert_action_to_raw(plus_one)

    # Body dims (all but last 2): -1 -> -1, +1 -> +1
    np.testing.assert_allclose(raw_lo[:-2], -1.0, atol=1e-5)
    np.testing.assert_allclose(raw_hi[:-2], +1.0, atol=1e-5)
    # Gripper dims: -1 -> 0, +1 -> 1
    np.testing.assert_allclose(raw_lo[-2:], 0.0, atol=1e-5)
    np.testing.assert_allclose(raw_hi[-2:], 1.0, atol=1e-5)


def test_action_passed_into_underlying_env(patched_factory):
    """The raw action that lands on the inner env must match what the
    adapter computed via ``_convert_action_to_raw`` from the agent input."""
    state = patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()

    agent_action = np.full(adapter.action_space.shape, 0.5, dtype=np.float32)
    expected_raw = adapter._convert_action_to_raw(agent_action)
    adapter.step(agent_action)

    np.testing.assert_allclose(state["env"].last_action, expected_raw, atol=1e-5)


def test_action_spec_matches_env_action_dim(patched_factory):
    patched_factory(include_human_pos=False, action_dim=20)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    spec = adapter.action_spec()
    assert spec.shape == (20,)
    assert spec.dtype == np.float32


# ---------------------------------------------------------------------------
# TimeStep step/reset typing + step boundaries
# ---------------------------------------------------------------------------


def test_reset_emits_first_timestep(patched_factory):
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    ts = adapter.reset()
    assert isinstance(ts, TimeStep)
    assert ts.first()
    assert not ts.mid()
    assert not ts.last()
    assert ts.step_type == StepType.FIRST
    assert ts.reward == 0.0
    assert ts.discount == 1.0


def test_mid_step_emits_mid_timestep(patched_factory):
    patched_factory(include_human_pos=False, steps_to_terminal=100)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()

    ts = adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))
    assert ts.mid()
    assert ts.step_type == StepType.MID
    assert ts.reward == 1.0  # stub returns t as reward
    assert ts.discount == 1.0


def test_terminal_step_emits_last_timestep_zero_discount(patched_factory):
    """Inner env terminated -> step_type=LAST, discount=0 (bootstrap kill)."""
    patched_factory(include_human_pos=False, steps_to_terminal=2)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))  # t=1, MID
    last = adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))  # t=2, term
    assert last.last()
    assert last.step_type == StepType.LAST
    assert last.discount == 0.0


def test_episode_length_truncation_forces_last(patched_factory):
    """Adapter caps episodes at ``episode_length // demo_down_sample_rate``.

    With episode_length=4, demo_down_sample_rate=2, the budget is 2 steps;
    on the 2nd step the adapter must force step_type=LAST even though the
    inner env hasn't terminated yet. Discount stays 1.0 because the
    truncation is a time limit, not a terminal state.
    """
    patched_factory(include_human_pos=False, steps_to_terminal=1000)
    cfg = _make_cfg(episode_length=4, demo_down_sample_rate=2)
    adapter = SafetyBiGymCQNAdapter(cfg)
    adapter.reset()
    adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))  # step 1, MID
    truncated = adapter.step(
        np.zeros(adapter.action_space.shape, dtype=np.float32)
    )  # step 2, LAST via budget
    assert truncated.last()
    assert truncated.discount == 1.0


def test_reset_after_terminal_returns_first(patched_factory):
    patched_factory(include_human_pos=False, steps_to_terminal=2)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))
    adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))
    ts = adapter.reset()
    assert ts.first()


# ---------------------------------------------------------------------------
# Safety info plumbing
# ---------------------------------------------------------------------------


def test_step_info_carries_safety_payload(patched_factory):
    patched_factory(include_human_pos=False, steps_to_terminal=5)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    ts = adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))

    assert "safety" in ts.info
    assert "ssm_margin" in ts.info["safety"]
    assert "pfl_force_ratio" in ts.info["safety"]


def test_terminal_step_carries_episode_safety(patched_factory):
    patched_factory(include_human_pos=False, steps_to_terminal=2)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))
    last = adapter.step(np.zeros(adapter.action_space.shape, dtype=np.float32))

    assert last.last()
    assert "episode_safety" in last.info
    assert "ep_ssm_violation_rate" in last.info["episode_safety"]


def test_reset_info_carries_safety_payload(patched_factory):
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    ts = adapter.reset()
    assert "safety" in ts.info


# ---------------------------------------------------------------------------
# Frame stacking
# ---------------------------------------------------------------------------


def test_frame_stack_widens_low_dim_obs(patched_factory):
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=3)
    raw = adapter.low_dim_raw_observation_spec().shape[0]
    stacked = adapter.low_dim_observation_spec().shape[0]
    assert stacked == 3 * raw


def test_frame_stack_initial_fill_repeats_first_frame(patched_factory):
    """On reset, the frame-stack buffer must be primed with copies of the
    first observation, not zeros."""
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=4)
    ts = adapter.reset()
    raw = adapter.low_dim_raw_observation_spec().shape[0]
    assert ts.low_dim_obs.shape == (4 * raw,)
    # The 4 stacked frames at reset should be identical -> repeated copies
    slabs = ts.low_dim_obs.reshape(4, raw)
    for i in range(1, 4):
        np.testing.assert_allclose(slabs[i], slabs[0])


# ---------------------------------------------------------------------------
# Pixels / RGB handling
# ---------------------------------------------------------------------------


def test_pixels_disabled_zero_rgb_placeholder(patched_factory):
    """With pixels=False the adapter must not key into rgb_<cam> obs; the
    emitted rgb_obs is a zero-shaped placeholder of the declared shape."""
    patched_factory(include_human_pos=False, cameras=())  # factory builds no cameras
    cfg = _make_cfg(pixels=False, cameras=())
    adapter = SafetyBiGymCQNAdapter(cfg)
    ts = adapter.reset()
    expected = adapter.rgb_observation_space.shape
    assert ts.rgb_obs.shape == expected
    assert ts.rgb_obs.dtype == np.uint8
    # 0 cameras -> leading axis is 0 in the placeholder shape
    assert expected[0] == 0


def test_pixels_enabled_passes_camera_frames_through(patched_factory):
    patched_factory(include_human_pos=False, cameras=("head",))
    cfg = _make_cfg(pixels=True, cameras=("head",))
    adapter = SafetyBiGymCQNAdapter(cfg, frame_stack=1)
    ts = adapter.reset()
    # Shape: (num_cams, 3 * frame_stack, H, W)
    assert ts.rgb_obs.shape == (1, 3, 84, 84)
    # Stub fills cameras with constant 17
    assert ts.rgb_obs.min() == 17
    assert ts.rgb_obs.max() == 17


# ---------------------------------------------------------------------------
# ExtendedTimeStepWrapper + make()
# ---------------------------------------------------------------------------


def test_extended_timestep_wrapper_attaches_action(patched_factory):
    patched_factory(include_human_pos=False)
    cfg = _make_cfg()
    env = env_adapter.make(cfg)
    assert isinstance(env, ExtendedTimeStepWrapper)

    ts = env.reset()
    assert isinstance(ts, ExtendedTimeStep)
    # reset has no action; wrapper fills with zeros of action-spec shape
    np.testing.assert_allclose(ts.action, 0.0)
    assert ts.action.shape == env.action_spec().shape

    agent_action = np.full(env.action_spec().shape, 0.3, dtype=np.float32)
    ts2 = env.step(agent_action)
    assert isinstance(ts2, ExtendedTimeStep)
    np.testing.assert_allclose(ts2.action, agent_action)


def test_extended_timestep_wrapper_forwards_specs(patched_factory):
    patched_factory(include_human_pos=True)
    cfg = _make_cfg(bodyslam_mode="oracle")
    env = env_adapter.make(cfg)
    # Spec accessors delegate through __getattr__ / explicit forwards
    assert env.low_dim_observation_spec().shape == env._env.low_dim_observation_spec().shape
    assert env.action_spec().shape == env._env.action_spec().shape


def test_timestep_indexable_by_str(patched_factory):
    """CQN-AS's replay buffer indexes TimeStep with strings; preserve that."""
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    ts = adapter.reset()
    assert ts["reward"] == ts.reward
    assert ts["step_type"] == ts.step_type


# ---------------------------------------------------------------------------
# Phase 3 P3.0c: per-step cost attachment
# ---------------------------------------------------------------------------


def test_reset_emits_zero_cost(patched_factory):
    """Reset has no prior dynamics — cost must be 0.0 by convention."""
    patched_factory(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    ts = adapter.reset()
    assert ts.cost == 0.0


def test_step_attaches_cost_from_safety_info(patched_factory):
    """Adapter step() must compute c_t from info['safety'] per env-step.

    Stub env's ssm_margin decreases by 0.05 each step from 0.5. With d_buffer=0.3
    (default), cost stays 0 while margin >= d_buffer, then rises linearly:
    - margin=0.30 (step 4) → c_ssm = max(0, 1 - 1) = 0
    - margin=0.25 (step 5) → c_ssm = 1 - 5/6 = 1/6
    - margin=0.20 (step 6) → c_ssm = 1 - 2/3 = 1/3
    """
    patched_factory(include_human_pos=False, steps_to_terminal=20)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    action = np.zeros(adapter.action_spec().shape, dtype=np.float32)

    costs = []
    for _ in range(8):
        ts = adapter.step(action)
        costs.append(ts.cost)
        if ts.last():
            break

    # First three steps: margin > d_buffer → zero cost
    assert costs[0] == 0.0
    assert costs[1] == 0.0
    assert costs[2] == 0.0
    # Step 4 lands at boundary margin=0.30 → still zero
    assert costs[3] == 0.0
    # Steps 5+ produce strictly positive monotonically-rising cost
    assert costs[4] > 0.0
    assert costs[5] > costs[4]
    assert costs[6] > costs[5]
    # Sanity bound — clipped to [0, 1]
    for c in costs:
        assert 0.0 <= c <= 1.0


def test_cost_per_env_step_not_aggregated_across_chunk(patched_factory):
    """Each env-step emits its own cost — the value changes between consecutive steps.

    This is the load-bearing P3.0c gate: a K-step action chunk that contains a
    cost spike must not have that spike averaged away. Confirmed by verifying
    the per-step cost field changes between consecutive step() calls when the
    underlying ssm_margin changes.
    """
    patched_factory(include_human_pos=False, steps_to_terminal=20)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    action = np.zeros(adapter.action_spec().shape, dtype=np.float32)

    seen_changes = 0
    prev = 0.0
    for _ in range(10):
        ts = adapter.step(action)
        if ts.cost != prev:
            seen_changes += 1
        prev = ts.cost
        if ts.last():
            break
    # At least 2 distinct cost transitions over the 10-step horizon (0 -> nonzero,
    # nonzero -> larger nonzero).
    assert seen_changes >= 2, f"cost barely varied across steps: only {seen_changes} change(s)"


def test_cost_timestep_field_is_indexable_by_string(patched_factory):
    """ReplayBufferStorage.add() reads time_step['cost'] — confirm the shim works."""
    patched_factory(include_human_pos=False, steps_to_terminal=20)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    action = np.zeros(adapter.action_spec().shape, dtype=np.float32)
    ts = adapter.step(action)
    assert ts["cost"] == ts.cost


def test_extended_timestep_carries_cost(patched_factory):
    """ExtendedTimeStepWrapper must forward the cost field unchanged."""
    patched_factory(include_human_pos=False, steps_to_terminal=20)
    cfg = _make_cfg()
    env = env_adapter.make(cfg)
    env.reset()
    action = np.full(env.action_spec().shape, 0.0, dtype=np.float32)
    # Step a few times to build up nonzero cost
    last_ts = None
    for _ in range(6):
        last_ts = env.step(action)
    assert isinstance(last_ts, ExtendedTimeStep)
    assert hasattr(last_ts, "cost")
    assert last_ts["cost"] == last_ts.cost


def test_cost_zero_when_safety_info_absent(patched_factory):
    """If info lacks 'safety' (e.g. some wrapper drops it), cost gracefully defaults to 0."""
    patched_factory(include_human_pos=False, steps_to_terminal=20)
    adapter = SafetyBiGymCQNAdapter(_make_cfg())
    adapter.reset()
    action = np.zeros(adapter.action_spec().shape, dtype=np.float32)
    # Monkeypatch the underlying stub env to emit info without 'safety'.
    real_step = adapter._env.step

    def _stripped_step(act):
        obs, reward, terminated, truncated, info = real_step(act)
        info.pop("safety", None)
        return obs, reward, terminated, truncated, info

    adapter._env.step = _stripped_step
    ts = adapter.step(action)
    assert ts.cost == 0.0
