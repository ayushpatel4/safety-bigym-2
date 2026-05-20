"""Unit tests for the CQN-AS demo pipeline (Workstream D).

Covers ``SafetyBiGymCQNAdapter.get_demos`` and its ported helpers
(``_convert_demo_to_timesteps`` / ``_extract_action_stats`` /
``_rescale_demo_actions`` / ``_inject_human_pos_into_demos``) without booting
MuJoCo or loading AMASS:

- ``SafetyBiGymEnvFactory._create_env`` is monkeypatched to a stub gym env
  (so the adapter constructs).
- ``SafetyBiGymEnvFactory._get_demo_fn`` is monkeypatched to return synthetic
  ``Demo``-like objects (so DemoStore / BiGym never run).
- ``AMASSDemoPositionProvider`` is monkeypatched to a fixed-vector stub for the
  bodyslam-on case (so no AMASS clips are read).

Asserts the acceptance criteria from PHASE3_DEMO_PIPELINE_HANDOFF.md:
- get_demos returns list[list[ExtendedTimeStep]]
- demo low_dim width matches live width for off / oracle (6D channel)
- actions land in [-1, 1] (gripper tail respected)
- post-first-reward truncation; last timestep typed LAST; FIRST on step 0
- every demo timestep carries cost == 0.0 and the demo success flag
- self._action_stats is overridden by demo-derived stats
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf

from safety_bigym.agents.cqn_as import env_adapter
from safety_bigym.agents.cqn_as.env_adapter import (
    ExtendedTimeStep,
    SafetyBiGymCQNAdapter,
)
from safety_bigym.perception.bodyslam_wrapper import OBS_KEY as BODYSLAM_OBS_KEY

try:
    from dm_env import StepType
except ImportError:  # pragma: no cover - dm_env is required by the adapter
    pytest.skip("dm_env not installed", allow_module_level=True)


PROPRIO_WIDTH = 60
GRIPPER_WIDTH = 2
FLOATING_BASE_WIDTH = 4
ACTION_DIM = 16  # 14 body joints + 2 grippers
LOW_DIM = PROPRIO_WIDTH + GRIPPER_WIDTH + FLOATING_BASE_WIDTH


class _StubSafetyBiGymEnv(gym.Env):
    """Gym env shaped like ``SafetyBiGymEnvFactory._create_env`` output."""

    metadata: dict = {}

    def __init__(self, *, include_human_pos: bool):
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
        self.observation_space = spaces.Dict(obs_dict)

        low = -np.ones(ACTION_DIM, dtype=np.float32)
        high = np.ones(ACTION_DIM, dtype=np.float32)
        low[-2:] = 0.0
        high[-2:] = 1.0
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {"safety": {}}

    def step(self, action):
        return self._obs(), 0.0, False, False, {"safety": {}}

    def _obs(self):
        obs = {
            "proprioception": np.zeros(PROPRIO_WIDTH, np.float32),
            "proprioception_grippers": np.full(GRIPPER_WIDTH, 0.5, np.float32),
            "proprioception_floating_base": np.zeros(FLOATING_BASE_WIDTH, np.float32),
        }
        if BODYSLAM_OBS_KEY in self.observation_space.spaces:
            obs[BODYSLAM_OBS_KEY] = np.zeros(6, np.float32)
        return obs

    def close(self):
        pass


class _FakeProvider:
    """Stub AMASSDemoPositionProvider — fixed pelvis position, no AMASS."""

    def __init__(self, *args, **kwargs):
        self._t = 0

    def reset(self):
        self._t = 0

    def __call__(self, step_idx):
        return np.array([1.0, 2.0, 0.5], dtype=np.float32)


def _make_demostep(t: int, *, reward: float, include_rgb: bool = False):
    """A mutable raw demostep with the BiGym demo schema the adapter reads."""
    obs = {
        "proprioception": np.full(PROPRIO_WIDTH, 0.01 * t, np.float32),
        "proprioception_grippers": np.full(GRIPPER_WIDTH, 0.5, np.float32),
        "proprioception_floating_base": np.full(FLOATING_BASE_WIDTH, 0.1, np.float32),
    }
    # Deterministic in-range action: body dims in [-0.8, 0.8], grippers in [0,1].
    rng = np.random.default_rng(t)
    action = np.empty(ACTION_DIM, dtype=np.float32)
    action[:-2] = rng.uniform(-0.8, 0.8, size=ACTION_DIM - 2)
    action[-2:] = rng.uniform(0.0, 1.0, size=2)
    return SimpleNamespace(
        observation=obs,
        reward=float(reward),
        info={"demo_action": action, "task_success": reward > 0},
        termination=False,
        truncation=False,
    )


def _make_demo(n_steps: int = 10, reward_at: int = 5):
    steps = [
        _make_demostep(t, reward=(1.0 if t == reward_at else 0.0))
        for t in range(n_steps)
    ]
    return SimpleNamespace(timesteps=steps)


def _make_cfg(*, bodyslam_mode: str = "off"):
    env = {
        "task_name": "saucepan_to_hob",
        "bodyslam": {"mode": bodyslam_mode},
        "cameras": [],
        "episode_length": 200,
        "demo_down_sample_rate": 20,
    }
    if bodyslam_mode != "off":
        env["motion_clip_dir"] = "/tmp/fake_amass"
        env["motion_clip_paths"] = ["74/74_01_poses.npz"]
    return OmegaConf.create({"pixels": False, "visual_observation_shape": [84, 84], "env": env})


@pytest.fixture
def patched(monkeypatch):
    """Patch _create_env (stub gym env) and _get_demo_fn (synthetic demos)."""

    def install(*, include_human_pos: bool, n_demos: int = 3, n_steps: int = 10,
                reward_at: int = 5):
        def _fake_create_env(self, cfg):
            return _StubSafetyBiGymEnv(include_human_pos=include_human_pos)

        def _fake_get_demo_fn(self, cfg, num_demos):
            return [_make_demo(n_steps, reward_at) for _ in range(num_demos)]

        monkeypatch.setattr(
            "safety_bigym.agents.cqn_as.env_adapter.SafetyBiGymEnvFactory._create_env",
            _fake_create_env, raising=True,
        )
        monkeypatch.setattr(
            "safety_bigym.agents.cqn_as.env_adapter.SafetyBiGymEnvFactory._get_demo_fn",
            _fake_get_demo_fn, raising=True,
        )
        monkeypatch.setattr(
            "safety_bigym.agents.cqn_as.env_adapter.AMASSDemoPositionProvider",
            _FakeProvider, raising=True,
        )
        return dict(n_demos=n_demos, n_steps=n_steps, reward_at=reward_at)

    return install


# ---------------------------------------------------------------------------
# Shape / structure
# ---------------------------------------------------------------------------


def test_get_demos_returns_list_of_lists(patched):
    patched(include_human_pos=False, n_demos=3)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=4)
    demos = adapter.get_demos(3)
    assert isinstance(demos, list) and len(demos) == 3
    for demo in demos:
        assert isinstance(demo, list) and len(demo) > 0
        assert all(isinstance(ts, ExtendedTimeStep) for ts in demo)


def test_low_dim_width_matches_live_without_bodyslam(patched):
    patched(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(bodyslam_mode="off"), frame_stack=4)
    demos = adapter.get_demos(2)
    raw = adapter.low_dim_raw_observation_spec().shape[0]
    stacked = adapter.low_dim_observation_spec().shape[0]
    assert raw == LOW_DIM
    assert demos[0][0].low_dim_obs.shape == (stacked,)
    assert stacked == 4 * LOW_DIM


def test_low_dim_width_matches_live_with_bodyslam(patched):
    patched(include_human_pos=True)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(bodyslam_mode="oracle"), frame_stack=4)
    demos = adapter.get_demos(2)
    raw = adapter.low_dim_raw_observation_spec().shape[0]
    assert raw == LOW_DIM + 6  # 6D human_pos_estimate channel
    assert demos[0][0].low_dim_obs.shape == (4 * (LOW_DIM + 6),)


def test_bodyslam_oracle_injects_clean_human_pos(patched):
    """Oracle demo replay emits clean [x, y, z, 0, 0, 1] in the last 6 dims."""
    patched(include_human_pos=True)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(bodyslam_mode="oracle"), frame_stack=1)
    demos = adapter.get_demos(1)
    tail = demos[0][1].low_dim_obs[-6:]  # step 1 (post-reset), single frame
    np.testing.assert_allclose(tail[:3], [1.0, 2.0, 0.5], atol=1e-5)
    assert tail[3] == 0.0          # not occluded
    assert tail[4] == 0.0          # staleness
    assert tail[5] == pytest.approx(1.0)  # confidence


# ---------------------------------------------------------------------------
# Step typing + truncation
# ---------------------------------------------------------------------------


def test_post_first_reward_truncation_and_step_types(patched):
    patched(include_human_pos=False, n_steps=10, reward_at=5)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=1)
    demos = adapter.get_demos(1)
    demo = demos[0]
    # Truncated at the first rewarding state (index 5) -> 6 timesteps.
    assert len(demo) == 6
    assert demo[0].step_type == StepType.FIRST
    assert all(ts.step_type == StepType.MID for ts in demo[1:-1])
    assert demo[-1].step_type == StepType.LAST
    assert demo[-1].reward == 1.0
    # Timelimit truncation (not termination) -> discount stays 1.0
    assert demo[-1].discount == 1.0


def test_demo_success_flag_set(patched):
    patched(include_human_pos=False, reward_at=5)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=1)
    demos = adapter.get_demos(1)
    # sum reward = 1.0 > 0.25 -> successful
    assert all(ts.demo == 1 for ts in demos[0])


# ---------------------------------------------------------------------------
# Action normalisation + stats
# ---------------------------------------------------------------------------


def test_demo_actions_in_unit_range(patched):
    patched(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=1)
    demos = adapter.get_demos(3)
    for demo in demos:
        for ts in demo:
            assert ts.action.shape == (ACTION_DIM,)
            assert np.all(ts.action >= -1.0 - 1e-5)
            assert np.all(ts.action <= 1.0 + 1e-5)


def test_action_stats_overridden_by_demos(patched):
    patched(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=1)
    default_min = adapter._action_stats["min"].copy()
    adapter.get_demos(3)
    # Demo-derived body-dim stats differ from the identity [-1, 1] default.
    assert not np.allclose(adapter._action_stats["min"][:-2], default_min[:-2])
    # Gripper tail stays hard-coded to [0, 1].
    np.testing.assert_allclose(adapter._action_stats["min"][-2:], 0.0)
    np.testing.assert_allclose(adapter._action_stats["max"][-2:], 1.0)


# ---------------------------------------------------------------------------
# Phase 3 cost field
# ---------------------------------------------------------------------------


def test_demo_timesteps_carry_zero_cost(patched):
    patched(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=1)
    demos = adapter.get_demos(2)
    for demo in demos:
        for ts in demo:
            assert ts.cost == 0.0
            assert ts["cost"] == 0.0  # string-indexable for ReplayBufferStorage.add


def test_get_demos_zero_raises(patched):
    patched(include_human_pos=False)
    adapter = SafetyBiGymCQNAdapter(_make_cfg(), frame_stack=1)
    with pytest.raises(ValueError):
        adapter.get_demos(0)
