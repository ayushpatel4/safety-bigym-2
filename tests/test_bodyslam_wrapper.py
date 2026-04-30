"""Tests for BodySLAMWrapper — Phase 1 mock perception layer.

The wrapper adds a noisy human-pose estimate to the env's observation dict
under the key ``human_pos_estimate`` (shape (6,)):
    [x, y, z, occluded, staleness, confidence]

Most tests use a stub env that scripts ``info["safety"]["human_pos"]`` per
step; integration tests with real MuJoCo are skipped when AMASS is unset.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Callable, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


HAS_AMASS = bool(os.environ.get("AMASS_DATA_DIR"))


# ---------------------------------------------------------------------------
# Stub env: emits a scripted sequence of human positions.
# ---------------------------------------------------------------------------


class _StubHumanEnv(gym.Env):
    """Minimal env that scripts ``info["safety"]["human_pos"]`` per step.

    The base obs is a tiny low-dim Dict so the wrapper has somewhere to graft
    the new key.
    """

    metadata: dict = {}

    def __init__(
        self,
        positions: Sequence[Sequence[float]],
        occluded_flags: Optional[Sequence[bool]] = None,
        emit_safety_info: bool = True,
    ):
        self._positions = [np.asarray(p, dtype=np.float32) for p in positions]
        self._occluded = (
            list(occluded_flags) if occluded_flags is not None else [False] * len(self._positions)
        )
        self._emit_safety = emit_safety_info
        self._t = 0
        self.observation_space = spaces.Dict(
            {
                "low_dim_state": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        obs = {"low_dim_state": np.zeros((2,), dtype=np.float32)}
        info = {}
        if self._emit_safety:
            info["safety"] = {
                "human_pos": self._positions[0].tolist(),
                "occluded": self._occluded[0],
            }
        return obs, info

    def step(self, action):
        # Advance time first, then read this step's scripted info.
        self._t += 1
        idx = min(self._t, len(self._positions) - 1)
        info: dict = {}
        if self._emit_safety:
            info["safety"] = {
                "human_pos": self._positions[idx].tolist(),
                "occluded": self._occluded[idx],
            }
        terminated = self._t >= len(self._positions) - 1
        obs = {"low_dim_state": np.zeros((2,), dtype=np.float32)}
        return obs, 0.0, terminated, False, info


def _scripted_occlusion(env, info) -> bool:
    """Occlusion strategy that reads from scripted info."""
    safety = info.get("safety", {}) or {}
    return bool(safety.get("occluded", False))


def _no_occlusion(env, info) -> bool:
    return False


def _make_wrapper(
    positions,
    *,
    mode: str = "noisy",
    ou_alpha: float = 0.9,
    noise_std: float = 0.05,
    latency_steps: int = 3,
    occlusion_noise_mult: float = 3.0,
    dropout_prob: float = 0.02,
    seed: int = 0,
    occluded_flags=None,
    occlusion_fn: Callable = _scripted_occlusion,
    position_provider: Optional[Callable[[int], np.ndarray]] = None,
    demo_replay: bool = False,
    emit_safety_info: bool = True,
):
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper

    inner = _StubHumanEnv(
        positions,
        occluded_flags=occluded_flags,
        emit_safety_info=emit_safety_info,
    )
    return BodySLAMWrapper(
        inner,
        mode=mode,
        ou_alpha=ou_alpha,
        noise_std=noise_std,
        latency_steps=latency_steps,
        occlusion_noise_mult=occlusion_noise_mult,
        dropout_prob=dropout_prob,
        seed=seed,
        occlusion_fn=occlusion_fn,
        position_provider=position_provider,
        demo_replay=demo_replay,
    )


def _roll(env, n_steps: int) -> List[np.ndarray]:
    out = []
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        # Wrapper places key in obs; grab from last obs via wrapper accessor
        out.append(np.array(env.last_estimate, dtype=np.float64))
        if terminated or truncated:
            break
    return out


# ---------------------------------------------------------------------------
# Unit tests (no MuJoCo)
# ---------------------------------------------------------------------------


def test_obs_space_extended():
    env = _make_wrapper([[1.0, 0.0, 0.0]] * 3)
    assert "human_pos_estimate" in env.observation_space.spaces
    box = env.observation_space.spaces["human_pos_estimate"]
    assert isinstance(box, spaces.Box)
    assert box.shape == (6,)


def test_oracle_mode_returns_clean_pos():
    positions = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]
    env = _make_wrapper(positions, mode="oracle", seed=0)
    obs, _ = env.reset()
    assert np.allclose(obs["human_pos_estimate"][:3], [1.0, 2.0, 3.0])
    assert obs["human_pos_estimate"][3] == 0.0  # occluded
    assert obs["human_pos_estimate"][4] == 0.0  # staleness
    assert obs["human_pos_estimate"][5] == 1.0  # confidence

    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert np.allclose(obs2["human_pos_estimate"][:3], [1.5, 2.5, 3.5])


def test_off_mode_skipped_at_factory_level():
    """mode='off' is enforced by the factory (not inserting the wrapper).
    The wrapper itself doesn't accept mode='off'; supplying it raises.
    """
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper

    inner = _StubHumanEnv([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        BodySLAMWrapper(inner, mode="off")


def test_noise_seeded_deterministic():
    positions = [[1.0, 0.0, 0.0]] * 50
    env_a = _make_wrapper(positions, seed=42, dropout_prob=0.1)
    env_b = _make_wrapper(positions, seed=42, dropout_prob=0.1)
    env_a.reset()
    env_b.reset()
    seq_a = _roll(env_a, 50)
    seq_b = _roll(env_b, 50)
    for a, b in zip(seq_a, seq_b):
        assert np.allclose(a, b), f"diverged: {a} vs {b}"


def test_ou_temporal_correlation():
    positions = [[1.0, 0.0, 0.0]] * 250
    env = _make_wrapper(
        positions, ou_alpha=0.9, noise_std=0.05, latency_steps=1, dropout_prob=0.0, seed=0
    )
    env.reset()
    seq = np.stack(_roll(env, 200))
    residual = seq[:, 0] - 1.0  # x-component residual
    # Lag-1 autocorrelation
    r = np.corrcoef(residual[:-1], residual[1:])[0, 1]
    assert r > 0.7, f"OU should give correlated noise, got autocorr={r:.3f}"


def test_iid_baseline_for_comparison():
    positions = [[1.0, 0.0, 0.0]] * 250
    env = _make_wrapper(
        positions, ou_alpha=0.0, noise_std=0.05, latency_steps=1, dropout_prob=0.0, seed=0
    )
    env.reset()
    seq = np.stack(_roll(env, 200))
    residual = seq[:, 0] - 1.0
    r = np.corrcoef(residual[:-1], residual[1:])[0, 1]
    assert abs(r) < 0.2, f"alpha=0 should be ~i.i.d., got autocorr={r:.3f}"


def test_noise_std_calibration():
    positions = [[1.0, 0.0, 0.0]] * 1100
    env = _make_wrapper(
        positions, ou_alpha=0.9, noise_std=0.05, latency_steps=1, dropout_prob=0.0, seed=0
    )
    env.reset()
    seq = np.stack(_roll(env, 1050))
    # Burn-in 50 steps for OU stationarity
    residual = seq[50:, :3] - np.array([1.0, 0.0, 0.0])
    sample_std = residual.std(axis=0).mean()
    # Stationary OU std with alpha=0.9, sigma=0.05: sigma_eff = sigma / sqrt(1-a^2)
    # = 0.05 / sqrt(0.19) ≈ 0.1147
    expected = 0.05 / np.sqrt(1 - 0.9**2)
    assert 0.7 * expected < sample_std < 1.3 * expected, (
        f"sample_std={sample_std:.4f}, expected≈{expected:.4f}"
    )


def test_latency_buffer_lag():
    # Indices 0..4: pos=0; indices 5+: pos=1. With advance-first stub, the
    # first env.step reads positions[1]; jump is first observed at step 5
    # (which reads positions[5]).
    positions = [[0.0, 0.0, 0.0]] * 5 + [[1.0, 0.0, 0.0]] * 20
    env = _make_wrapper(
        # alpha=0.5 makes OU convergence within a few steps so we can
        # disentangle latency-buffer lag from OU low-pass.
        positions, ou_alpha=0.5, noise_std=0.0, latency_steps=3, dropout_prob=0.0, seed=0
    )
    env.reset()
    seq = np.stack(_roll(env, 20))
    # seq[i] is the emit at the (i+1)-th env.step call. With latency_steps=3
    # the emit is OU 3 steps stale.
    # Step 4 reads positions[4]=0 → emit≈0. Buffer hasn't seen jump yet.
    assert seq[3, 0] < 0.2, f"pre-jump should emit ~0, got {seq[3,0]:.3f}"
    # By step 12 (8 steps after the jump, buffer has flushed), emit ≈ 1.
    assert seq[11, 0] > 0.5, f"post-jump should catch up, got {seq[11,0]:.3f}"


def test_dropout_repeats_last_known_and_increments_staleness():
    positions = [[float(i), 0.0, 0.0] for i in range(20)]
    env = _make_wrapper(
        positions, ou_alpha=1.0, noise_std=0.0, latency_steps=1, dropout_prob=1.0, seed=0
    )
    env.reset()
    seq = np.stack(_roll(env, 15))
    # All emits should equal the initial position (no fresh emits ever).
    for i, row in enumerate(seq):
        assert np.allclose(row[:3], [0.0, 0.0, 0.0]), f"step {i}: {row}"
    # Staleness: 1, 2, 3, ...
    assert seq[0, 4] == 1.0
    assert seq[5, 4] == 6.0


def test_dropout_prob_zero_means_zero_staleness():
    positions = [[1.0, 0.0, 0.0]] * 30
    env = _make_wrapper(
        positions, ou_alpha=0.9, noise_std=0.05, dropout_prob=0.0, seed=0
    )
    env.reset()
    seq = np.stack(_roll(env, 25))
    assert np.all(seq[:, 4] == 0.0), f"non-zero staleness with p=0: {seq[:, 4]}"


def test_reset_resets_state():
    positions = [[1.0, 0.0, 0.0]] * 20
    env = _make_wrapper(
        positions, ou_alpha=0.9, noise_std=0.05, dropout_prob=1.0, seed=0
    )
    env.reset()
    _roll(env, 10)
    # Now staleness should be ≥ 9 inside the wrapper.
    pre = float(env.last_estimate[4])
    assert pre >= 9.0
    obs, _ = env.reset()
    assert obs["human_pos_estimate"][4] == 0.0  # staleness back to 0
    assert obs["human_pos_estimate"][5] == 1.0  # confidence back to 1


def test_seeding_propagates_from_scenario():
    """If env.unwrapped exposes ``_current_scenario.seed``, the wrapper uses
    it (per-episode determinism); else it falls back to constructor seed."""
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper

    class _ScenarioStubEnv(_StubHumanEnv):
        def __init__(self, positions, scen_seed):
            super().__init__(positions)
            self._current_scenario = type("S", (), {"seed": scen_seed})()

    inner_a = _ScenarioStubEnv([[1.0, 0.0, 0.0]] * 30, scen_seed=111)
    inner_b = _ScenarioStubEnv([[1.0, 0.0, 0.0]] * 30, scen_seed=222)
    wa = BodySLAMWrapper(
        inner_a, mode="noisy", noise_std=0.05, dropout_prob=0.0, seed=0
    )
    wb = BodySLAMWrapper(
        inner_b, mode="noisy", noise_std=0.05, dropout_prob=0.0, seed=0
    )
    wa.reset()
    wb.reset()
    seq_a = _roll(wa, 25)
    seq_b = _roll(wb, 25)
    diffs = [np.linalg.norm(a[:3] - b[:3]) for a, b in zip(seq_a, seq_b)]
    assert max(diffs) > 0.01, "different scenario seeds produced identical noise"


def test_dropout_recovery_no_discontinuity():
    """OU state should keep updating during dropout. After 10 dropped steps,
    the first recovered emit must not jump > 2σ from the last emit.
    """
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper

    positions = [[1.0, 0.0, 0.0]] * 30
    inner = _StubHumanEnv(positions)
    # Custom dropout schedule: drop steps 5-14, then resume.
    dropout_schedule = [False] * 5 + [True] * 10 + [False] * 15
    sched_iter = iter(dropout_schedule)

    def scheduled_uniform(self):
        # Hook into wrapper's RNG via a scripted draw
        try:
            return 0.0 if next(sched_iter) else 1.0
        except StopIteration:
            return 1.0

    w = BodySLAMWrapper(
        inner, mode="noisy", noise_std=0.05, latency_steps=1,
        dropout_prob=0.5, seed=0, occlusion_fn=_no_occlusion,
    )
    # Override the dropout decision with a scripted schedule.
    sched_q = deque(dropout_schedule)
    w._force_dropout_schedule = sched_q  # implementation hook
    w.reset()
    seq = np.stack(_roll(w, 30))
    # Steps 5-14: emits should equal step-4's emit (last fresh).
    for i in range(5, 15):
        assert np.allclose(seq[i, :3], seq[4, :3]), (
            f"during dropout step {i} differs from step 4"
        )
    # Step 15: first recovered emit. Should be within 2σ_stat of last emit.
    sigma_stat = 0.05 / np.sqrt(1 - 0.9**2)
    delta = np.linalg.norm(seq[15, :3] - seq[4, :3])
    # Both seq[15] and seq[4] are stationary OU samples around μ=1.
    # Expected diff ≈ σ_stat * sqrt(6) ≈ 0.28; allow up to 5*σ_stat.
    assert delta < 5 * sigma_stat, (
        f"discontinuity at recovery: jumped {delta:.3f}m > {5*sigma_stat:.3f}"
    )


def test_confidence_derivation():
    # Build a sequence where we can hit four corners.
    # Step 0: clear & fresh -> conf=1
    # Step 1: occluded & fresh (force occluded via flag) -> conf=0.5
    # Then drop 5 steps with occluded flag mixed -> staleness builds.
    positions = [[1.0, 0.0, 0.0]] * 12
    occ_flags = [False, True, False, True, False, True, False, False, False, False, False, False]
    env = _make_wrapper(
        positions,
        ou_alpha=1.0,
        noise_std=0.0,
        dropout_prob=0.0,
        occluded_flags=occ_flags,
        seed=0,
    )
    env.reset()
    obs0 = env.last_estimate.copy()
    assert obs0[5] == pytest.approx(1.0)  # clear, fresh

    obs1, *_ = env.step(env.action_space.sample())
    e1 = obs1["human_pos_estimate"]
    # occluded=1, staleness=0 -> conf = (1 - 0.5)*(1) = 0.5
    assert e1[3] == 1.0
    assert e1[4] == 0.0
    assert e1[5] == pytest.approx(0.5)


def test_demo_replay_mode_drives_from_position_provider():
    """In demo_replay mode the wrapper reads positions from a provider
    (synthetic AMASS playback), not from info["safety"]."""
    # Provider returns a moving target so we can verify it advances.
    def provider(step_idx):
        return np.array([0.1 * step_idx, 0.0, 0.0], dtype=np.float32)

    # Stub env emits NO safety info — wrapper must fall back to provider.
    env = _make_wrapper(
        [[0.0, 0.0, 0.0]] * 20,  # positions are ignored when emit_safety_info=False
        emit_safety_info=False,
        mode="oracle",  # use oracle so we read provider directly
        position_provider=provider,
        demo_replay=True,
        seed=0,
    )
    obs, _ = env.reset()
    assert "human_pos_estimate" in obs
    seq = []
    for _ in range(10):
        o, *_ = env.step(env.action_space.sample())
        seq.append(o["human_pos_estimate"][:3].copy())
    seq = np.stack(seq)
    # x should advance from 0.1 to 1.0 (monotone, stepping 0.1 each)
    assert seq[0, 0] < seq[-1, 0] - 0.5


def test_long_episode_no_drift():
    positions = [[1.0, 0.0, 0.0]] * 1010
    env = _make_wrapper(
        positions, ou_alpha=0.9, noise_std=0.05, dropout_prob=0.05, seed=0
    )
    env.reset()
    seq = np.stack(_roll(env, 1000))
    # At step 500 and 999, position residual should be bounded.
    sigma_stat = 0.05 / np.sqrt(1 - 0.9**2)
    for step in (500, 999):
        delta = np.linalg.norm(seq[step, :3] - np.array([1.0, 0.0, 0.0]))
        assert delta < 5 * sigma_stat, f"step {step}: drift {delta:.3f}"
        # Staleness should not run away with p=0.05
        assert seq[step, 4] < 50, f"step {step}: staleness {seq[step, 4]}"


def test_noisy_mode_never_nans_or_infs():
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2, 2, size=(5005, 3)).tolist()
    env = _make_wrapper(
        positions, ou_alpha=0.9, noise_std=0.05, dropout_prob=0.05, seed=7
    )
    env.reset()
    seq = np.stack(_roll(env, 5000))
    assert np.all(np.isfinite(seq)), "non-finite values in obs sequence"


# ---------------------------------------------------------------------------
# Integration tests (real MuJoCo / minimal MJCF, AMASS-skip-guarded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_AMASS, reason="AMASS_DATA_DIR not set")
def test_factory_inserts_wrapper_when_enabled():
    """Build SafetyBiGymEnvFactory with bodyslam.mode='noisy' and check that
    BodySLAMWrapper is in the wrapper chain.
    """
    from omegaconf import OmegaConf
    from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper

    cfg = OmegaConf.create(
        {
            "pixels": False,
            "visual_observation_shape": [84, 84],
            "env": {
                "task_name": "reach_target_single",
                "demo_down_sample_rate": 20,
                "render_mode": "rgb_array",
                "enable_all_floating_dof": False,
                "action_mode": "absolute",
                "cameras": [],
                "inject_human": True,
                "motion_clip_dir": os.environ["AMASS_DATA_DIR"],
                "motion_clip_paths": ["74/74_01_poses.npz"],
                "bodyslam": {"mode": "noisy"},
            },
        }
    )
    factory = SafetyBiGymEnvFactory()
    env = factory._create_env(cfg)

    # Walk the wrapper chain.
    found = False
    cur = env
    while hasattr(cur, "env"):
        if isinstance(cur, BodySLAMWrapper):
            found = True
            break
        cur = cur.env
    if not found and isinstance(cur, BodySLAMWrapper):
        found = True

    assert found, "BodySLAMWrapper not in chain when bodyslam.mode='noisy'"
    assert "human_pos_estimate" in env.observation_space.spaces
    env.close()


def test_occlusion_flag_set_when_geom_blocks():
    """Minimal MJCF: camera + target body + a wall geom between them.
    Wall present -> occluded=1; wall absent -> occluded=0.
    """
    import mujoco
    from safety_bigym.perception.bodyslam_wrapper import MujocoRayOcclusion

    xml_with_wall = """
    <mujoco>
      <worldbody>
        <camera name="head" pos="0 0 0" euler="0 90 0"/>
        <body name="target">
          <geom name="target_geom" type="sphere" size="0.05" pos="2 0 0"/>
        </body>
        <body name="wall_body">
          <geom name="wall" type="box" size="0.02 0.5 0.5" pos="1 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    xml_no_wall = """
    <mujoco>
      <worldbody>
        <camera name="head" pos="0 0 0" euler="0 90 0"/>
        <body name="target">
          <geom name="target_geom" type="sphere" size="0.05" pos="2 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    for xml, expected in [(xml_with_wall, True), (xml_no_wall, False)]:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        target_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom"
        )
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head")
        checker = MujocoRayOcclusion(
            model=model,
            data=data,
            camera_id=cam_id,
            target_geom_id=target_geom_id,
        )
        assert checker.is_occluded() == expected, (
            f"expected occluded={expected} for xml block"
        )
