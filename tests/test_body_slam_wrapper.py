"""Unit tests for BodySLAMWrapper (Phase 1).

The wrapper ingests the full tracked SMPL-H skeleton (exposed under
info['safety']['human_joint_positions'] + names) and produces a noisy,
latency-delayed, occasionally-lost observation under three new keys in the
Dict observation space:
  - human_state_estimate    (N, 3) float32
  - human_state_occluded    (N,)   float32, 1.0 iff that joint is occluded
  - human_state_staleness   (1,)   float32, seconds since last fresh estimate

Tests use a stub env so we don't pay MuJoCo startup cost per test — the
occlusion path is covered separately in test_ray_cast.py.
"""

from typing import List

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces


NUM_JOINTS = 3  # small for test speed; real env uses 18


def _joint_array(value: float) -> List[List[float]]:
    """All joints at the same (value, 0, 0) — constant truth for delay tests."""
    return [[float(value), 0.0, 0.0] for _ in range(NUM_JOINTS)]


class _StubEnv(gym.Env):
    """Minimal env emitting scripted info['safety']['human_joint_positions'].

    If `positions` is a callable it receives the step counter and returns the
    per-step joint array; otherwise it cycles a fixed list."""

    metadata: dict = {}

    def __init__(self, positions):
        self.observation_space = spaces.Dict({
            "proprioception": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._positions = positions
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return {"proprioception": np.zeros(4, dtype=np.float32)}, {}

    def step(self, action):
        if callable(self._positions):
            pos = self._positions(self._t)
        else:
            pos = self._positions[self._t % len(self._positions)]
        self._t += 1
        info = {
            "safety": {
                "human_joint_positions": pos,
                "human_joint_names": [f"j{i}" for i in range(NUM_JOINTS)],
            }
        }
        obs = {"proprioception": np.zeros(4, dtype=np.float32)}
        return obs, 0.0, False, False, info


def _make_wrapper(
    positions,
    *,
    mode="noisy",
    sigma=0.05,
    alpha=0.9,
    latency_steps=0,
    use_occlusion=False,
    occlusion_multiplier=3.0,
    dropout_prob=0.0,
    dt=0.02,
    seed=0,
):
    from safety_bigym import BodySLAMConfig
    from safety_bigym.filters.body_slam_wrapper import BodySLAMWrapper

    cfg = BodySLAMConfig(
        mode=mode,
        sigma=sigma,
        alpha=alpha,
        latency_steps=latency_steps,
        use_occlusion=use_occlusion,
        occlusion_multiplier=occlusion_multiplier,
        dropout_prob=dropout_prob,
        dt=dt,
    )
    env = _StubEnv(positions)
    wrapped = BodySLAMWrapper(env, config=cfg, num_joints=NUM_JOINTS, seed=seed)
    wrapped.reset(seed=seed)
    return wrapped


# ---------------------------------------------------------------------------
# Observation-space shape tests
# ---------------------------------------------------------------------------


class TestObservationSpace:
    def test_human_state_estimate_key_added(self):
        env = _make_wrapper([_joint_array(1.0)])
        space = env.observation_space.spaces
        assert "human_state_estimate" in space
        box = space["human_state_estimate"]
        assert box.shape == (NUM_JOINTS, 3)
        assert box.dtype == np.float32

    def test_occluded_flag_key_added(self):
        env = _make_wrapper([_joint_array(1.0)])
        box = env.observation_space.spaces["human_state_occluded"]
        assert box.shape == (NUM_JOINTS,)
        assert box.dtype == np.float32

    def test_staleness_key_added(self):
        env = _make_wrapper([_joint_array(1.0)])
        box = env.observation_space.spaces["human_state_staleness"]
        assert box.shape == (1,)
        assert box.dtype == np.float32

    def test_base_keys_preserved(self):
        env = _make_wrapper([_joint_array(1.0)])
        assert "proprioception" in env.observation_space.spaces

    def test_reset_observation_has_all_new_keys(self):
        env = _make_wrapper([_joint_array(1.0)])
        obs, _ = env.reset()
        for k in ("human_state_estimate", "human_state_occluded",
                  "human_state_staleness", "proprioception"):
            assert k in obs


# ---------------------------------------------------------------------------
# Noise statistics tests
# ---------------------------------------------------------------------------


class TestNoiseStatistics:
    def test_estimate_is_unbiased(self):
        env = _make_wrapper(
            lambda t: _joint_array(1.0),
            sigma=0.05, alpha=0.0, latency_steps=0, seed=0,
        )
        residuals = []
        for _ in range(3000):
            obs, *_ = env.step(env.action_space.sample())
            est = obs["human_state_estimate"]
            residuals.append(est - np.array(_joint_array(1.0), dtype=np.float32))
        residuals = np.stack(residuals)
        mean_err = residuals.mean()
        # Standard error of the mean ≈ sigma / sqrt(N * joints * dims)
        tol = 0.05 / np.sqrt(3000 * NUM_JOINTS * 3) * 4  # 4-sigma band
        assert abs(mean_err) < tol, f"bias {mean_err:.4f} too large (tol={tol:.4f})"

    def test_std_matches_sigma_with_alpha_zero(self):
        env = _make_wrapper(
            lambda t: _joint_array(0.0),
            sigma=0.1, alpha=0.0, latency_steps=0, seed=0,
        )
        samples = []
        for _ in range(3000):
            obs, *_ = env.step(env.action_space.sample())
            samples.append(obs["human_state_estimate"][0, 0])
        std = float(np.std(samples))
        assert std == pytest.approx(0.1, rel=0.15)

    def test_temporal_correlation_with_alpha(self):
        env = _make_wrapper(
            lambda t: _joint_array(0.0),
            sigma=0.1, alpha=0.9, latency_steps=0, seed=0,
        )
        # Warm-up past the OU transient.
        for _ in range(500):
            env.step(env.action_space.sample())
        samples = []
        for _ in range(5000):
            obs, *_ = env.step(env.action_space.sample())
            samples.append(obs["human_state_estimate"][0, 0])
        samples = np.array(samples)
        corr = float(np.corrcoef(samples[:-1], samples[1:])[0, 1])
        assert corr > 0.8


# ---------------------------------------------------------------------------
# Latency buffer tests
# ---------------------------------------------------------------------------


class TestLatencyBuffer:
    def test_zero_latency_is_current_truth(self):
        # sigma=0 so we can compare exactly.
        truths = [_joint_array(float(t)) for t in range(5)]
        env = _make_wrapper(
            truths, sigma=0.0, alpha=0.0, latency_steps=0, seed=0,
        )
        obs, *_ = env.step(env.action_space.sample())
        np.testing.assert_allclose(obs["human_state_estimate"], truths[0])

    def test_n_step_latency_delays_estimate(self):
        truths = [_joint_array(float(t)) for t in range(10)]
        env = _make_wrapper(
            truths, sigma=0.0, alpha=0.0, latency_steps=3, seed=0,
        )
        # Semantics: at step t the wrapper surfaces truth[t - latency_steps],
        # clamped to truth[0] while the buffer is still filling.
        surfaced = []
        for _ in range(8):
            obs, *_ = env.step(env.action_space.sample())
            surfaced.append(obs["human_state_estimate"][0, 0])
        # Steps 0..3 all clamp to truth[0] = 0; step 4 first surfaces truth[1].
        assert surfaced[0] == pytest.approx(0.0)
        assert surfaced[1] == pytest.approx(0.0)
        assert surfaced[2] == pytest.approx(0.0)
        assert surfaced[3] == pytest.approx(0.0)
        assert surfaced[4] == pytest.approx(1.0)
        assert surfaced[5] == pytest.approx(2.0)
        assert surfaced[6] == pytest.approx(3.0)

    def test_reset_clears_latency_buffer(self):
        truths = [_joint_array(float(t)) for t in range(10)]
        env = _make_wrapper(
            truths, sigma=0.0, alpha=0.0, latency_steps=2, seed=0,
        )
        for _ in range(5):
            env.step(env.action_space.sample())
        # Reset the stub env's counter and the wrapper's buffer.
        env.env._t = 0
        env.reset(seed=0)
        obs, *_ = env.step(env.action_space.sample())
        # After reset, step 0 must surface the fresh first truth (t=0), not a
        # leftover value from the prior rollout.
        assert obs["human_state_estimate"][0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Occlusion tests (ray-cast mocked)
# ---------------------------------------------------------------------------


class TestOcclusion:
    def test_all_occluded_inflates_sigma(self, monkeypatch):
        # Mock check_joint_visibility to always say fully occluded.
        import safety_bigym.filters.body_slam_wrapper as mod

        def fake_vis(model, data, camera, targets, exclude_geoms=None, tol=0.05):
            return np.zeros(len(targets), dtype=bool)
        monkeypatch.setattr(mod, "check_joint_visibility", fake_vis)

        env = _make_wrapper(
            lambda t: _joint_array(0.0),
            sigma=0.1, alpha=0.0, latency_steps=0,
            use_occlusion=True, occlusion_multiplier=3.0, seed=0,
        )
        samples = []
        for _ in range(3000):
            obs, *_ = env.step(env.action_space.sample())
            samples.append(obs["human_state_estimate"][0, 0])
        std = float(np.std(samples))
        # Expected std ≈ 3 * sigma = 0.3.
        assert std == pytest.approx(0.3, rel=0.2)

    def test_per_joint_occlusion_flag_mirrors_visibility(self, monkeypatch):
        import safety_bigym.filters.body_slam_wrapper as mod

        def fake_vis(model, data, camera, targets, exclude_geoms=None, tol=0.05):
            # joint 0 visible, joint 1 occluded, joint 2 visible
            return np.array([True, False, True])
        monkeypatch.setattr(mod, "check_joint_visibility", fake_vis)

        env = _make_wrapper(
            lambda t: _joint_array(0.0),
            sigma=0.05, alpha=0.0, latency_steps=0,
            use_occlusion=True, seed=0,
        )
        obs, *_ = env.step(env.action_space.sample())
        np.testing.assert_array_equal(
            obs["human_state_occluded"],
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    def test_use_occlusion_false_never_calls_raycast(self, monkeypatch):
        import safety_bigym.filters.body_slam_wrapper as mod

        called = {"n": 0}

        def fake_vis(*a, **kw):
            called["n"] += 1
            return np.ones(NUM_JOINTS, dtype=bool)
        monkeypatch.setattr(mod, "check_joint_visibility", fake_vis)

        env = _make_wrapper(
            lambda t: _joint_array(0.0), use_occlusion=False, seed=0,
        )
        for _ in range(10):
            env.step(env.action_space.sample())
        assert called["n"] == 0


# ---------------------------------------------------------------------------
# Tracking-lost dropout tests
# ---------------------------------------------------------------------------


class TestTrackingLost:
    def test_dropout_rate_matches_configured_probability(self):
        env = _make_wrapper(
            lambda t: _joint_array(float(t)),
            sigma=0.0, alpha=0.0, latency_steps=0,
            dropout_prob=0.02, seed=42,
        )
        N = 10_000
        drops = 0
        prev = None
        for _ in range(N):
            obs, *_ = env.step(env.action_space.sample())
            cur = obs["human_state_estimate"][0, 0]
            # A dropout keeps the estimate frozen: cur == prev and staleness > 0.
            if prev is not None and cur == prev and obs["human_state_staleness"][0] > 0:
                drops += 1
            prev = cur
        rate = drops / N
        # 3-sigma band on Binomial(N, 0.02).
        sigma = np.sqrt(0.02 * 0.98 / N)
        assert abs(rate - 0.02) < 3 * sigma + 0.005

    def test_staleness_increments_during_dropout(self):
        # Force-drop every step by setting dropout_prob=1.0.
        env = _make_wrapper(
            lambda t: _joint_array(float(t)),
            sigma=0.0, alpha=0.0, latency_steps=0,
            dropout_prob=1.0, dt=0.02, seed=0,
        )
        prev_stale = None
        for i in range(5):
            obs, *_ = env.step(env.action_space.sample())
            stale = float(obs["human_state_staleness"][0])
            if prev_stale is not None:
                assert stale > prev_stale
            prev_stale = stale

    def test_staleness_resets_after_recovery(self):
        # Run dropout for a few steps, then force recovery (dropout_prob=0).
        env = _make_wrapper(
            lambda t: _joint_array(float(t)),
            sigma=0.0, alpha=0.0, latency_steps=0,
            dropout_prob=1.0, seed=0,
        )
        for _ in range(3):
            env.step(env.action_space.sample())
        # Flip the prob to zero and step once more.
        env.body_slam_config.dropout_prob = 0.0
        obs, *_ = env.step(env.action_space.sample())
        assert float(obs["human_state_staleness"][0]) == 0.0
        assert float(obs["human_state_occluded"].sum()) == 0.0

    def test_dropout_returns_last_known_estimate(self):
        # sigma=0, latency=0, so pre-dropout the estimate equals truth.
        env = _make_wrapper(
            lambda t: _joint_array(float(t)),
            sigma=0.0, alpha=0.0, latency_steps=0,
            dropout_prob=0.0, seed=0,
        )
        obs_a, *_ = env.step(env.action_space.sample())
        pre = obs_a["human_state_estimate"].copy()
        # Now force dropout on subsequent step.
        env.body_slam_config.dropout_prob = 1.0
        obs_b, *_ = env.step(env.action_space.sample())
        np.testing.assert_array_equal(obs_b["human_state_estimate"], pre)


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_identical_rollout(self):
        def rollout(seed):
            env = _make_wrapper(
                lambda t: _joint_array(0.0),
                sigma=0.05, alpha=0.9, latency_steps=1,
                dropout_prob=0.02, seed=seed,
            )
            out = []
            for _ in range(50):
                obs, *_ = env.step(env.action_space.sample())
                out.append(obs["human_state_estimate"].copy())
            return np.stack(out)

        a = rollout(123)
        b = rollout(123)
        np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# Oracle mode (plan requirement for E1.1 baseline/oracle/noisy ablation)
# ---------------------------------------------------------------------------


class TestOracleMode:
    def test_oracle_mode_returns_exact_truth(self):
        truths = [_joint_array(float(t)) for t in range(5)]
        env = _make_wrapper(
            truths, mode="oracle",
            # The below are ignored when mode=oracle, but pass arbitrary values.
            sigma=0.5, alpha=0.9, latency_steps=3,
            use_occlusion=True, dropout_prob=1.0, seed=0,
        )
        for i in range(5):
            obs, *_ = env.step(env.action_space.sample())
            np.testing.assert_allclose(obs["human_state_estimate"], truths[i])
            assert float(obs["human_state_occluded"].sum()) == 0.0
            assert float(obs["human_state_staleness"][0]) == 0.0
