"""Tests for EpisodeSafetyMetrics wrapper.

The wrapper aggregates per-step info["safety"] into per-episode scalars
and emits them under info["episode_safety"] at terminated/truncated=True.
"""

import numpy as np
import pytest
from gymnasium import spaces
import gymnasium as gym


class _StubSafetyEnv(gym.Env):
    """Minimal env that emits scripted info['safety'] payloads per step."""

    metadata: dict = {}

    def __init__(self, scripted_infos):
        self._infos = list(scripted_infos)
        self._t = 0
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        safety = self._infos[self._t]
        self._t += 1
        terminated = self._t >= len(self._infos)
        return (
            np.zeros((1,), dtype=np.float32),
            0.0,
            terminated,
            False,
            {"safety": safety},
        )


def _safety(
    ssm_v=False,
    pfl_v=False,
    margin=1.0,
    ratio=0.0,
    force=0.0,
    region=None,
    *,
    ssm_v_actual=None,
    margin_actual=None,
    proximity_v=None,
    min_separation=1.0,
    proximity_threshold=0.5,
    robot_vel=0.0,
    human_vel=0.0,
):
    # Default the new fields off existing ones for back-compat with older
    # test cases (they pass only the legacy kwargs).
    if ssm_v_actual is None:
        ssm_v_actual = ssm_v
    if margin_actual is None:
        margin_actual = margin
    if proximity_v is None:
        proximity_v = min_separation < proximity_threshold
    return {
        "ssm_violation": ssm_v,
        "ssm_violation_actual": ssm_v_actual,
        "proximity_violation": proximity_v,
        "pfl_violation": pfl_v,
        "ssm_margin": margin,
        "ssm_margin_actual": margin_actual,
        "pfl_force_ratio": ratio,
        "min_separation": min_separation,
        "max_contact_force": force,
        "contact_region": "" if region is None else region,
        "contact_type": "",
        "proximity_threshold": proximity_threshold,
        "robot_vel": robot_vel,
        "human_vel": human_vel,
        "violations_by_region": ({region: 1} if region else {}),
        "robot_pos": [0.0, 0.0, 0.0],
        "human_pos": [1.0, 0.0, 0.0],
        "closest_human_joint": "",
        "closest_robot_link": "",
    }


def _make_wrapped(scripted):
    from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics

    return EpisodeSafetyMetrics(_StubSafetyEnv(scripted))


def test_episode_safety_emitted_every_step():
    """EpisodeSafetyMetrics injects ``episode_safety`` on every step (not just
    at episode end) so VectorEnv/W&B always sees the key. The running summary
    is valid mid-episode; consumers read it at done for the final aggregate."""
    scripted = [_safety(), _safety(), _safety()]
    env = _make_wrapped(scripted)
    env.reset()
    _, _, done, trunc, info = env.step(env.action_space.sample())
    assert not done and not trunc
    assert "episode_safety" in info  # running summary present mid-episode
    assert "ep_ssm_violation_rate" in info["episode_safety"]


def test_episode_safety_emitted_on_terminated():
    scripted = [_safety(margin=0.8), _safety(margin=0.3, ssm_v=True)]
    env = _make_wrapped(scripted)
    env.reset()
    for i in range(len(scripted)):
        _, _, done, trunc, info = env.step(env.action_space.sample())
    assert done
    assert "episode_safety" in info
    ep = info["episode_safety"]
    assert ep["ep_ssm_violation_rate"] == pytest.approx(0.5)
    assert ep["ep_min_ssm_margin"] == pytest.approx(0.3)


def test_max_ratio_and_force():
    scripted = [
        _safety(ratio=0.4, force=10.0),
        _safety(ratio=1.2, force=40.0, pfl_v=True),
        _safety(ratio=0.9, force=20.0),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    assert done
    ep = info["episode_safety"]
    assert ep["ep_max_pfl_force_ratio"] == pytest.approx(1.2)
    assert ep["ep_max_contact_force"] == pytest.approx(40.0)
    assert ep["ep_pfl_violation_rate"] == pytest.approx(1 / 3)


def test_time_to_first_violation_reports_step_index():
    scripted = [
        _safety(),
        _safety(),
        _safety(ssm_v=True, margin=0.1),
        _safety(),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    assert done
    ep = info["episode_safety"]
    assert ep["ep_time_to_first_violation"] == 2


def test_time_to_first_violation_none_when_clean():
    scripted = [_safety(), _safety()]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_time_to_first_violation"] == -1


def test_region_counts_aggregated():
    scripted = [
        _safety(pfl_v=True, region="hand"),
        _safety(pfl_v=True, region="hand"),
        _safety(pfl_v=True, region="chest"),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_region_hand"] == 2
    assert ep["ep_region_chest"] == 1


def test_reset_clears_state_between_episodes():
    from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics

    scripted_a = [_safety(ssm_v=True, margin=0.1)]
    env = EpisodeSafetyMetrics(_StubSafetyEnv(scripted_a))
    env.reset()
    _, _, done, _, info_a = env.step(env.action_space.sample())
    assert done
    assert info_a["episode_safety"]["ep_ssm_violation_rate"] == pytest.approx(1.0)

    scripted_b = [_safety(margin=0.9)]
    env.env._infos = scripted_b
    env.reset()
    _, _, done, _, info_b = env.step(env.action_space.sample())
    assert done
    assert info_b["episode_safety"]["ep_ssm_violation_rate"] == pytest.approx(0.0)
    assert info_b["episode_safety"]["ep_min_ssm_margin"] == pytest.approx(0.9)


# --- Phase 3 thesis-metric additions (2026-05-26) ---


def test_proximity_violation_rate():
    """ep_proximity_violation_rate counts steps with the proximity flag set."""
    scripted = [
        _safety(min_separation=2.0),  # 2.0 > 0.5 → no proximity violation
        _safety(min_separation=0.3),  # 0.3 < 0.5 → proximity violation
        _safety(min_separation=0.4),  # 0.4 < 0.5 → proximity violation
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_proximity_violation_rate"] == pytest.approx(2 / 3)


def test_ssm_violation_actual_rate_independent_from_worst_case():
    """ep_ssm_violation_actual_rate tracks the velocity-adaptive flag,
    not the worst-case one. When robot is stationary, actual should
    fire less often than worst-case."""
    scripted = [
        _safety(ssm_v=True, ssm_v_actual=False, margin=-0.4, margin_actual=0.2),
        _safety(ssm_v=True, ssm_v_actual=True, margin=-0.4, margin_actual=-0.1),
        _safety(ssm_v=False, ssm_v_actual=False, margin=0.5, margin_actual=0.7),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_ssm_violation_rate"] == pytest.approx(2 / 3)
    assert ep["ep_ssm_violation_actual_rate"] == pytest.approx(1 / 3)


def test_time_in_proximity_probes_count_correctly():
    """Each ep_time_in_proximity_<label> reports the fraction of steps
    with min_separation strictly below that threshold."""
    scripted = [
        _safety(min_separation=2.0),  # > all thresholds
        _safety(min_separation=0.7),  # < 1.0 only
        _safety(min_separation=0.4),  # < 1.0 and 0.5
        _safety(min_separation=0.2),  # < all three
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_time_in_proximity_1p0m"] == pytest.approx(3 / 4)
    assert ep["ep_time_in_proximity_0p5m"] == pytest.approx(2 / 4)
    assert ep["ep_time_in_proximity_0p3m"] == pytest.approx(1 / 4)


def test_separation_distribution_quantiles():
    """Mean / p5 / p25 reflect the separation distribution, not just min."""
    # 10 samples; p5 ~= the 5th-percentile value, p25 ~= 25th-percentile.
    seps = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    scripted = [_safety(min_separation=s) for s in seps]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    expected_mean = sum(seps) / len(seps)
    assert ep["ep_min_separation"] == pytest.approx(0.1)
    assert ep["ep_mean_separation"] == pytest.approx(expected_mean)
    # numpy.quantile linear interpolation: q05 over 10 sorted points falls
    # between idx 0 and 1 (alpha = 0.05*9 = 0.45) → 0.1 + 0.45*(0.2-0.1)=0.145.
    assert ep["ep_p5_separation"] == pytest.approx(0.145, abs=1e-3)
    # q25: alpha = 0.25*9 = 2.25 → 0.3 + 0.25*(0.4-0.3) = 0.325.
    assert ep["ep_p25_separation"] == pytest.approx(0.325, abs=1e-3)


def test_separation_quantiles_no_samples_default_zero():
    """No safety steps observed → quantiles emit 0.0 placeholders rather
    than raising on the empty array."""
    env = _make_wrapped([])
    env.reset()
    ep = env.env._t  # noqa: unused; just to confirm the stub has no steps
    info = {"episode_safety": env._summary()}
    assert info["episode_safety"]["ep_min_separation"] == 0.0
    assert info["episode_safety"]["ep_mean_separation"] == 0.0
    assert info["episode_safety"]["ep_p5_separation"] == 0.0
    assert info["episode_safety"]["ep_p25_separation"] == 0.0


def test_robot_velocity_max_and_mean():
    """ep_max_robot_vel / ep_mean_robot_vel reflect the per-step robot speed."""
    scripted = [
        _safety(robot_vel=0.0),
        _safety(robot_vel=1.0),
        _safety(robot_vel=0.5),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_max_robot_vel"] == pytest.approx(1.0)
    assert ep["ep_mean_robot_vel"] == pytest.approx(0.5)


def test_min_ssm_margin_actual_tracks_separately_from_worst_case():
    """ep_min_ssm_margin_actual is the per-episode min of the velocity-
    adaptive margin and may diverge from ep_min_ssm_margin."""
    scripted = [
        _safety(margin=-0.4, margin_actual=0.2),  # worst-case stricter
        _safety(margin=0.1, margin_actual=-0.5),  # actual stricter on this step
        _safety(margin=0.3, margin_actual=0.6),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_min_ssm_margin"] == pytest.approx(-0.4)
    assert ep["ep_min_ssm_margin_actual"] == pytest.approx(-0.5)
