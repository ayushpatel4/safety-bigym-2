"""Tests for the thesis 3-flavour safety metrics contract.

Reference: docs/safety_metrics.md. The wrapper must emit all three
violation flavours per env-step (worst-case SSM, observed-velocity SSM,
geometric proximity), and the episode aggregator must roll them into
matching ``ep_*_rate`` / dwell / quantile keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.config import SSMConfig  # noqa: E402
from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics  # noqa: E402
from safety_bigym.safety.iso15066_wrapper import SafetyInfo  # noqa: E402


def _build_safety_info(min_separation, robot_vel, human_vel, ssm_config):
    """Minimally-populated SafetyInfo with 3-flavour fields set, mimicking
    what ISO15066Wrapper._ssm_into emits.
    """
    info = SafetyInfo()
    info.min_separation = min_separation
    info.robot_vel = float(robot_vel)
    info.human_vel = float(human_vel)
    info.proximity_threshold = float(ssm_config.proximity_threshold)

    S_p_worst = ssm_config.compute_separation_distance(robot_vel, None)
    S_p_actual = ssm_config.compute_separation_distance(robot_vel, human_vel)
    info.ssm_margin = float(min_separation - S_p_worst)
    info.ssm_margin_actual = float(min_separation - S_p_actual)
    info.ssm_violation = info.ssm_margin < 0
    info.ssm_violation_actual = info.ssm_margin_actual < 0
    info.proximity_violation = min_separation < ssm_config.proximity_threshold
    return info


def test_safetyinfo_to_dict_carries_all_three_flavours():
    cfg = SSMConfig()
    info = _build_safety_info(min_separation=0.6, robot_vel=0.5,
                              human_vel=0.4, ssm_config=cfg)
    d = info.to_dict()
    for key in (
        "ssm_violation", "ssm_violation_actual", "proximity_violation",
        "ssm_margin", "ssm_margin_actual", "min_separation",
        "robot_vel", "human_vel", "proximity_threshold",
    ):
        assert key in d, f"to_dict missing {key}"


def test_failure_mode_robot_fast_human_distant():
    """Robot fast, human distant: SSM violations fire, proximity doesn't.

    With ``v_r = 3.0 m/s`` and default ISO 15066 params,
    ``S_p_worst ≈ 1.54 m`` and ``S_p_actual(v_h=0) ≈ 1.30 m``. A
    ``min_separation = 0.7 m`` is below both bounds but above the default
    ``proximity_threshold = 0.5 m``, so SSM fires but proximity doesn't.
    """
    cfg = SSMConfig()
    info = _build_safety_info(min_separation=0.7, robot_vel=3.0,
                              human_vel=0.0, ssm_config=cfg)
    assert info.ssm_violation, "expected worst-case SSM to fire"
    assert info.ssm_violation_actual, "expected actual SSM to fire"
    assert not info.proximity_violation, "geometric proximity should be safe"


def test_failure_mode_robot_still_human_inside_0p4m():
    """Robot still, human at 0.4 m → proximity fires; SSM_actual depends."""
    cfg = SSMConfig()
    info = _build_safety_info(min_separation=0.4, robot_vel=0.0,
                              human_vel=0.0, ssm_config=cfg)
    assert info.proximity_violation, "0.4 m < default 0.5 m threshold"


def test_safe_regime_none_fire():
    """Both still, 0.6 m apart → no flavour fires (default τ=0.5)."""
    cfg = SSMConfig()
    info = _build_safety_info(min_separation=0.6, robot_vel=0.0,
                              human_vel=0.0, ssm_config=cfg)
    assert not info.proximity_violation
    # With both velocities zero, S_p = C = 0.1 m, so 0.6 > 0.1 → no SSM either.
    assert not info.ssm_violation
    assert not info.ssm_violation_actual


def test_proximity_threshold_is_configurable():
    cfg = SSMConfig(proximity_threshold=0.3)
    info = _build_safety_info(min_separation=0.4, robot_vel=0.0,
                              human_vel=0.0, ssm_config=cfg)
    # 0.4 m > 0.3 m → safe under tighter threshold
    assert not info.proximity_violation
    assert info.proximity_threshold == 0.3


# ----------------------------------------------------------------------
# EpisodeSafetyMetrics — new aggregate fields
# ----------------------------------------------------------------------


class _StubEnv(gym.Env):
    """Tiny stub: emits a configured sequence of info["safety"] dicts."""

    observation_space = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, safety_seq):
        self._seq = list(safety_seq)
        self._i = 0

    def reset(self, **kwargs):
        self._i = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        info = {"safety": self._seq[self._i]} if self._i < len(self._seq) else {}
        self._i += 1
        terminated = self._i >= len(self._seq)
        return np.zeros(1, dtype=np.float32), 0.0, terminated, False, info


def _safety_step(min_sep, ssm=False, ssm_actual=False, prox=False,
                 margin=1.0, margin_actual=1.0, robot_vel=0.0, pfl=False):
    return {
        "ssm_violation": ssm,
        "ssm_violation_actual": ssm_actual,
        "proximity_violation": prox,
        "pfl_violation": pfl,
        "ssm_margin": margin,
        "ssm_margin_actual": margin_actual,
        "min_separation": min_sep,
        "pfl_force_ratio": 0.0,
        "max_contact_force": 0.0,
        "robot_vel": robot_vel,
        "violations_by_region": {},
    }


def test_episode_aggregator_emits_all_new_keys():
    """End-of-episode summary contains every key the thesis Pareto plot reads."""
    seq = [
        _safety_step(min_sep=1.2, robot_vel=0.5),
        _safety_step(min_sep=0.6, robot_vel=1.0),
        _safety_step(min_sep=0.4, prox=True, ssm_actual=True,
                     margin_actual=-0.1, robot_vel=1.2),
        _safety_step(min_sep=0.25, prox=True, ssm=True, ssm_actual=True,
                     margin=-0.2, margin_actual=-0.3, robot_vel=1.5),
    ]
    env = EpisodeSafetyMetrics(_StubEnv(seq))
    env.reset()
    for _ in range(len(seq)):
        env.step(env.action_space.sample())
    summary = env._summary()  # final aggregate

    expected = {
        "ep_steps", "ep_ssm_violation_rate", "ep_ssm_violation_actual_rate",
        "ep_proximity_violation_rate", "ep_pfl_violation_rate",
        "ep_min_ssm_margin", "ep_min_ssm_margin_actual",
        "ep_min_separation", "ep_mean_separation",
        "ep_p5_separation", "ep_p25_separation",
        "ep_max_pfl_force_ratio", "ep_max_contact_force",
        "ep_max_robot_vel", "ep_mean_robot_vel",
        "ep_time_to_first_violation",
        "ep_time_in_proximity_0p3m",
        "ep_time_in_proximity_0p5m",
        "ep_time_in_proximity_1p0m",
    }
    missing = expected - set(summary.keys())
    assert not missing, f"summary missing keys: {missing}"

    # Sanity checks against the constructed sequence.
    assert summary["ep_steps"] == 4
    assert summary["ep_proximity_violation_rate"] == pytest.approx(2 / 4)
    assert summary["ep_ssm_violation_rate"] == pytest.approx(1 / 4)
    assert summary["ep_ssm_violation_actual_rate"] == pytest.approx(2 / 4)
    assert summary["ep_min_separation"] == pytest.approx(0.25)
    assert summary["ep_max_robot_vel"] == pytest.approx(1.5)
    assert summary["ep_mean_robot_vel"] == pytest.approx(
        (0.5 + 1.0 + 1.2 + 1.5) / 4
    )
    # 0.25 < 0.3, 0.5, 1.0; 0.4 < 0.5, 1.0; 0.6 < 1.0; 1.2 < (none)
    assert summary["ep_time_in_proximity_0p3m"] == pytest.approx(1 / 4)
    assert summary["ep_time_in_proximity_0p5m"] == pytest.approx(2 / 4)
    assert summary["ep_time_in_proximity_1p0m"] == pytest.approx(3 / 4)
    # First violation step uses worst-case ssm_violation OR pfl_violation
    # (docs/safety_metrics.md). Step 3 is the first one with ssm=True.
    assert summary["ep_time_to_first_violation"] == 3


def test_episode_aggregator_clean_episode():
    """All-safe episode → all rates 0, no first-violation, min_sep = sample min."""
    seq = [_safety_step(min_sep=2.0, robot_vel=0.1) for _ in range(5)]
    env = EpisodeSafetyMetrics(_StubEnv(seq))
    env.reset()
    for _ in range(len(seq)):
        env.step(env.action_space.sample())
    summary = env._summary()
    assert summary["ep_proximity_violation_rate"] == 0.0
    assert summary["ep_ssm_violation_actual_rate"] == 0.0
    assert summary["ep_time_to_first_violation"] == -1
    assert summary["ep_min_separation"] == pytest.approx(2.0)
    assert summary["ep_time_in_proximity_1p0m"] == 0.0


def test_episode_aggregator_handles_missing_new_keys():
    """Backward-compat: a step that omits the new fields shouldn't crash."""
    # Old-style safety dict (no ssm_violation_actual / proximity_violation)
    legacy = {
        "ssm_violation": False, "pfl_violation": False,
        "ssm_margin": 1.0, "min_separation": 1.0,
        "pfl_force_ratio": 0.0, "max_contact_force": 0.0,
        "violations_by_region": {},
    }
    env = EpisodeSafetyMetrics(_StubEnv([legacy, legacy]))
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    summary = env._summary()
    # All new aggregates default to safe values.
    assert summary["ep_ssm_violation_actual_rate"] == 0.0
    assert summary["ep_proximity_violation_rate"] == 0.0
    assert summary["ep_max_robot_vel"] == 0.0
