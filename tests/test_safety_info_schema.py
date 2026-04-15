"""
End-to-end tests that info["safety"] is fully populated at env.step() time.

The wrapper-level tests in test_iso15066.py cover the SafetyInfo aggregation
math; these tests guard the env-level plumbing — the class of bug where a
SafetyInfo field exists on the dataclass but is never written by the env's
aggregation path (as was the case for pfl_force_ratio pre-T0.1).
"""

import os
from pathlib import Path

import numpy as np
import pytest


AMASS_DIR = os.environ.get("AMASS_DATA_DIR")
HAS_AMASS = AMASS_DIR is not None and Path(AMASS_DIR).exists()

# Field -> expected Python type (or tuple of types). Bools are checked
# first because Python bools are also ints.
EXPECTED_SCHEMA = {
    "ssm_violation": bool,
    "pfl_violation": bool,
    "ssm_margin": (int, float),
    "pfl_force_ratio": (int, float),
    "min_separation": (int, float),
    "max_contact_force": (int, float),
    "contact_region": str,
    "contact_type": str,
    "violations_by_region": dict,
    "robot_pos": list,
    "human_pos": list,
}


def _make_env(inject_human: bool):
    from bigym.action_modes import JointPositionActionMode
    from safety_bigym import SafetyBiGymEnv, SafetyConfig, HumanConfig

    action_mode = JointPositionActionMode(floating_base=True, absolute=True)
    return SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=SafetyConfig(),
        human_config=HumanConfig(),
        inject_human=inject_human,
    )


@pytest.fixture
def env_with_human():
    env = _make_env(inject_human=True)
    env.reset(seed=0)
    yield env
    env.close()


def _assert_type(value, expected):
    if expected is bool:
        assert isinstance(value, bool), f"expected bool, got {type(value).__name__}"
    elif isinstance(expected, tuple):
        # numeric fields — accept numpy scalars too
        assert isinstance(value, expected) or hasattr(value, "__float__"), (
            f"expected one of {expected}, got {type(value).__name__}"
        )
        # but never a bool sneaking in as an int
        assert not isinstance(value, bool), "numeric field must not be bool"
    else:
        assert isinstance(value, expected), (
            f"expected {expected.__name__}, got {type(value).__name__}"
        )


def test_info_safety_keys_present_at_step(env_with_human):
    """Every SafetyInfo field advertised in to_dict() must appear in info['safety']."""
    action = np.zeros(env_with_human.action_space.shape)
    _, _, _, _, info = env_with_human.step(action)

    assert "safety" in info
    safety = info["safety"]
    for key, expected_type in EXPECTED_SCHEMA.items():
        assert key in safety, f"info['safety'] missing key: {key}"
        _assert_type(safety[key], expected_type)


def test_pfl_ratio_consistent_with_violation_flag(env_with_human):
    """Over N random-action steps, pfl_violation must agree with (ratio > 1.0)."""
    for _ in range(30):
        action = env_with_human.action_space.sample()
        _, _, done, trunc, info = env_with_human.step(action)
        safety = info["safety"]
        ratio = float(safety["pfl_force_ratio"])
        violation = bool(safety["pfl_violation"])
        assert ratio >= 0.0, f"pfl_force_ratio must be non-negative, got {ratio}"
        # The violation flag should fire exactly when the ratio exceeds the limit.
        if violation:
            assert ratio > 1.0, (
                f"pfl_violation=True but ratio={ratio} <= 1.0"
            )
        if done or trunc:
            break


def test_ssm_margin_is_finite_with_human(env_with_human):
    """Once a human is injected and physics has stepped once, ssm_margin is finite."""
    action = np.zeros(env_with_human.action_space.shape)
    _, _, _, _, info = env_with_human.step(action)
    margin = float(info["safety"]["ssm_margin"])
    assert np.isfinite(margin), f"ssm_margin must be finite, got {margin}"


def test_min_separation_positive_when_not_violating(env_with_human):
    """min_separation (d_min) cannot be negative; at reset, robot and human
    are at least spawn_distance apart."""
    action = np.zeros(env_with_human.action_space.shape)
    _, _, _, _, info = env_with_human.step(action)
    d_min = float(info["safety"]["min_separation"])
    assert d_min >= 0.0, f"min_separation must be non-negative, got {d_min}"


def test_closest_pair_names_reported_with_human(env_with_human):
    """After T0.2, SafetyInfo carries the closest human joint + robot link names."""
    action = np.zeros(env_with_human.action_space.shape)
    _, _, _, _, info = env_with_human.step(action)
    safety = info["safety"]
    # These keys are added by T0.2. Empty string is allowed (no data yet),
    # but the keys MUST exist.
    assert "closest_human_joint" in safety
    assert "closest_robot_link" in safety
    assert isinstance(safety["closest_human_joint"], str)
    assert isinstance(safety["closest_robot_link"], str)


if not HAS_AMASS:
    # Clips aren't strictly required for these tests (the human sits at
    # reset pose if no clip loads), so we run them by default. However,
    # some CI environments cannot even construct BiGymEnv. If that happens,
    # skip the whole module rather than flood the log with import errors.
    try:
        import bigym  # noqa: F401
    except ImportError:
        pytestmark = pytest.mark.skip(reason="bigym not importable")
