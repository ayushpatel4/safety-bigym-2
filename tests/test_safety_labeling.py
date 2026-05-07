"""Tests for filters/labeling.py — binary safety reward labeller."""

import pytest

from safety_bigym.filters.labeling import label_transition


def _safety_info(ssm_violation: bool = False, pfl_violation: bool = False) -> dict:
    return {
        "safety": {
            "ssm_violation": ssm_violation,
            "pfl_violation": pfl_violation,
            "ssm_margin": -0.05 if ssm_violation else 0.40,
            "pfl_force_ratio": 1.2 if pfl_violation else 0.0,
        }
    }


def test_clean_step_yields_one():
    r_safe, terminal = label_transition(_safety_info())
    assert r_safe == 1.0
    assert terminal is False


def test_ssm_violation_yields_zero_and_terminal():
    r_safe, terminal = label_transition(_safety_info(ssm_violation=True))
    assert r_safe == 0.0
    assert terminal is True


def test_pfl_ignored_when_flag_off():
    """v1 default: PFL is broken (always 0). Even if pfl_violation=True is
    spuriously set, the SSM-only labeller must ignore it."""
    r_safe, terminal = label_transition(_safety_info(pfl_violation=True))
    assert r_safe == 1.0
    assert terminal is False


def test_pfl_or_ssm_when_flag_on():
    """Future PFL-fix retrofit path: use_pfl=True ORs in pfl_violation."""
    r_safe, terminal = label_transition(
        _safety_info(pfl_violation=True), use_pfl=True
    )
    assert r_safe == 0.0
    assert terminal is True

    r_safe, terminal = label_transition(
        _safety_info(ssm_violation=True), use_pfl=True
    )
    assert r_safe == 0.0
    assert terminal is True


def test_both_violations_with_use_pfl_still_terminal():
    r_safe, terminal = label_transition(
        _safety_info(ssm_violation=True, pfl_violation=True), use_pfl=True
    )
    assert r_safe == 0.0
    assert terminal is True


def test_missing_safety_key_raises():
    """The labeller is a strict pure function; the caller must provide info['safety']."""
    with pytest.raises(KeyError):
        label_transition({})


def test_returns_python_floats_and_bools():
    """Labels must be JSON-serialisable primitives (not numpy scalars), so they
    can be saved to npz / manifest.json without conversion."""
    r_safe, terminal = label_transition(_safety_info(ssm_violation=True))
    assert isinstance(r_safe, float)
    assert isinstance(terminal, bool)
