"""Tests for filters/labeling.py — binary safety reward labeller.

v2 (2026-05-16): labelling switched from ISO 15066 SSM (``ssm_violation``)
to geometric proximity (``min_separation < proximity_threshold``). ISO 15066's
required separation is industrial-cell calibrated and demands ≥5m clearance
at kitchen-scale robot velocities; this made every transition unsafe. See
labeling.py for the rationale.
"""

import pytest

from safety_bigym.filters.labeling import label_transition


def _safety_info(
    min_separation: float = 0.50,
    pfl_violation: bool = False,
) -> dict:
    return {
        "safety": {
            "min_separation": min_separation,
            "ssm_margin": min_separation - 0.30,  # informational only
            "ssm_violation": min_separation < 0.30,  # informational only
            "pfl_violation": pfl_violation,
            "pfl_force_ratio": 1.2 if pfl_violation else 0.0,
        }
    }


def test_clean_step_yields_one():
    """0.50 m separation >> default 0.10 m threshold → safe."""
    r_safe, terminal = label_transition(_safety_info())
    assert r_safe == 1.0
    assert terminal is False


def test_near_contact_yields_zero_and_terminal():
    """0.03 m separation < 0.10 m threshold → violation."""
    r_safe, terminal = label_transition(_safety_info(min_separation=0.03))
    assert r_safe == 0.0
    assert terminal is True


def test_threshold_boundary_is_safe():
    """``min_separation == threshold`` is *not* a violation (strict <)."""
    r_safe, terminal = label_transition(
        _safety_info(min_separation=0.10), proximity_threshold=0.10,
    )
    assert r_safe == 1.0
    assert terminal is False


def test_custom_threshold():
    """Tighter threshold makes the same separation safe; looser makes it unsafe."""
    info = _safety_info(min_separation=0.08)
    # default 0.10 → violation
    r_safe, _ = label_transition(info)
    assert r_safe == 0.0
    # tightened to 0.05 → safe
    r_safe, _ = label_transition(info, proximity_threshold=0.05)
    assert r_safe == 1.0


def test_pfl_ignored_when_flag_off():
    """PFL contact-detection is broken; default ``use_pfl=False`` must ignore it."""
    r_safe, terminal = label_transition(
        _safety_info(min_separation=0.50, pfl_violation=True)
    )
    assert r_safe == 1.0
    assert terminal is False


def test_pfl_or_proximity_when_flag_on():
    """Future PFL-fix retrofit path: ``use_pfl=True`` ORs in pfl_violation."""
    r_safe, terminal = label_transition(
        _safety_info(min_separation=0.50, pfl_violation=True), use_pfl=True,
    )
    assert r_safe == 0.0
    assert terminal is True

    r_safe, terminal = label_transition(
        _safety_info(min_separation=0.03), use_pfl=True,
    )
    assert r_safe == 0.0
    assert terminal is True


def test_both_violations_with_use_pfl_still_terminal():
    r_safe, terminal = label_transition(
        _safety_info(min_separation=0.03, pfl_violation=True), use_pfl=True,
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
    r_safe, terminal = label_transition(_safety_info(min_separation=0.03))
    assert isinstance(r_safe, float)
    assert isinstance(terminal, bool)
