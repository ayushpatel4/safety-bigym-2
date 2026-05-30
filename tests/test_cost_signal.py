"""Unit tests for filters/cost_signal.py — the Phase 3 per-step cost formula."""

import math

import pytest

from safety_bigym.filters.cost_signal import (
    COST_FORMS,
    D_BUFFER_DEFAULT,
    PFL_RATIO_THRESHOLD_DEFAULT,
    compute_cost,
    select_cost,
)


def test_empty_info_returns_zero():
    assert compute_cost({}) == 0.0


def test_missing_keys_treated_as_zero_contribution():
    # No ssm_margin / pfl_force_ratio at all → no contribution from either
    assert compute_cost({"unrelated": 99.0}) == 0.0


def test_ssm_margin_far_zero_cost():
    # ssm_margin = d_buffer → 1 - 1 = 0
    assert compute_cost({"ssm_margin": D_BUFFER_DEFAULT}) == 0.0


def test_ssm_margin_at_zero_full_cost():
    # ssm_margin = 0 → c_ssm = 1.0
    assert compute_cost({"ssm_margin": 0.0}) == pytest.approx(1.0)


def test_ssm_margin_half_buffer_half_cost():
    # ssm_margin = d_buffer / 2 → c_ssm = 0.5
    assert compute_cost({"ssm_margin": D_BUFFER_DEFAULT / 2.0}) == pytest.approx(0.5)


def test_ssm_margin_negative_clipped_to_one():
    # Deep violation (margin < 0) → c_ssm would exceed 1; clipped to 1.0
    c = compute_cost({"ssm_margin": -1.0})
    assert c == pytest.approx(1.0)
    assert c <= 1.0


def test_ssm_margin_well_above_buffer_zero():
    # 5 m clearance → no cost
    assert compute_cost({"ssm_margin": 5.0}) == 0.0


def test_pfl_below_threshold_zero():
    # PFL ratio = 0.5 < 0.8 threshold → c_pfl = 0
    assert compute_cost({"pfl_force_ratio": 0.5}) == 0.0


def test_pfl_at_threshold_zero():
    assert compute_cost({"pfl_force_ratio": PFL_RATIO_THRESHOLD_DEFAULT}) == 0.0


def test_pfl_above_threshold_linear():
    # PFL ratio = 0.9, threshold = 0.8 → c_pfl = 0.1
    assert compute_cost({"pfl_force_ratio": 0.9}) == pytest.approx(0.1)


def test_cost_is_max_of_ssm_and_pfl():
    info = {"ssm_margin": 0.15, "pfl_force_ratio": 0.95}
    # c_ssm = 1 - 0.15/0.3 = 0.5; c_pfl = 0.95 - 0.8 = 0.15; max → 0.5
    assert compute_cost(info) == pytest.approx(0.5)


def test_cost_picks_pfl_when_dominant():
    # PFL strongly active, SSM only weakly so → max picks PFL
    info = {"ssm_margin": 0.28, "pfl_force_ratio": 1.0}
    c_ssm = 1.0 - 0.28 / D_BUFFER_DEFAULT  # ≈ 0.0667
    c_pfl = 1.0 - PFL_RATIO_THRESHOLD_DEFAULT  # 0.2
    assert compute_cost(info) == pytest.approx(max(c_ssm, c_pfl))


def test_custom_d_buffer_sweepable():
    # Tightening d_buffer makes the same margin look closer (higher cost).
    info = {"ssm_margin": 0.1}
    c_loose = compute_cost(info, d_buffer=1.0)  # 1 - 0.1 = 0.9
    c_tight = compute_cost(info, d_buffer=0.2)  # 1 - 0.5 = 0.5
    assert c_loose == pytest.approx(0.9)
    assert c_tight == pytest.approx(0.5)


def test_custom_pfl_threshold_sweepable():
    info = {"pfl_force_ratio": 0.9}
    c_strict = compute_cost(info, pfl_threshold=0.5)  # 0.9 - 0.5 = 0.4
    c_loose = compute_cost(info, pfl_threshold=0.95)  # max(0, -0.05) = 0
    assert c_strict == pytest.approx(0.4)
    assert c_loose == 0.0


def test_output_in_unit_interval_for_all_reasonable_inputs():
    for margin in (-10.0, -1.0, 0.0, 0.1, 0.3, 1.0, 100.0):
        for ratio in (0.0, 0.5, 0.8, 0.9, 2.0):
            c = compute_cost(
                {"ssm_margin": margin, "pfl_force_ratio": ratio}
            )
            assert 0.0 <= c <= 1.0


def test_nan_margin_does_not_crash():
    # Float-typed but pathological — should not propagate NaN through Q_c later.
    # Current implementation: float(nan) flows; max(0, 1-nan/0.3) is nan; min(1, nan) is nan.
    # We accept that the function is _not_ NaN-defensive at this layer; if NaN ever
    # appears in info["safety"], it's a wrapper bug to fix there. This test pins the
    # contract: compute_cost itself does not throw.
    c = compute_cost({"ssm_margin": float("nan")})
    # NaN may propagate; we only assert no exception.
    assert math.isnan(c) or 0.0 <= c <= 1.0


# --- E3.1 cost-form selector (select_cost) --------------------------------


def test_select_cost_continuous_matches_compute_cost():
    # The default form is the graded compute_cost value, verbatim.
    info = {"ssm_margin": 0.15, "pfl_force_ratio": 0.95}
    assert select_cost(info, cost_form="continuous") == compute_cost(info)


def test_select_cost_continuous_threads_d_buffer():
    info = {"ssm_margin": 0.1}
    assert select_cost(info, cost_form="continuous", d_buffer=0.2) == pytest.approx(
        compute_cost(info, d_buffer=0.2)
    )


def test_select_cost_binary_fires_on_ssm_violation():
    assert select_cost({"ssm_violation": True}, cost_form="binary") == 1.0
    assert select_cost({"ssm_violation": 1}, cost_form="binary") == 1.0


def test_select_cost_binary_zero_when_no_violation():
    assert select_cost({"ssm_violation": False}, cost_form="binary") == 0.0
    assert select_cost({}, cost_form="binary") == 0.0          # empty
    assert select_cost({"ssm_margin": 0.0}, cost_form="binary") == 0.0  # key missing


def test_select_cost_binary_ignores_graded_margin():
    # Binary is purely the violation flag — a near-zero margin without the flag
    # set is still 0.0 (contrast: continuous would report a large graded cost).
    info = {"ssm_margin": 0.01, "ssm_violation": False}
    assert select_cost(info, cost_form="binary") == 0.0
    assert select_cost(info, cost_form="continuous") > 0.5


def test_select_cost_binary_in_unit_interval():
    for v in (True, False, 0, 1):
        assert 0.0 <= select_cost({"ssm_violation": v}, cost_form="binary") <= 1.0


def test_select_cost_rejects_unknown_form():
    with pytest.raises(ValueError):
        select_cost({"ssm_margin": 0.0}, cost_form="fixed")  # not a cost form
    with pytest.raises(ValueError):
        select_cost({}, cost_form="bogus")


def test_cost_forms_constant():
    assert COST_FORMS == ("continuous", "binary")
