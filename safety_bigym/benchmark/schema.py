"""Canonical per-cell CSV schema for the benchmark harness.

``COLUMNS`` is the single source of truth for the order and membership of the per-cell
CSV. Every results table in the report (8.1, 8.3, 8.5, 8.7, headline 8.9, 8.11) reads
columns from here, so the schema is append-only: add a column, never reorder/rename one
that a table consumes.

``assemble_row`` maps a flat ``values`` dict (identification meta merged with the
aggregate produced by :func:`safety_bigym.benchmark.aggregate.aggregate_cell`) onto a row
in ``COLUMNS`` order. Filter-mechanics columns are emitted as ``""`` when no filter is
attached, so the column set is identical with and without a filter.

Pure (stdlib only) — importable by tests without torch / mujoco / pandas.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["COLUMNS", "FILTER_COLUMNS", "assemble_row"]


# --- column groups (kept as named tuples-of-strings for readability) ---

_IDENTIFICATION = (
    "task",
    "disruption",
    "obs_mode",
    "human_model",
    "policy_kind",
    "snapshot",
    "filter_snapshot",
    "filter_threshold",
    "seeds",
    "episodes_per_seed",
    "n_episodes",
    "n_steps",
    "git_sha",
    "timestamp_utc",
)

_TASK = (
    "success_rate",
    "success_rate_ci_lo",
    "success_rate_ci_hi",
    "episode_reward_mean",
    "episode_reward_ci_lo",
    "episode_reward_ci_hi",
    "mean_episode_length",
    "steps_to_completion",
    "steps_to_completion_ci_lo",
    "steps_to_completion_ci_hi",
)

_SAFETY = (
    # headline axis (+ bootstrap CI)
    "ep_proximity_violation_rate",
    "ep_proximity_violation_rate_ci_lo",
    "ep_proximity_violation_rate_ci_hi",
    # ISO flavours
    "ep_ssm_violation_rate",
    "ep_ssm_violation_actual_rate",
    "ep_ssm_violation_actual_rate_ci_lo",
    "ep_ssm_violation_actual_rate_ci_hi",
    "ep_pfl_violation_rate",
    # proximity dwell
    "ep_time_in_proximity_0p3m",
    "ep_time_in_proximity_0p5m",
    "ep_time_in_proximity_1p0m",
    # separation distribution (+ CI on the mean min-separation)
    "ep_min_separation",
    "ep_min_separation_ci_lo",
    "ep_min_separation_ci_hi",
    "ep_min_separation_lowest",
    "ep_mean_separation",
    "ep_p5_separation",
    "ep_p25_separation",
    # SSM margin troughs
    "ep_min_ssm_margin",
    "ep_min_ssm_margin_actual",
    # robot kinematics
    "ep_max_robot_vel",
    "ep_mean_robot_vel",
    # reaction-time diagnostic
    "ep_time_to_first_violation",
    # per-region PFL counts (inert under the open contact bug; always present)
    "pfl_violations_per_region_json",
)

_TAIL_RISK = (
    "cvar95_ep_cost_integral",
    "mean_ep_cost_integral",
    "cvar95_ep_min_separation",
    "p99_ep_min_separation",
    "p5_ep_min_separation",
)

# Only populated when a runtime filter is attached; "" otherwise.
FILTER_COLUMNS = (
    "filter_intervention_rate",
    "filter_intervention_rate_ci_lo",
    "filter_intervention_rate_ci_hi",
    "filter_passthrough_rate",
    "mean_per_episode_interventions",
    "mean_q_value",
    "n_interventions",
    "filter_fallback",
)

COLUMNS: List[str] = list(
    _IDENTIFICATION + _TASK + _SAFETY + _TAIL_RISK + FILTER_COLUMNS
)


def assemble_row(values: Dict[str, Any], *, filtered: bool) -> Dict[str, Any]:
    """Build one CSV row in ``COLUMNS`` order from a flat ``values`` dict.

    Any column absent from ``values`` is emitted as ``""``. Filter columns are forced
    to ``""`` when ``filtered`` is False so the header is identical across filtered and
    unfiltered cells.
    """
    filter_set = set(FILTER_COLUMNS)
    row: Dict[str, Any] = {}
    for col in COLUMNS:
        if col in filter_set and not filtered:
            row[col] = ""
        else:
            row[col] = values.get(col, "")
    return row
