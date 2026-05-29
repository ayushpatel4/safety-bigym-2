"""Aggregate a cell's per-episode records into the flat per-cell column dict.

``aggregate_cell`` consumes the :class:`~safety_bigym.benchmark.records.EpisodeRecord`
list for one (task, disruption, obs-mode, seeds) cell and produces every non-identification
value in :data:`safety_bigym.benchmark.schema.COLUMNS` — episode means, bootstrap CIs on the
headline axes, CVaR/percentile tail-risk, and filter mechanics. The CLI merges this with the
identification fields and hands the result to :func:`schema.assemble_row`.

Pure (numpy + stdlib + the pure ``stats`` module). No torch / mujoco / pandas.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from safety_bigym.benchmark import stats
from safety_bigym.benchmark.records import EpisodeRecord

__all__ = ["aggregate_cell"]


def _ep_field(records: Sequence[EpisodeRecord], key: str, default: float = float("nan")) -> List[float]:
    return [float(r.ep_safety.get(key, default)) for r in records]


def _mean(xs: Sequence[float]) -> float:
    arr = np.asarray([x for x in xs if math.isfinite(x)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def aggregate_cell(
    records: List[EpisodeRecord],
    *,
    filter_meta: Optional[Dict[str, Any]] = None,
    stats_seed: int = 12345,
    n_resamples: int = 10_000,
) -> Dict[str, Any]:
    """Reduce per-episode records to a flat dict of per-cell column values.

    ``filter_meta`` (e.g. ``{"fallback": "zero_velocity"}``) signals a filter was
    attached → the filter-mechanics columns are populated. CIs are reproducible via
    ``stats_seed``.
    """
    out: Dict[str, Any] = {}
    if not records:
        return out

    filtered = filter_meta is not None

    # ---- per-episode arrays ----
    successes = [1.0 if r.success else 0.0 for r in records]
    rewards = [float(r.episode_reward) for r in records]
    lengths = [float(r.n_steps) for r in records]
    cost_integrals = [float(r.ep_cost_integral) for r in records]
    min_seps = _ep_field(records, "ep_min_separation")
    stc = [float(r.steps_to_completion) for r in records if r.success and math.isfinite(r.steps_to_completion)]

    def ci(samples, agg=np.mean):
        return stats.bootstrap_ci(samples, agg=agg, n_resamples=n_resamples, seed=stats_seed)

    # ---- identification helpers (derived) ----
    out["n_episodes"] = len(records)
    out["n_steps"] = int(sum(int(r.n_steps) for r in records))

    # ---- task ----
    sr, sr_lo, sr_hi = ci(successes)
    out["success_rate"], out["success_rate_ci_lo"], out["success_rate_ci_hi"] = sr, sr_lo, sr_hi
    er, er_lo, er_hi = ci(rewards)
    out["episode_reward_mean"], out["episode_reward_ci_lo"], out["episode_reward_ci_hi"] = er, er_lo, er_hi
    out["mean_episode_length"] = _mean(lengths)
    if stc:
        s, s_lo, s_hi = ci(stc)
    else:
        s = s_lo = s_hi = float("nan")
    out["steps_to_completion"], out["steps_to_completion_ci_lo"], out["steps_to_completion_ci_hi"] = s, s_lo, s_hi

    # ---- safety (episode means; CI on the three headline axes) ----
    prox = _ep_field(records, "ep_proximity_violation_rate")
    p, p_lo, p_hi = ci(prox)
    out["ep_proximity_violation_rate"], out["ep_proximity_violation_rate_ci_lo"], out["ep_proximity_violation_rate_ci_hi"] = p, p_lo, p_hi

    out["ep_ssm_violation_rate"] = _mean(_ep_field(records, "ep_ssm_violation_rate"))
    ssa = _ep_field(records, "ep_ssm_violation_actual_rate")
    a, a_lo, a_hi = ci(ssa)
    out["ep_ssm_violation_actual_rate"], out["ep_ssm_violation_actual_rate_ci_lo"], out["ep_ssm_violation_actual_rate_ci_hi"] = a, a_lo, a_hi
    out["ep_pfl_violation_rate"] = _mean(_ep_field(records, "ep_pfl_violation_rate"))

    out["ep_time_in_proximity_0p3m"] = _mean(_ep_field(records, "ep_time_in_proximity_0p3m"))
    out["ep_time_in_proximity_0p5m"] = _mean(_ep_field(records, "ep_time_in_proximity_0p5m"))
    out["ep_time_in_proximity_1p0m"] = _mean(_ep_field(records, "ep_time_in_proximity_1p0m"))

    ms, ms_lo, ms_hi = ci(min_seps)
    out["ep_min_separation"], out["ep_min_separation_ci_lo"], out["ep_min_separation_ci_hi"] = ms, ms_lo, ms_hi
    finite_min = [x for x in min_seps if math.isfinite(x)]
    out["ep_min_separation_lowest"] = float(min(finite_min)) if finite_min else float("nan")
    out["ep_mean_separation"] = _mean(_ep_field(records, "ep_mean_separation"))
    out["ep_p5_separation"] = _mean(_ep_field(records, "ep_p5_separation"))
    out["ep_p25_separation"] = _mean(_ep_field(records, "ep_p25_separation"))

    out["ep_min_ssm_margin"] = _mean(_ep_field(records, "ep_min_ssm_margin"))
    out["ep_min_ssm_margin_actual"] = _mean(_ep_field(records, "ep_min_ssm_margin_actual"))
    out["ep_max_robot_vel"] = _mean(_ep_field(records, "ep_max_robot_vel"))
    out["ep_mean_robot_vel"] = _mean(_ep_field(records, "ep_mean_robot_vel"))

    ttf = [v for v in _ep_field(records, "ep_time_to_first_violation", -1.0) if v != -1.0]
    out["ep_time_to_first_violation"] = _mean(ttf) if ttf else float("nan")

    # Per-region PFL counts: sum across episodes of any ep_region_* keys.
    region_totals: Dict[str, int] = {}
    for r in records:
        for k, v in r.ep_safety.items():
            if k.startswith("ep_region_") and isinstance(v, (int, float)):
                region = k[len("ep_region_"):]
                region_totals[region] = region_totals.get(region, 0) + int(v)
    out["pfl_violations_per_region_json"] = json.dumps(region_totals)

    # ---- tail-risk (over per-episode arrays) ----
    out["cvar95_ep_cost_integral"] = stats.cvar(cost_integrals, q=0.95, tail="upper")
    out["mean_ep_cost_integral"] = _mean(cost_integrals)
    out["cvar95_ep_min_separation"] = stats.cvar(min_seps, q=0.95, tail="lower")
    out["p99_ep_min_separation"] = stats.percentile(min_seps, 1.0)
    out["p5_ep_min_separation"] = stats.percentile(min_seps, 5.0)

    # ---- filter mechanics ----
    if filtered:
        total_interventions = int(sum(r.n_interventions for r in records))
        total_filter_steps = int(sum(r.filter_steps for r in records))
        per_ep_rate = [
            (r.n_interventions / r.filter_steps) if r.filter_steps else 0.0 for r in records
        ]
        rate = (total_interventions / total_filter_steps) if total_filter_steps else float("nan")
        _, ir_lo, ir_hi = ci(per_ep_rate)
        out["filter_intervention_rate"] = rate
        out["filter_intervention_rate_ci_lo"] = ir_lo
        out["filter_intervention_rate_ci_hi"] = ir_hi
        out["filter_passthrough_rate"] = (1.0 - rate) if math.isfinite(rate) else float("nan")
        out["mean_per_episode_interventions"] = _mean([float(r.n_interventions) for r in records])
        out["mean_q_value"] = (
            sum(r.sum_q_value for r in records) / total_filter_steps
            if total_filter_steps else float("nan")
        )
        out["n_interventions"] = total_interventions
        out["filter_fallback"] = str(filter_meta.get("fallback", ""))

    return out
