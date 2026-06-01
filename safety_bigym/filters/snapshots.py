"""Per-task snapshot path lookup — single source of truth for SVF scripts.

Update :data:`SNAPSHOTS` after each Phase-0 retrain by inspecting W&B
``pretrain_eval/episode_success`` curves and picking the peak step. See
the session-handling rules in CLAUDE.md.

Tasks left as ``None`` are *deliberately* skipped by the snapshot-source
collection / snapshot-policy eval paths. A task whose value points to a
nonexistent file raises :class:`FileNotFoundError` from
:func:`resolve_snapshot` — that's a typo / stale path, not a deliberate skip.

Paths are resolved relative to the safety_bigym repo root (i.e. the parent
of this package), so the dict stays portable across local and GPU layouts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

# Repo root is two levels up from this file: safety_bigym/filters/snapshots.py
_REPO_ROOT = Path(__file__).resolve().parents[2]


# Keyed by safety_bigym task_key. Values are paths relative to the repo root,
# absolute paths, or ``None`` for "no snapshot yet — skip this task".
#
# Set after each retrain; pick the W&B `pretrain_eval/episode_success` peak.
#
# Phase 0.5 task pool (per .claude/IMPLEMENTATION_STATUS.md, 2026-05-15):
# the three working long-horizon tasks are dishwasher_close, drawers_open_all,
# saucepan_to_hob. reach_target_single excluded by user (horizon too short).
# B1 ACT re-rolls on COWORKER train space have been run on the GPU box;
# paths pending — fill in once W&B `pretrain_eval/episode_success` peaks
# are identified. Until then `--source snapshot` collection skips these
# three tasks (returns None from resolve_snapshot → ignored by caller).
SNAPSHOTS: Dict[str, Optional[str]] = {
    "reach_target_single": None,
    "dishwasher_close": "~/Documents/safety_bigym/exp_local/act_safety/dishwasher_close_20260515184635/snapshots/50000_snapshot.pt",
    "dishwasher_load_plates": None,
    "saucepan_to_hob": "~/Documents/safety_bigym/exp_local/act_safety/saucepan_to_hob_20260516123308/snapshots/70000_snapshot.pt",
    "drawers_open_all": "~/Documents/safety_bigym/exp_local/act_safety/drawers_open_all_20260515184721/snapshots/40000_snapshot.pt",
}


# ---------------------------------------------------------------------------
# SVF runtime-filter operating points (Phase 2 → headline E4.1 rows 4/5).
#
# Each task maps to (trained SVF critic checkpoint, recommended veto threshold
# R). SafetyFilterWrapper vetoes the proposed action when Q_safe(s, a) < R.
#
# G1 coworker — v3 PENDING (2026-06-01). The SVF critic had TWO bugs where the
# collection policy (`_CQNASSnapshotPolicy`) didn't match the deployed policy, so
# the critic trained on actions deployment never executes and over-vetoes:
#   1. action de-normalisation via env.action_space, not the agent's demo-derived
#      stats (fixed `41fd93b`; the v2 re-collect addressed this).
#   2. action-execution mode: collection ran RECEDING-HORIZON (raw chunk[0]) while
#      deployment runs an action_sequence=16 + temporal_ensemble=true BLEND.
#      The v2 critic (trained on chunk[0]) thus vetoed ~89% of deployed actions
#      (E4.1 row-4: 89.5% intervention, mean_q 0.97, success 0). Fixed: the
#      snapshot policy now mirrors benchmark CQNASRunner.step (open-loop chunks +
#      the SAME TemporalEnsembleControl) — tests/test_snapshot_policy_execution.py.
#
# The path below points at the v3 target (NOT yet collected) so filtered E4.1/E4.3
# rows fail LOUD rather than silently re-running the superseded v2 critic. Produce
# it with `scripts/run_p2_recollect_g1.sh` (VER=v3, the default), then update R
# below from `analyze_svf_sweep.py --sweep-dir results/svf_sweep_g1_0p3_v3`.
#
# SUPERSEDED, do NOT use for deployment:
#   - v2 (svf_coworker_train_g1_0p3_v2.pt, R=2.50; sweep results/svf_sweep_g1_0p3_v2/):
#     de-norm-fixed but trained on receding-horizon chunk[0]. Its sweep looked good
#     (31.9% reduction @ 7.9%) ONLY because the sweep reuses the collection path
#     (in-distribution); the benchmark exposed the gap. Bug 2 above.
#   - v1 (svf_coworker_train_g1_0p3.pt, R=2.25; results/svf_sweep_g1_v1/): mis-de-
#     normalised policy. Bug 1 above.
SVF_FILTERS: Dict[str, Optional[str]] = {
    # v3 target — fails loud until run_p2_recollect_g1.sh produces it.
    "saucepan_to_hob": "checkpoints/svf_coworker_train_g1_0p3_v3.pt",
}

# Recommended veto threshold R per task. PLACEHOLDER (2.50 carried from the
# superseded v2 knee) — RE-PICK from the v3 sweep via analyze_svf_sweep.py once
# the v3 re-collect lands. Unused until the v3 checkpoint exists (resolve_svf_filter
# FileNotFoundErrors on the missing path first).
SVF_FILTER_THRESHOLD_R: Dict[str, float] = {
    "saucepan_to_hob": 2.50,
}


# G1 base-policy curriculum snapshots (P1 — run base_g1_30k_30k_40k_20260529_124749).
#   stage1 (coworker_easy) = warm-start for P3 (E3.1) and P4 (E3.2) Lagrangian runs.
#   stage2 (coworker_train) = the unconstrained baseline: row-1 reference, the
#     policy the P2 R/baseline sweep rolls out, and the P5 row-1/row-4 eval input.
# Stored repo-relative so they resolve on both the GPU box and a local clone
# (resolved via _resolve_path against the safety_bigym repo root).
G1_CURRICULUM: Dict[str, Dict[str, str]] = {
    "saucepan_to_hob": {
        "stage1": "exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage1_easy/snapshot_2588.pt",
        "stage2": "exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage2_full/snapshot_28203.pt",
    },
}


def _resolve_path(value: str) -> Path:
    p = Path(os.path.expanduser(value))
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def resolve_snapshot(
    task_key: str,
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Return the on-disk path for ``task_key``, or ``None`` if deliberately unset.

    Parameters
    ----------
    task_key
        The task name (matches ``TASK_REGISTRY`` keys in svf_collect_dataset.py).
    overrides
        Optional per-task override mapping — typically populated from CLI flags.
        Values in this mapping take precedence over :data:`SNAPSHOTS`.

    Raises
    ------
    KeyError
        If ``task_key`` is not in :data:`SNAPSHOTS` and no override is provided.
    FileNotFoundError
        If the resolved path is non-empty but doesn't exist on disk (a stale
        / typo'd entry — distinct from deliberate ``None``).
    """
    raw = None
    if overrides and task_key in overrides:
        raw = overrides[task_key]
    elif task_key in SNAPSHOTS:
        raw = SNAPSHOTS[task_key]
    else:
        raise KeyError(
            f"Unknown task {task_key!r} — add it to SNAPSHOTS in "
            "safety_bigym/filters/snapshots.py"
        )

    if raw is None or raw == "":
        return None

    path = _resolve_path(str(raw))
    if not path.is_file():
        raise FileNotFoundError(
            f"Snapshot for task {task_key!r} listed but missing on disk: {path}. "
            "Update SNAPSHOTS in safety_bigym/filters/snapshots.py."
        )
    return path


def resolve_svf_filter(
    task_key: str,
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> Optional[tuple[Path, float]]:
    """Return ``(critic_path, threshold_R)`` for ``task_key``, or ``None`` if unset.

    Mirrors :func:`resolve_snapshot` semantics: ``None`` means "no SVF filter
    configured for this task" (deliberate skip); a listed-but-missing checkpoint
    raises :class:`FileNotFoundError` (stale / typo'd path). The threshold falls
    back to ``SVF_FILTER_THRESHOLD_R`` (default 4.0 if the task is absent).
    """
    raw = None
    if overrides and task_key in overrides:
        raw = overrides[task_key]
    elif task_key in SVF_FILTERS:
        raw = SVF_FILTERS[task_key]

    if raw is None or raw == "":
        return None

    path = _resolve_path(str(raw))
    if not path.is_file():
        raise FileNotFoundError(
            f"SVF filter for task {task_key!r} listed but missing on disk: {path}. "
            "Update SVF_FILTERS in safety_bigym/filters/snapshots.py."
        )
    return path, float(SVF_FILTER_THRESHOLD_R.get(task_key, 4.0))


__all__ = [
    "SNAPSHOTS",
    "resolve_snapshot",
    "SVF_FILTERS",
    "SVF_FILTER_THRESHOLD_R",
    "resolve_svf_filter",
    "G1_CURRICULUM",
]
