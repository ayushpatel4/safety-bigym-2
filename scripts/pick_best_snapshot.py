#!/usr/bin/env python3
"""Pick the best snapshot from a train_cqn_as stage directory.

Prefers ``snapshot_best.pt`` (written by train_cqn_as since the eval-aligned
snapshot change). Falls back to ``final_metrics.json``, then the eval row in
``metrics.jsonl`` with the highest ``eval/success_rate`` (nearest on-disk
``snapshot_<step>.pt`` for legacy runs), then the newest snapshot by mtime.

For a constrained (Lagrangian) run, peak-success selection is the WRONG default
— it picks against the cost constraint. Use ``--by safety`` to pick the lowest
eval proximity-violation rate among checkpoints clearing a success floor (the
ROW3 criterion); always benchmark-confirm the nominee.

Usage::

    python scripts/pick_best_snapshot.py exp_local/.../stage0_idle
    python scripts/pick_best_snapshot.py exp_local/.../d0p3_seed0 --by safety --success-floor 0.75
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _snapshot_step(path: Path) -> int:
    return int(path.stem.removeprefix("snapshot_"))


def _nearest_snapshot(stage_dir: Path, target_step: int) -> Path | None:
    snaps = [
        p for p in stage_dir.glob("snapshot_*.pt")
        if p.name != "snapshot_best.pt"
    ]
    if not snaps:
        return None
    before = [p for p in snaps if _snapshot_step(p) <= target_step]
    if before:
        return max(before, key=_snapshot_step)
    return min(snaps, key=_snapshot_step)


def _best_from_metrics_jsonl(stage_dir: Path) -> Path | None:
    metrics_path = stage_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    best_step: int | None = None
    best_sr = float("-inf")
    best_er = float("-inf")
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ty") != "eval":
            continue
        sr = row.get("eval/success_rate")
        if not isinstance(sr, (int, float)):
            continue
        er = row.get("eval/episode_reward")
        er_f = float(er) if isinstance(er, (int, float)) else float("-inf")
        sr_f = float(sr)
        if sr_f > best_sr or (sr_f == best_sr and er_f > best_er):
            best_sr, best_er, best_step = sr_f, er_f, int(row["step"])
    if best_step is None:
        return None
    exact = stage_dir / f"snapshot_{best_step}.pt"
    if exact.is_file():
        return exact
    return _nearest_snapshot(stage_dir, best_step)


def pick_best_snapshot(stage_dir: Path) -> Path | None:
    stage_dir = Path(stage_dir)
    best = stage_dir / "snapshot_best.pt"
    if best.is_file():
        return best

    final_path = stage_dir / "final_metrics.json"
    if final_path.is_file():
        try:
            data = json.loads(final_path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        bs = data.get("best_snapshot") or {}
        rel = bs.get("path") or bs.get("source")
        if rel:
            cand = stage_dir / rel
            if cand.is_file():
                return cand

    from_jsonl = _best_from_metrics_jsonl(stage_dir)
    if from_jsonl is not None:
        return from_jsonl

    snaps = sorted(
        [
            p for p in stage_dir.glob("snapshot_*.pt")
            if p.name != "snapshot_best.pt"
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snaps[0] if snaps else None


def _safe_from_metrics_jsonl(stage_dir: Path, success_floor: float) -> Path | None:
    """Eval row with the LOWEST ``eval/ep_proximity_violation_rate`` among rows
    whose ``eval/success_rate >= success_floor``.

    This is the safety-aware ROW3 criterion for a Lagrangian policy: the most
    proximity-avoiding checkpoint that still does the task. It exists because
    ``pick_best_snapshot`` (peak eval success) selects AGAINST the constraint —
    for d=0.3/seed0 it returned a ~baseline-proximity checkpoint while the
    avoidance lived at a mid-training step (benchmark: 0.30 vs 0.23). Tie-break:
    higher success, then earlier step (more stable, less reward-overfit).
    """
    metrics_path = stage_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    best_step: int | None = None
    best_prox = float("inf")
    best_sr = float("-inf")
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ty") != "eval":
            continue
        sr = row.get("eval/success_rate")
        prox = row.get("eval/ep_proximity_violation_rate")
        if not isinstance(sr, (int, float)) or not isinstance(prox, (int, float)):
            continue
        sr_f, prox_f = float(sr), float(prox)
        if sr_f < success_floor:
            continue
        step = int(row["step"])
        if (
            prox_f < best_prox
            or (prox_f == best_prox and sr_f > best_sr)
            or (prox_f == best_prox and sr_f == best_sr
                and (best_step is None or step < best_step))
        ):
            best_prox, best_sr, best_step = prox_f, sr_f, step
    if best_step is None:
        return None
    exact = stage_dir / f"snapshot_{best_step}.pt"
    if exact.is_file():
        return exact
    return _nearest_snapshot(stage_dir, best_step)


def pick_safe_snapshot(stage_dir: Path, success_floor: float = 0.75) -> Path | None:
    """Safety-aware pick: lowest eval proximity s.t. success >= ``success_floor``.

    Deliberately ignores ``snapshot_best.pt`` (peak success, biased against the
    constraint). Falls back to :func:`pick_best_snapshot` only when no eval row
    clears the floor (or there are no metrics). Always benchmark-confirm the
    result — the eval-proximity signal is a noisy nomination, not the measurement.
    """
    stage_dir = Path(stage_dir)
    safe = _safe_from_metrics_jsonl(stage_dir, success_floor)
    if safe is not None:
        return safe
    return pick_best_snapshot(stage_dir)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Pick a snapshot from a train_cqn_as stage directory."
    )
    ap.add_argument("stage_dir", type=Path)
    ap.add_argument(
        "--by", choices=("success", "safety"), default="success",
        help="success = peak eval success_rate (default, legacy); "
             "safety = lowest eval proximity s.t. success>=floor (Lagrangian ROW3).",
    )
    ap.add_argument(
        "--success-floor", type=float, default=0.75,
        help="Min eval success_rate for --by safety (default 0.75).",
    )
    args = ap.parse_args()
    if args.by == "safety":
        path = pick_safe_snapshot(args.stage_dir, args.success_floor)
    else:
        path = pick_best_snapshot(args.stage_dir)
    if path is None:
        print(f"no snapshot found under {args.stage_dir}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
