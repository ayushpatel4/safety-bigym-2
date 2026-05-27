#!/usr/bin/env python3
"""Pick the best snapshot from a train_cqn_as stage directory.

Prefers ``snapshot_best.pt`` (written by train_cqn_as since the eval-aligned
snapshot change). Falls back to ``final_metrics.json``, then the eval row in
``metrics.jsonl`` with the highest ``eval/success_rate`` (nearest on-disk
``snapshot_<step>.pt`` for legacy runs), then the newest snapshot by mtime.

Usage::

    python scripts/pick_best_snapshot.py exp_local/.../stage0_idle
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


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <stage_dir>", file=sys.stderr)
        return 2
    path = pick_best_snapshot(Path(sys.argv[1]))
    if path is None:
        print(f"no snapshot found under {sys.argv[1]}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
