"""Tests for scripts/pick_best_snapshot.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.pick_best_snapshot import pick_best_snapshot


def test_pick_prefers_snapshot_best_pt(tmp_path: Path):
    stage = tmp_path / "stage0"
    stage.mkdir()
    (stage / "snapshot_5000.pt").write_bytes(b"old")
    (stage / "snapshot_best.pt").write_bytes(b"best")
    assert pick_best_snapshot(stage) == stage / "snapshot_best.pt"


def test_pick_from_metrics_jsonl_nearest_legacy(tmp_path: Path):
    stage = tmp_path / "stage0"
    stage.mkdir()
    (stage / "snapshot_10000.pt").write_bytes(b"a")
    (stage / "snapshot_20000.pt").write_bytes(b"b")
    rows = [
        {"step": 2500, "ty": "eval", "eval/success_rate": 0.3},
        {"step": 13185, "ty": "eval", "eval/success_rate": 0.5, "eval/episode_reward": -2.0},
        {"step": 28253, "ty": "eval", "eval/success_rate": 0.1},
    ]
    (stage / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    # Peak eval at 13185; legacy 10k cadence → nearest snapshot_10000.pt.
    assert pick_best_snapshot(stage) == stage / "snapshot_10000.pt"
