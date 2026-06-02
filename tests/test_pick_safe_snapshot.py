"""Safety-aware snapshot selection (Lagrangian ROW3 criterion).

Guards the regression that produced two false-null benchmark waves: the
peak-success picker (``pick_best_snapshot`` / ``snapshot_best.pt``) selects
AGAINST the cost constraint, so for a Lagrangian run it returns a
~baseline-proximity checkpoint while the avoidance lives at a mid-training
step. ``pick_safe_snapshot`` must instead pick the lowest-proximity checkpoint
that still clears a success floor.
"""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "pbs", Path(__file__).resolve().parents[1] / "scripts" / "pick_best_snapshot.py"
)
pbs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pbs)


def _eval_row(step, sr, prox):
    return {
        "ty": "eval", "step": step,
        "eval/success_rate": sr,
        "eval/ep_proximity_violation_rate": prox,
        "eval/episode_reward": 1.0,
    }


def _make_stage(tmp_path, rows, steps, with_best=True):
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    for s in steps:
        (tmp_path / f"snapshot_{s}.pt").write_bytes(b"x")
    if with_best:
        (tmp_path / "snapshot_best.pt").write_bytes(b"x")
    return tmp_path


def test_safety_pick_takes_min_proximity_above_floor(tmp_path):
    # Mirrors d0p3_seed0: all three clear the floor; avoidance is mid-run.
    rows = [_eval_row(10, 0.9, 0.30), _eval_row(20, 0.9, 0.12), _eval_row(30, 0.9, 0.24)]
    stage = _make_stage(tmp_path, rows, steps=(10, 20, 30))
    assert pbs.pick_safe_snapshot(stage, success_floor=0.75).name == "snapshot_20.pt"
    # The legacy success picker takes the peak-success shortcut instead.
    assert pbs.pick_best_snapshot(stage).name == "snapshot_best.pt"


def test_safety_pick_respects_success_floor(tmp_path):
    # The lowest-proximity row (step 10) is below the floor and must be skipped.
    rows = [_eval_row(10, 0.40, 0.05), _eval_row(20, 0.80, 0.22)]
    stage = _make_stage(tmp_path, rows, steps=(10, 20), with_best=False)
    assert pbs.pick_safe_snapshot(stage, success_floor=0.75).name == "snapshot_20.pt"


def test_safety_pick_falls_back_when_no_row_clears_floor(tmp_path):
    rows = [_eval_row(10, 0.40, 0.05), _eval_row(20, 0.50, 0.06)]
    stage = _make_stage(tmp_path, rows, steps=(10, 20))
    # Nothing clears 0.75 -> fall back to the success-based pick (snapshot_best.pt).
    assert pbs.pick_safe_snapshot(stage, success_floor=0.75).name == "snapshot_best.pt"
