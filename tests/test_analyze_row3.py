"""ROW3 operating-point pick + 3-seed pooling (analyze_row3)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ar3", Path(__file__).resolve().parents[1] / "scripts" / "analyze_row3.py"
)
ar3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ar3)

_COLS = "snapshot,success_rate,ep_proximity_violation_rate,ep_mean_robot_vel"


def _sweep_csv(stage_dir: Path, step: int, succ: float, prox: float):
    (stage_dir / f"s{step}.csv").write_text(
        f"{_COLS}\n/run/d0p3_seed0/snapshot_{step}.pt,{succ},{prox},0.3\n"
    )


def test_pick_min_proximity_above_floor(tmp_path):
    _sweep_csv(tmp_path, 20279, 0.78, 0.251)
    _sweep_csv(tmp_path, 27554, 0.78, 0.234)   # min prox among succ>=0.75
    _sweep_csv(tmp_path, 33386, 0.72, 0.214)   # lower prox but BELOW floor -> excluded
    op = ar3.pick_operating_point(tmp_path, success_floor=0.75)
    assert op["step"] == 27554
    assert op["snapshot"].endswith("snapshot_27554.pt")


def test_pick_returns_none_when_floor_unmet(tmp_path):
    _sweep_csv(tmp_path, 100, 0.40, 0.05)
    _sweep_csv(tmp_path, 200, 0.50, 0.06)
    assert ar3.pick_operating_point(tmp_path, success_floor=0.75) is None


def test_pick_tiebreak_prefers_higher_success_then_earlier_step(tmp_path):
    _sweep_csv(tmp_path, 300, 0.80, 0.230)
    _sweep_csv(tmp_path, 100, 0.90, 0.230)   # same prox, higher succ -> wins
    _sweep_csv(tmp_path, 200, 0.90, 0.230)   # same prox+succ, later step -> loses to 100
    assert ar3.pick_operating_point(tmp_path, success_floor=0.75)["step"] == 100


def test_bootstrap_ci_brackets_mean_and_is_deterministic(tmp_path):
    vals = [0.1, 0.2, 0.3, 0.4, 0.5] * 12  # 60 episodes
    m, lo, hi = ar3.bootstrap_ci(vals, n_resamples=2000, seed=7)
    assert abs(m - 0.3) < 1e-9
    assert lo < m < hi
    # deterministic under fixed seed
    assert ar3.bootstrap_ci(vals, n_resamples=2000, seed=7) == (m, lo, hi)
