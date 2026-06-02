#!/usr/bin/env python3
"""ROW3 operating-point selection + 3-seed aggregation for the d0.3 Lagrangian.

Two subcommands:

``pick``  — from one seed's basin-sweep dir (``run_basin_sweep.sh`` output, one
            ``s<step>.csv`` per checkpoint), choose the operating point: the
            LOWEST deployment proximity-violation rate among checkpoints whose
            success_rate >= ``--success-floor``. This is the deployment-confirmed
            analogue of ``pick_best_snapshot.py --by safety`` (which nominates
            from train-eval) — here we pick from the actual benchmark, so it is
            robust to the noisy train-eval signal. Prints the snapshot path.

``aggregate`` — pool the per-episode rows of the chosen operating points across
            the 3 CONFIRM seeds (their ``*.episodes.jsonl`` benchmark files) into
            one ROW3 number with a bootstrap CI. Pools episodes (not means), so
            the CI reflects all 180 episodes. Reports proximity, success, mean
            robot vel, and ssm-actual vs a ``--baseline-prox`` reference.

Usage::

    # per seed, after run_basin_sweep.sh:
    python scripts/analyze_row3.py pick --sweep-dir results/e4_1/basin_d0p3_seed1_noisy --success-floor 0.75
    # after benchmarking each seed's operating point on noisy (or oracle):
    python scripts/analyze_row3.py aggregate --episodes \
        results/e4_1/row3_final/seed0_noisy.episodes.jsonl \
        results/e4_1/row3_final/seed1_noisy.episodes.jsonl \
        results/e4_1/row3_final/seed2_noisy.episodes.jsonl \
        --baseline-prox 0.296
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# pick
# --------------------------------------------------------------------------- #
def _step_from_snapshot(path: str) -> int | None:
    base = os.path.basename(str(path))
    if not base.startswith("snapshot_") or not base.endswith(".pt"):
        return None
    core = base[len("snapshot_"):-len(".pt")]
    return int(core) if core.isdigit() else None


def pick_operating_point(sweep_dir: Path, success_floor: float) -> dict | None:
    """Lowest deploy proximity among sweep checkpoints with success >= floor.

    Tie-break: higher success, then earlier step (more stable). Returns a dict
    with step/proximity/success/snapshot, or None if nothing clears the floor.
    """
    rows = []
    for f in sorted(Path(sweep_dir).glob("s*.csv")):
        try:
            r = pd.read_csv(f).iloc[-1]
        except Exception:
            continue
        sr = r.get("success_rate")
        prox = r.get("ep_proximity_violation_rate")
        if not isinstance(sr, (int, float)) or not isinstance(prox, (int, float)):
            continue
        rows.append({
            "step": _step_from_snapshot(r.get("snapshot", "")),
            "prox": float(prox), "succ": float(sr),
            "snapshot": str(r.get("snapshot", "")),
            "vel": float(r.get("ep_mean_robot_vel", float("nan"))),
        })
    cand = [r for r in rows if r["succ"] >= success_floor]
    if not cand:
        return None
    cand.sort(key=lambda r: (r["prox"], -r["succ"], r["step"] if r["step"] is not None else 1 << 30))
    return cand[0]


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def _load_episodes(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        recs = [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]
        df = pd.DataFrame(recs)
        df["__src"] = os.path.basename(str(p))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(values, n_resamples: int = 10000, seed: int = 12345):
    """Percentile bootstrap CI of the mean over episode-level values."""
    import numpy as np
    v = np.asarray([x for x in values if x is not None and not pd.isna(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(paths, baseline_prox=None) -> int:
    df = _load_episodes(paths)
    # NB: the per-episode ``seed`` field is the rollout seed (shared across files),
    # NOT the training seed — so we report files pooled, one per train-seed by convention.
    print(f"== ROW3 aggregate | {len(paths)} operating-point benchmark(s) pooled, {len(df)} episodes ==")
    fields = [
        ("ep_proximity_violation_rate", "proximity (τ=0.3)"),
        ("success", "success"),
        ("ep_mean_robot_vel", "mean robot vel"),
        ("ep_ssm_violation_actual_rate", "ssm-actual"),
    ]
    prox_mean = None
    for key, label in fields:
        if key not in df.columns:
            continue
        m, lo, hi = bootstrap_ci(df[key].tolist())
        if key == "ep_proximity_violation_rate":
            prox_mean = m
        print(f"  {label:20s} {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    if baseline_prox is not None and prox_mean is not None:
        d = prox_mean - baseline_prox
        print(f"\n  proximity vs baseline {baseline_prox:.3f}:  {d:+.3f}  ({100*d/baseline_prox:+.1f}%)")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pick", help="pick a seed's ROW3 operating point from its basin sweep")
    p.add_argument("--sweep-dir", type=Path, required=True)
    p.add_argument("--success-floor", type=float, default=0.75)

    a = sub.add_parser("aggregate", help="pool operating-point benchmarks across seeds")
    a.add_argument("--episodes", nargs="+", required=True, help="per-seed *.episodes.jsonl files")
    a.add_argument("--baseline-prox", type=float, default=None)

    args = ap.parse_args()
    if args.cmd == "pick":
        op = pick_operating_point(args.sweep_dir, args.success_floor)
        if op is None:
            print(f"no checkpoint clears success>={args.success_floor} under {args.sweep_dir}; "
                  f"loosen --success-floor or widen STEPS", flush=True)
            return 1
        print(f"# step={op['step']} prox={op['prox']:.3f} succ={op['succ']:.2f} vel={op['vel']:.3f}")
        print(op["snapshot"])
        return 0
    return aggregate(args.episodes, args.baseline_prox)


if __name__ == "__main__":
    raise SystemExit(main())
