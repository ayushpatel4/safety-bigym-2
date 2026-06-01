#!/usr/bin/env python3
"""Pick the SVF veto-threshold operating point from a dense threshold sweep.

Seed-averages the per-seed sweep CSVs written by ``svf_threshold_sweep.py``
(columns: ``threshold_R, intervention_rate, proximity_violation_rate,
mean_q_value, ...``), computes proximity reduction vs the R=0 (filterless)
baseline, and reports the recommended operating point: the *smallest* R that
meets the P2 acceptance bar (proximity reduction >= --min-reduction at
intervention <= --max-intervention).

This is the single, reproducible knee pick that feeds ``snapshots.py``
(``SVF_FILTERS`` + ``SVF_FILTER_THRESHOLD_R``). Used by the P2 re-do
(``run_p2_recollect_g1.sh``) after the action-de-normalisation fix.

Usage:
    python scripts/analyze_svf_sweep.py --sweep-dir results/svf_sweep_g1_0p3_v2
    python scripts/analyze_svf_sweep.py --csv results/svf_sweep_g1_0p3_v2/sweep_dense_seed*.csv
    python scripts/analyze_svf_sweep.py --sweep-dir ... --max-intervention 0.25 --min-reduction 0.30

Exit code is 0 if a qualifying knee is found, 3 if none qualifies (so a
launcher can branch on "no acceptable operating point").
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _load_rows(csv_paths: List[Path]) -> Dict[float, Dict[str, List[float]]]:
    """Group metric lists by threshold_R across all seed CSVs."""
    by_r: Dict[float, Dict[str, List[float]]] = {}
    for p in csv_paths:
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                r = float(row["threshold_R"])
                slot = by_r.setdefault(r, {"intervention": [], "proximity": [], "mean_q": []})
                slot["intervention"].append(float(row["intervention_rate"]))
                slot["proximity"].append(float(row["proximity_violation_rate"]))
                # mean_q_value present in the dense schema; tolerate absence.
                if "mean_q_value" in row and row["mean_q_value"] not in ("", None):
                    slot["mean_q"].append(float(row["mean_q_value"]))
    return by_r


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def analyze(
    csv_paths: List[Path],
    *,
    max_intervention: float,
    min_reduction: float,
) -> Tuple[List[dict], dict | None]:
    by_r = _load_rows(csv_paths)
    if not by_r:
        raise SystemExit("No rows loaded — are the CSV paths correct?")

    rs = sorted(by_r)
    # Baseline proximity = the filterless R=0 row (closest-to-zero if exact 0 absent).
    base_r = 0.0 if 0.0 in by_r else min(rs, key=abs)
    base_prox = _mean(by_r[base_r]["proximity"])

    table: List[dict] = []
    for r in rs:
        interv = _mean(by_r[r]["intervention"])
        prox = _mean(by_r[r]["proximity"])
        reduction = (1.0 - prox / base_prox) if base_prox > 0 else float("nan")
        table.append(
            {
                "R": r,
                "intervention": interv,
                "proximity": prox,
                "reduction": reduction,
                "mean_q": _mean(by_r[r]["mean_q"]),
                "n_seeds": len(by_r[r]["intervention"]),
            }
        )

    # Knee = smallest R meeting BOTH bars.
    qualifying = [
        row
        for row in table
        if row["R"] > base_r
        and not math.isnan(row["reduction"])
        and row["reduction"] >= min_reduction
        and row["intervention"] <= max_intervention
    ]
    knee = min(qualifying, key=lambda row: row["R"]) if qualifying else None
    return table, knee


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sweep-dir", type=Path, default=None,
                    help="Dir of sweep_dense_seed*.csv (seed-averaged).")
    ap.add_argument("--csv", nargs="*", default=[],
                    help="Explicit CSV paths/globs (overrides --sweep-dir).")
    ap.add_argument("--max-intervention", type=float, default=0.25,
                    help="P2 acceptance: max seed-mean intervention rate (default 0.25).")
    ap.add_argument("--min-reduction", type=float, default=0.30,
                    help="P2 acceptance: min proximity reduction vs R=0 (default 0.30).")
    args = ap.parse_args()

    paths: List[Path] = []
    if args.csv:
        for pat in args.csv:
            paths.extend(Path(p) for p in glob.glob(pat))
    elif args.sweep_dir:
        paths = [Path(p) for p in glob.glob(str(args.sweep_dir / "sweep_dense_seed*.csv"))]
        if not paths:  # fall back to any sweep CSV in the dir
            paths = [Path(p) for p in glob.glob(str(args.sweep_dir / "sweep*seed*.csv"))]
    if not paths:
        ap.error("No CSVs found — pass --sweep-dir <dir> or --csv <glob>.")

    table, knee = analyze(
        sorted(paths),
        max_intervention=args.max_intervention,
        min_reduction=args.min_reduction,
    )

    print(f"# Seed-averaged over {len(paths)} CSV(s): {', '.join(p.name for p in sorted(paths))}")
    print(f"# Acceptance bar: reduction >= {args.min_reduction:.0%} at intervention <= {args.max_intervention:.0%}\n")
    print(f"{'R':>6}  {'interv':>8}  {'proximity':>10}  {'reduction':>10}  {'mean_q':>8}  seeds")
    for row in table:
        mark = "  <- KNEE" if (knee and row["R"] == knee["R"]) else ""
        red = "baseline" if row["reduction"] != row["reduction"] or row["R"] == (0.0 if any(r["R"] == 0.0 for r in table) else table[0]["R"]) else f"{row['reduction']:+.1%}"
        print(f"{row['R']:>6.2f}  {row['intervention']:>8.3f}  {row['proximity']:>10.4f}  "
              f"{red:>10}  {row['mean_q']:>8.3f}  {row['n_seeds']:>5d}{mark}")

    print()
    if knee:
        print(f"RECOMMENDED OPERATING POINT: R = {knee['R']:.2f}")
        print(f"  proximity reduction {knee['reduction']:.1%} @ intervention {knee['intervention']:.1%} "
              f"(mean_q {knee['mean_q']:.3f})")
        print(f"\nUpdate safety_bigym/filters/snapshots.py:")
        print(f"  SVF_FILTER_THRESHOLD_R['<task>'] = {knee['R']:.2f}")
        sys.exit(0)
    else:
        print("NO QUALIFYING KNEE — no R meets both bars. Options: relax --max-intervention,")
        print("accept a lower --min-reduction, or report the filter as a velocity-axis-only win.")
        sys.exit(3)


if __name__ == "__main__":
    main()
