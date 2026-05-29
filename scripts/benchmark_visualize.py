#!/usr/bin/env python
"""Visualize benchmark_policy.py outputs — Pareto, per-cell bars, separation distribution.

Reads one or more per-cell CSVs (and their ``*.raw_episodes.parquet`` sidecars) and writes
PNG figures next to them. Stdlib ``csv`` + matplotlib (Agg backend, headless-safe).

    python scripts/benchmark_visualize.py --csv results/row1.csv results/row5.csv \\
        --out-dir results/figs

Figures
-------
1. ``pareto.png``      — filter_intervention_rate (x) vs ep_proximity_violation_rate (y),
                         one point per cell, CI error bars. The headline safety/cost plot.
2. ``cells_bars.png``  — grouped bars of proximity / ssm-actual violation rate + success
                         rate per cell.
3. ``separation.png``  — per-episode ep_min_separation distribution per cell (from parquet),
                         with p99 / cvar95 markers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("benchmark_visualize")


def _f(row, key):
    """Parse a CSV cell to float; '' / non-numeric -> nan."""
    v = row.get(key, "")
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _label(row) -> str:
    base = f"{row.get('policy_kind','?')}/{row.get('task','?')}/{row.get('disruption','?')}/{row.get('obs_mode','?')}"
    return base + ("+filter" if row.get("filter_snapshot") else "")


def _read_rows(csv_paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for p in csv_paths:
        with Path(p).open() as f:
            for r in csv.DictReader(f):
                r["__csv__"] = str(p)
                rows.append(r)
    return rows


def plot_pareto(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in rows:
        x = _f(r, "filter_intervention_rate")
        y = _f(r, "ep_proximity_violation_rate")
        if np.isnan(x):
            x = 0.0  # unfiltered cells sit at zero intervention
        xerr = None
        ylo, yhi = _f(r, "ep_proximity_violation_rate_ci_lo"), _f(r, "ep_proximity_violation_rate_ci_hi")
        yerr = None
        if not (np.isnan(ylo) or np.isnan(yhi)):
            yerr = [[max(0.0, y - ylo)], [max(0.0, yhi - y)]]
        ax.errorbar([x], [y], yerr=yerr, xerr=xerr, fmt="o", capsize=3, label=_label(r))
    ax.set_xlabel("filter intervention rate")
    ax.set_ylabel("ep_proximity_violation_rate")
    ax.set_title("Safety vs filter intervention (Pareto)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_cells_bars(rows, out: Path) -> None:
    labels = [_label(r) for r in rows]
    metrics = [
        ("ep_proximity_violation_rate", "proximity viol."),
        ("ep_ssm_violation_actual_rate", "ssm-actual viol."),
        ("success_rate", "success"),
    ]
    x = np.arange(len(rows))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(rows)), 5))
    for i, (key, name) in enumerate(metrics):
        ax.bar(x + (i - 1) * w, [_f(r, key) for r in rows], width=w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("rate")
    ax.set_title("Per-cell safety + task metrics")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_separation(rows, out: Path) -> None:
    import pandas as pd

    series = []
    for r in rows:
        csv_path = Path(r["__csv__"])
        parquet = csv_path.with_suffix(".raw_episodes.parquet")
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        if "ep_min_separation" in df.columns:
            vals = df["ep_min_separation"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                series.append((_label(r), vals))
    if not series:
        logger.warning("No parquet sidecars with ep_min_separation; skipping separation plot.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, vals in series:
        ax.hist(vals, bins=20, histtype="step", label=name, density=True)
        p99 = float(np.percentile(vals, 1))   # dangerous lower tail
        ax.axvline(p99, ls="--", alpha=0.5)
    ax.set_xlabel("per-episode ep_min_separation (m)")
    ax.set_ylabel("density")
    ax.set_title("Min-separation distribution (dashed = p99 lower tail)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, nargs="+", required=True, help="One or more per-cell CSVs.")
    p.add_argument("--out-dir", type=Path, default=None, help="Where to write PNGs (default: next to first CSV).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    rows = _read_rows(args.csv)
    if not rows:
        raise SystemExit("No rows found in the provided CSV(s).")
    out_dir = args.out_dir or args.csv[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto(rows, out_dir / "pareto.png")
    plot_cells_bars(rows, out_dir / "cells_bars.png")
    plot_separation(rows, out_dir / "separation.png")
    for name in ("pareto.png", "cells_bars.png", "separation.png"):
        fp = out_dir / name
        if fp.exists():
            print(fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
