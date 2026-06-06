#!/usr/bin/env python3
"""Filter taxonomy figure: success vs proximity-violation rate for every safety
configuration on saucepan_to_hob (G1 coworker, noisy BodySLAM).

Shows the central result: reactive filtering is freeze-vs-flee bounded — the learned
veto filter (SVF) sits on the FREEZE horn (high intervention, no proximity cut), the
model-based CBF dodge sits on the FLEE horn (cuts proximity but retreats from the task),
and only the proactive constrained-RL policy reaches a graceful operating point.

Points are the final benchmark numbers (3 seeds x 20 ep unless noted). Each carries its
source CSV in a comment so the figure is auditable; edit the constants if numbers change.

Run with the venv python (on the Mac use ./venv/bin/python, NOT `source activate`):
    ./venv/bin/python scripts/plot_filter_taxonomy.py --out results/figs/filter_taxonomy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (success, proximity, colour, marker, label, (label_dx, label_dy)) — source CSVs:
#   Baseline       results/e4_1/e4_1_saucepan_to_hob_noisy_20260605_213348/row1_baseline.csv
#   Policy λ=0.1   results/e4_1/basin_lam0p1_seed0_noisy/s30546.csv  (ROW3 seed-0)
#   Policy λ=0.27  fixlam λ=0.27 sweep (seed-pooled, approx)
#   SVF hybrid     results/e4_1/.../row5_hybrid.csv  (learned veto, zero_velocity)
#   CBF d=0.35/45/55  results/e4_1/hybrid_cbf_d0p{35,45,55}.csv
POINTS = [
    (0.85, 0.296, "#555555", "o", "Baseline",      (-0.04, 0.006)),
    (0.75, 0.198, "#1f77b4", "o", "Policy λ=0.1",   (0.012, 0.004)),
    (0.60, 0.170, "#1f77b4", "o", "Policy λ=0.27",  (0.012, 0.006)),
    (0.62, 0.265, "#d62728", "s", "SVF hybrid",     (0.012, 0.0)),
    (0.57, 0.150, "#2ca02c", "^", "CBF d=0.35",     (0.012, -0.004)),
    (0.43, 0.137, "#2ca02c", "^", "CBF d=0.45",     (0.012, -0.002)),
    (0.35, 0.148, "#2ca02c", "^", "CBF d=0.55",     (-0.005, 0.010)),
]
FRONTIER = [(0.85, 0.296), (0.75, 0.198), (0.60, 0.170)]  # baseline -> policy points
BASELINE_PROX = 0.296


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figs/filter_taxonomy.png"))
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.plot([p[0] for p in FRONTIER], [p[1] for p in FRONTIER], "--", color="#1f77b4",
            alpha=0.5, lw=1.5, zorder=1, label="proactive frontier")
    for x, y, c, m, name, (lx, ly) in POINTS:
        ax.scatter(x, y, c=c, marker=m, s=140, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate(name, (x, y), (x + lx, y + ly), fontsize=8.5)
    ax.axhline(BASELINE_PROX, ls=":", color="#888", alpha=0.5)
    ax.annotate("FREEZE horn\n(no proximity cut, dwell)", (0.62, 0.265), (0.40, 0.272),
                fontsize=8.5, color="#d62728", ha="center",
                arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.6))
    ax.annotate("FLEE horn\n(cuts proximity, but\nretreats from task)", (0.45, 0.142),
                (0.40, 0.105), fontsize=8.5, color="#2ca02c", ha="center")
    ax.annotate("graceful\n(proactive avoidance)", (0.75, 0.198), (0.80, 0.215),
                fontsize=8.5, color="#1f77b4", ha="center")
    ax.set_xlabel("task success rate  →  (higher better)")
    ax.set_ylabel("proximity-violation rate (τ=0.3 m)   ↓ safer")
    ax.set_title("Filter taxonomy on saucepan_to_hob (G1 coworker, noisy):\n"
                 "reactive filtering is freeze-vs-flee bounded; only the proactive policy is graceful")
    ax.set_xlim(0.28, 0.94); ax.set_ylim(0.095, 0.32); ax.grid(alpha=0.25)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
