#!/usr/bin/env python3
"""Filter taxonomy figure: success vs proximity-violation rate for every safety
configuration on saucepan_to_hob (G1 coworker, noisy BodySLAM).

Central result: reactive filtering is freeze-vs-flee bounded — the learned veto (SVF)
sits on the FREEZE horn (no proximity cut); the model-based CBF, whether it dodges the
base or retracts the arm, sits on a FLEE frontier (cuts proximity only by costing task
success); and only the proactive constrained-RL policy reaches a graceful operating point.

Final benchmark numbers (3 seeds x 20 ep); source CSVs in comments. Run with the venv
python (on the Mac use ./venv/bin/python, NOT `source activate`):
    ./venv/bin/python scripts/plot_filter_taxonomy.py --out results/figs/filter_taxonomy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Proactive policy frontier: baseline -> Policy λ=0.1 -> Policy λ=0.27.
#   baseline  results/e4_1/.../row1_baseline.csv ; λ0.1 basin_lam0p1_seed0/s30546 ; λ0.27 fixlam sweep
FRONTIER = [(0.85, 0.296, "Baseline"), (0.75, 0.198, "Policy λ=0.1"), (0.60, 0.170, "Policy λ=0.27")]
SVF = (0.62, 0.265)                                                    # row5_hybrid (learned veto, freeze)
BASE_CBF = [(0.58, 0.170), (0.57, 0.151), (0.57, 0.150),              # hybrid_cbf_d0p30..55
            (0.53, 0.141), (0.43, 0.137), (0.35, 0.148)]
EE_RETRACT = [(0.50, 0.155), (0.45, 0.145), (0.38, 0.108)]            # hybrid_cbf_ee_d0p35..55
BASELINE_PROX = 0.296


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figs/filter_taxonomy.png"))
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.axhline(BASELINE_PROX, ls=":", color="#888", alpha=0.5)

    # proactive frontier (graceful)
    ax.plot([p[0] for p in FRONTIER], [p[1] for p in FRONTIER], "-o", color="#1f77b4",
            lw=2, ms=10, mec="k", mew=0.5, zorder=4, label="proactive policy (graceful)")
    for x, y, name in FRONTIER:
        ax.annotate(name, (x, y), (x + 0.008, y + 0.006), fontsize=8)

    # SVF — freeze horn
    ax.scatter(*SVF, c="#d62728", marker="s", s=150, zorder=4, edgecolors="k",
               linewidths=0.5, label="SVF veto (FREEZE: no cut)")
    ax.annotate("SVF hybrid", (SVF[0], SVF[1]), (SVF[0] + 0.01, SVF[1]), fontsize=8)

    # base-CBF — flee curve
    ax.plot([p[0] for p in BASE_CBF], [p[1] for p in BASE_CBF], "-^", color="#2ca02c",
            lw=1.6, ms=8, mec="k", mew=0.4, zorder=3, label="CBF base-dodge (FLEE)")
    # EE-retract — flinch curve
    ax.plot([p[0] for p in EE_RETRACT], [p[1] for p in EE_RETRACT], "-D", color="#9467bd",
            lw=1.6, ms=8, mec="k", mew=0.4, zorder=3, label="CBF EE-retract (FLINCH)")

    ax.annotate("reactive filters: freeze or flee,\nall below the policy frontier",
                (0.45, 0.13), (0.40, 0.305), fontsize=8.5, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", alpha=0.6))
    ax.set_xlabel("task success rate  →  (higher better)")
    ax.set_ylabel("proximity-violation rate (τ=0.3 m)   ↓ safer")
    ax.set_title("Filter taxonomy (saucepan_to_hob, G1 coworker, noisy):\n"
                 "reactive filtering is freeze-vs-flee bounded — neither base-dodge nor "
                 "EE-retract\nreaches the proactive policy; only anticipation is graceful")
    ax.set_xlim(0.28, 0.94); ax.set_ylim(0.09, 0.33); ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
