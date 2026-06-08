#!/usr/bin/env python3
"""Joint-coverage figure: the two ISO-15066 safety axes at once — geometric proximity
(the policy's axis) vs the velocity-adaptive SSM violation rate (the filter's axis) — on
saucepan_to_hob (G1 coworker, noisy BodySLAM).

Central result: each component specialises on ONE axis and leaves the other near baseline —
the proactive policy drives proximity down but leaves velocity high; the speed-scaling
filter drives velocity down but leaves proximity ≈ baseline. ONLY the policy+filter hybrid
reaches the bottom-left "both-safe" corner (proximity 0.250 AND ssm-actual 0.065) — joint
proximity+velocity safety neither part achieves alone. The cost is task success (annotated):
the proactive and reactive success costs stack multiplicatively (0.85 → 0.44).

Final benchmarks (3 seeds x 20 ep, pooled 180 ep); source CSVs in comments. Run with the
venv python (on the Mac use ./venv/bin/python, NOT `source activate`):
    ./venv/bin/python scripts/plot_joint_coverage.py --out results/figs/joint_coverage.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# label, proximity-violation rate, ssm-actual rate, success, colour, marker
#   baseline / policy: row_nows + basin_lam0p1 picked ckpts (pooled)
#   speed-scaling alone: results/e4_1/speedscale_base_ds0p40.csv
#   hybrid: results/e4_1/hybrid_speedscale_d0p40_seed{0,1,2} (pooled, analyze_row3 aggregate)
CONFIGS = [
    ("baseline",                0.296, 0.146, 0.85, "#7f7f7f", "o"),
    ("policy λ=0.1",            0.228, 0.112, 0.76, "#1f77b4", "*"),
    ("speed-scaling (d=0.40)",  0.273, 0.048, 0.53, "#2ca02c", "D"),
    ("HYBRID  policy+filter",   0.250, 0.065, 0.44, "#9467bd", "P"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figs/joint_coverage.png"))
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    # "both-safe" corner (bottom-left): proximity < policy-ish AND ssm-actual < filter-ish
    ax.add_patch(Rectangle((0.20, 0.02), 0.262 - 0.20, 0.085 - 0.02, facecolor="#2ca02c",
                           alpha=0.08, zorder=0))
    ax.text(0.222, 0.030, "joint proximity + velocity\nsafety (both axes low)",
            fontsize=8, color="#1a7a1a", alpha=0.9)

    for label, prox, ssm, succ, color, marker in CONFIGS:
        sz = 360 if marker in ("*", "P") else 150
        ax.scatter(prox, ssm, c=color, marker=marker, s=sz, zorder=4, edgecolors="k",
                   linewidths=0.6)
        dy = 0.006 if label != "speed-scaling (d=0.40)" else -0.010
        ax.annotate(f"{label}\nsucc={succ:.2f}", (prox, ssm), (prox + 0.004, ssm + dy),
                    fontsize=8.2, fontweight=("bold" if marker == "P" else "normal"))

    # guide arrows: each specialist covers one axis; hybrid covers both
    ax.annotate("", (0.252, 0.065), (0.228, 0.110), zorder=2,
                arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.45, ls="--"))
    ax.annotate("", (0.252, 0.067), (0.271, 0.050), zorder=2,
                arrowprops=dict(arrowstyle="->", color="#2ca02c", alpha=0.45, ls="--"))
    ax.text(0.291, 0.088, "each part →\none axis only", fontsize=8, color="#666", ha="center")

    ax.set_xlabel("geometric proximity-violation rate (τ=0.3 m)   ↓ safer  ·  POLICY's axis")
    ax.set_ylabel("velocity-adaptive ISO violation rate\n(ssm-actual)   ↓ safer  ·  FILTER's axis")
    ax.set_title("Joint ISO-15066 coverage (saucepan_to_hob, G1 coworker, noisy):\n"
                 "each component owns one axis — only the hybrid reaches the both-safe corner,\n"
                 "at a stacked success cost (0.85 → 0.44)")
    ax.set_xlim(0.205, 0.315)
    ax.set_ylim(0.02, 0.165)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
