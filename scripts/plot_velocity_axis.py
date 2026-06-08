#!/usr/bin/env python3
"""Velocity-axis companion to the filter-taxonomy figure: success vs the
velocity-adaptive ISO-15066 violation rate (`ep_ssm_violation_actual_rate`) on
saucepan_to_hob (G1 coworker, noisy BodySLAM).

Where `plot_filter_taxonomy.py` judges every config on the *proximity* axis (the
policy's axis, where reactive filters fail), this figure judges them on the *velocity*
axis (the filter's proper ISO-SSM axis). Central result: the graded speed-scaling filter
is the ONLY configuration that drives ssm-actual far below baseline (0.146 -> 0.048,
-67%, optimum d_slow=0.40) — the SVF *veto* leaves it untouched (binary stop-or-go is not
the graded velocity margin ISO-SSM prescribes), and the proactive policy helps only
partially (~0.11, via avoidance). So the filter, on its proper axis, genuinely works —
the division of labour: policy -> proximity, speed-scaling -> velocity.

Numbers are the final benchmarks (3 seeds x 20 ep); source CSVs in comments. Run with the
venv python (on the Mac use ./venv/bin/python, NOT `source activate`):
    ./venv/bin/python scripts/plot_velocity_axis.py --out results/figs/velocity_axis.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (success, ssm_violation_actual_rate)
BASELINE = (0.85, 0.146)        # unconstrained baseline (speedscale_base reference row)
SVF = (0.78, 0.148)             # SVF veto (row4) — slows the robot but no velocity-margin help
# fixed-λ=0.1 policy, pooled over 3 picked basin checkpoints
#   seed0 s30546 ssmA 0.088 / seed1 s8225 0.153 / seed2 s32696 0.096  -> mean 0.112
POLICY = (0.76, 0.112)
# graded speed-scaling sweep on the baseline, results/e4_1/speedscale_base_ds*.csv
#   (success, ssm-actual, d_slow)
SPEEDSCALE = [
    (0.57, 0.078, 0.25),
    (0.55, 0.077, 0.30),
    (0.50, 0.067, 0.35),
    (0.53, 0.048, 0.40),   # optimum (U-shape minimum)
    (0.42, 0.059, 0.50),
    (0.35, 0.061, 0.60),
]
OPT_DSLOW = 0.40
# policy+speed-scaling hybrid (d=0.40, pooled): the filter ADDS velocity reduction on top of
# the policy (ssm-actual 0.112 -> 0.065) while keeping proximity ~0.25; (success, ssm-actual)
HYBRID = (0.44, 0.065)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figs/velocity_axis.png"))
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.axhline(BASELINE[1], ls=":", color="#888", alpha=0.6)
    ax.text(0.305, BASELINE[1] + 0.002, "baseline ssm-actual", fontsize=7.5, color="#777")

    # baseline + SVF veto (the failed filter) — both sit at the baseline velocity level
    ax.scatter(*BASELINE, c="#7f7f7f", marker="o", s=130, zorder=4, edgecolors="k",
               linewidths=0.5, label="baseline (no safety)")
    ax.annotate("baseline", BASELINE, (BASELINE[0] - 0.005, BASELINE[1] + 0.004), fontsize=8)
    ax.scatter(*SVF, c="#d62728", marker="s", s=150, zorder=4, edgecolors="k",
               linewidths=0.5, label="SVF veto (no velocity help)")
    ax.annotate("SVF veto", SVF, (SVF[0] - 0.055, SVF[1] - 0.004), fontsize=8)

    # proactive policy — helps the velocity axis partially, via avoidance
    ax.scatter(*POLICY, c="#1f77b4", marker="*", s=320, zorder=5, edgecolors="k",
               linewidths=0.6, label="proactive policy λ=0.1 (helps via avoidance)")
    ax.annotate("policy λ=0.1", POLICY, (POLICY[0] + 0.008, POLICY[1] + 0.003), fontsize=8)

    # speed-scaling sweep — the velocity specialist
    xs = [p[0] for p in SPEEDSCALE]
    ys = [p[1] for p in SPEEDSCALE]
    ax.plot(xs, ys, "-D", color="#2ca02c", lw=1.6, ms=8, mec="k", mew=0.4, zorder=3,
            label="speed-scaling filter (velocity axis)")
    for sx, sy, d in SPEEDSCALE:
        tag = f"d={d:.2f}" + ("  ◀ optimum" if abs(d - OPT_DSLOW) < 1e-6 else "")
        ax.annotate(tag, (sx, sy), (sx + 0.008, sy - 0.004), fontsize=7.2,
                    color=("#1a7a1a" if abs(d - OPT_DSLOW) < 1e-6 else "#555"),
                    fontweight=("bold" if abs(d - OPT_DSLOW) < 1e-6 else "normal"))

    ax.annotate("only the filter, on its proper axis,\ndrives ssm-actual to the floor (−67%)",
                (0.50, 0.048), (0.58, 0.030), fontsize=8.5, color="#1a7a1a", ha="center",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", alpha=0.7))

    # policy+filter hybrid — adds velocity reduction on top of the policy
    ax.scatter(*HYBRID, c="#9467bd", marker="P", s=240, zorder=5, edgecolors="k",
               linewidths=0.6, label="policy+filter hybrid")
    ax.annotate("hybrid", HYBRID, (HYBRID[0] + 0.008, HYBRID[1] + 0.003), fontsize=8,
                fontweight="bold")
    ax.annotate("", HYBRID, POLICY, zorder=2,
                arrowprops=dict(arrowstyle="->", color="#9467bd", alpha=0.5, ls="--"))

    ax.set_xlabel("task success rate  →  (higher better)")
    ax.set_ylabel("velocity-adaptive ISO violation rate\n$ep\\_ssm\\_violation\\_actual\\_rate$   ↓ safer")
    ax.set_title("Velocity axis (saucepan_to_hob, G1 coworker, noisy):\n"
                 "on the filter's PROPER ISO-SSM axis, graded speed-scaling works —\n"
                 "the veto does not, and the policy only partially (division of labour)")
    ax.set_xlim(0.30, 0.92)
    ax.set_ylim(0.02, 0.165)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
