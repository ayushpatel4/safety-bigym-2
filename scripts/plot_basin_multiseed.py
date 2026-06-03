#!/usr/bin/env python3
"""Overlay deployment proximity (+success) vs training step for several seeds.

One colored series per ``--sweep-dir`` (a ``run_basin_sweep.sh`` output dir of
``s<step>.csv`` benchmark files). Top panel = proximity-violation rate vs the
unconstrained baseline; bottom panel = success rate (so a "low proximity by
task-collapse" seed is visible as crashing success, not safety). Use it on the
fixed-lambda run to show whether the graceful avoidance basin reproduces across
seeds (the headline robustness figure), and it reproduces the d=0.3 PID-regime
figure too.

Usage::

    python scripts/plot_basin_multiseed.py \
        --sweep-dir results/e4_1/basin_lam0p27_seed0_noisy \
                    results/e4_1/basin_lam0p27_seed1_noisy \
                    results/e4_1/basin_lam0p27_seed2_noisy \
        --baseline-prox 0.296 --baseline-succ 0.85 --ci \
        --out results/figs/fixlam_basin_3seed.png \
        --title "fixed-lambda=0.27: proximity basin across 3 seeds"
    # optional explicit labels (e.g. to annotate lambda per seed):
    #   --labels "seed0 (lam=0)" "seed1 (lam=0.27)" "seed2 (lam=3.86)"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _step(path: str) -> int | None:
    m = re.match(r"s(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_sweep(d: str):
    """-> sorted list of (step, prox, prox_lo, prox_hi, success) for one seed dir."""
    rows = []
    for f in glob.glob(os.path.join(d, "s*.csv")):
        st = _step(f)
        if st is None:
            continue
        try:
            r = pd.read_csv(f).iloc[-1]
        except Exception:
            continue
        rows.append((st, r.ep_proximity_violation_rate,
                     r.ep_proximity_violation_rate_ci_lo,
                     r.ep_proximity_violation_rate_ci_hi, r.success_rate))
    return sorted(rows)


def _label_for(d: str) -> str:
    m = re.search(r"seed(\d+)", os.path.basename(os.path.normpath(d)))
    return f"seed{m.group(1)}" if m else os.path.basename(os.path.normpath(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None, help="one per --sweep-dir (default: seedN from dir name)")
    ap.add_argument("--baseline-prox", type=float, default=None)
    ap.add_argument("--baseline-succ", type=float, default=None)
    ap.add_argument("--ci", action="store_true", help="shade each seed's proximity 95% CI band")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="proximity basin across seeds")
    a = ap.parse_args()

    labels = a.labels or [_label_for(d) for d in a.sweep_dir]
    if len(labels) != len(a.sweep_dir):
        raise SystemExit(f"--labels ({len(labels)}) must match --sweep-dir count ({len(a.sweep_dir)})")

    colors = plt.cm.tab10.colors
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)
    pmax, any_data = 0.0, False
    for i, (d, lab) in enumerate(zip(a.sweep_dir, labels)):
        rows = load_sweep(d)
        if not rows:
            print(f"WARN: no s*.csv in {d} (skipping)")
            continue
        any_data = True
        c = colors[i % len(colors)]
        st = [r[0] for r in rows]; pr = [r[1] for r in rows]; su = [r[4] for r in rows]
        ax.plot(st, pr, "-o", color=c, lw=1.8, ms=4, label=lab)
        if a.ci:
            ax.fill_between(st, [r[2] for r in rows], [r[3] for r in rows], color=c, alpha=0.12)
            pmax = max(pmax, max(r[3] for r in rows))
        else:
            pmax = max(pmax, max(pr))
        ax2.plot(st, su, "-o", color=c, lw=1.8, ms=4, label=lab)
    if not any_data:
        raise SystemExit("no data found in any --sweep-dir")

    if a.baseline_prox is not None:
        ax.axhline(a.baseline_prox, ls="--", color="gray", label=f"baseline {a.baseline_prox:.3f}")
        pmax = max(pmax, a.baseline_prox)
    if a.baseline_succ is not None:
        ax2.axhline(a.baseline_succ, ls=":", color="gray", label=f"baseline {a.baseline_succ:.2f}")
    ax.set_ylabel("deploy proximity (τ=0.3 m)"); ax.set_ylim(0, pmax * 1.1)
    ax.legend(fontsize=8, ncol=2); ax.set_title(a.title)
    ax2.set_ylabel("success rate"); ax2.set_xlabel("training step"); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}  ({sum(1 for d in a.sweep_dir if load_sweep(d))} seeds plotted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
