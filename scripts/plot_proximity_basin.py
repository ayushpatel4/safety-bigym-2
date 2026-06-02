#!/usr/bin/env python3
"""Plot deployment proximity (+success) vs training step for a Lagrangian run.

Reads ``benchmark_policy`` per-cell CSVs, keeps the rows whose snapshot lives in
the target stage dir (``--stage-substr``) and obs mode (``--obs``), parses the
training step from ``snapshot_<step>.pt``, and plots the proximity-violation rate
(with bootstrap CI band) and success rate vs step. A horizontal reference line
marks the unconstrained baseline.

This is the figure behind the d0.3 "avoidance basin" finding: the constraint
induces a mid-training proximity dip that peak-success / final-checkpoint
selection misses. Reusable for the 3-seed CONFIRM (pass several stage dirs'
sweeps and it overlays them).

Usage::

    python scripts/plot_proximity_basin.py \
        --csv-dir results/e4_1/d0p3_window_0006 results/e4_1/row3_converged_2243 \
        --stage-substr d0p3_seed0 --obs noisy \
        --baseline-prox 0.296 --baseline-succ 0.85 \
        --out results/figs/d0p3_basin_seed0.png
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _step_from_snapshot(path: str) -> int | None:
    base = os.path.basename(str(path))
    if not base.startswith("snapshot_") or not base.endswith(".pt"):
        return None
    core = base[len("snapshot_"):-len(".pt")]
    return int(core) if core.isdigit() else None


def collect(csv_dirs, stage_substr: str, obs: str) -> pd.DataFrame:
    rows = []
    for d in csv_dirs:
        for f in glob.glob(os.path.join(d, "*.csv")):
            try:
                r = pd.read_csv(f).iloc[-1]
            except Exception:
                continue
            if stage_substr not in str(r.get("snapshot", "")):
                continue
            if str(r.get("obs_mode", "")) != obs:
                continue
            step = _step_from_snapshot(r["snapshot"])
            if step is None:
                continue
            rows.append({
                "step": step,
                "prox": r["ep_proximity_violation_rate"],
                "prox_lo": r["ep_proximity_violation_rate_ci_lo"],
                "prox_hi": r["ep_proximity_violation_rate_ci_hi"],
                "succ": r["success_rate"],
                "vel": r.get("ep_mean_robot_vel", float("nan")),
            })
    return pd.DataFrame(rows).drop_duplicates("step").sort_values("step").reset_index(drop=True)


def plot(df: pd.DataFrame, baseline_prox, baseline_succ, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # proximity (left axis) with CI band
    ax.fill_between(df.step, df.prox_lo, df.prox_hi, color="#1f77b4", alpha=0.15, label="proximity 95% CI")
    ax.plot(df.step, df.prox, "-o", color="#1f77b4", lw=2, label="proximity (deploy)")
    if baseline_prox is not None:
        ax.axhline(baseline_prox, ls="--", color="#1f77b4", alpha=0.7, label=f"baseline proximity ({baseline_prox:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("proximity-violation rate (τ=0.3 m)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_ylim(0, max(df.prox_hi.max(), (baseline_prox or 0)) * 1.1)

    # success (right axis)
    ax2 = ax.twinx()
    ax2.plot(df.step, df.succ, "-s", color="#d62728", lw=1.3, alpha=0.8, label="success (deploy)")
    if baseline_succ is not None:
        ax2.axhline(baseline_succ, ls=":", color="#d62728", alpha=0.6, label=f"baseline success ({baseline_succ:.2f})")
    ax2.set_ylabel("success rate", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 1.0)

    # merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", framealpha=0.9, ncol=2)
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(df)} checkpoints: steps {df.step.min()}..{df.step.max()})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", nargs="+", required=True)
    ap.add_argument("--stage-substr", default="d0p3_seed0")
    ap.add_argument("--obs", default="noisy")
    ap.add_argument("--baseline-prox", type=float, default=None)
    ap.add_argument("--baseline-succ", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    df = collect(a.csv_dir, a.stage_substr, a.obs)
    if df.empty:
        raise SystemExit(f"no matching CSVs (stage~{a.stage_substr}, obs={a.obs}) under {a.csv_dir}")
    title = a.title or f"d0.3 avoidance basin — {a.stage_substr} ({a.obs})"
    plot(df, a.baseline_prox, a.baseline_succ, a.out, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
