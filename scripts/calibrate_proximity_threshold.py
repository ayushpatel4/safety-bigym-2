#!/usr/bin/env python
"""Calibrate the proximity-violation threshold from real rollout data.

Rolls out a policy (or a random policy) under the G1 coworker, collects the per-step
closest-joint ``min_separation`` across all episodes, and plots:

  1. the distribution (histogram + key percentiles) of how close the human gets, and
  2. the **violation-rate vs candidate-threshold** curve — i.e. the empirical CDF,
     since ``P(min_separation < tau)`` is exactly the proximity-violation rate you would
     get if ``proximity_threshold = tau``.

Read the threshold off the knee of curve (2) and sanity-check it against the contact
geometry from ``visualize_separation_distances.py``. Note ``proximity_threshold`` is
speed-independent by design — calibrate it to geometric contact risk, not robot speed.

    # quick local pass (random policy, no snapshot)
    python scripts/calibrate_proximity_threshold.py --episodes 5 --out results/prox_calib.png
    # on your trained policy (representative distribution)
    AMASS_DATA_DIR=<CMU> python scripts/calibrate_proximity_threshold.py \
        --snapshot runs/.../snapshot.pt --obs-mode oracle --episodes 20 --num-demos-for-stats 5

Run with the plain venv Python (no rendering here, so no mjpython needed).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("calibrate_proximity_threshold")

CURRENT_THRESHOLD = 0.5  # SSMConfig.proximity_threshold default
DEFAULT_CANDIDATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)


def collect_separations(
    *, snapshot, task, disruption, obs_mode, human_model, seeds, episodes,
    max_steps, num_demos_for_stats,
) -> np.ndarray:
    from safety_bigym.benchmark.loader import load_policy
    from safety_bigym.benchmark.runners import build_cell_runner

    meta, payload = load_policy(snapshot)
    logger.info("Policy kind=%s; collecting per-step min_separation.", meta.kind)
    runner, _renderable = build_cell_runner(
        meta, payload, snapshot_path=snapshot, task=task, disruption=disruption,
        obs_mode=obs_mode, human_model=human_model, filter_critic=None,
        num_demos_for_stats=num_demos_for_stats,
    )
    seps: List[float] = []
    try:
        for seed in seeds:
            for ep in range(int(episodes)):
                runner.reset(seed * 100_000 + ep)
                for _ in range(int(max_steps)):
                    rec = runner.step()
                    if math.isfinite(rec.min_separation):
                        seps.append(float(rec.min_separation))
                    if rec.terminated or rec.truncated:
                        break
                logger.info("seed=%d ep=%d collected (total steps=%d)", seed, ep, len(seps))
    finally:
        try:
            runner.close()
        except Exception:
            pass
    return np.asarray(seps, dtype=float)


def _plot(seps: np.ndarray, candidates, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = np.linspace(0.0, float(max(1.5, np.percentile(seps, 99))), 400)
    viol = np.array([(seps < t).mean() for t in taus])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (1) distribution
    ax1.hist(seps, bins=60, density=True, color="0.6", edgecolor="white")
    for c in candidates:
        ax1.axvline(c, color="tab:blue", ls=":", alpha=0.5)
    ax1.axvline(CURRENT_THRESHOLD, color="tab:red", ls="--", lw=2, label=f"current {CURRENT_THRESHOLD:.2f} m")
    ax1.set_xlabel("per-step min_separation (m)")
    ax1.set_ylabel("density")
    ax1.set_title(f"How close the human gets ({seps.size} steps)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) violation-rate vs threshold = empirical CDF
    ax2.plot(taus, viol, color="tab:purple", lw=2)
    for c in candidates:
        r = float((seps < c).mean())
        ax2.plot([c], [r], "o", color="tab:blue")
        ax2.annotate(f"{c:.2f}m\n{r*100:.1f}%", (c, r), fontsize=8,
                     textcoords="offset points", xytext=(4, -2))
    rc = float((seps < CURRENT_THRESHOLD).mean())
    ax2.axvline(CURRENT_THRESHOLD, color="tab:red", ls="--", lw=2,
                label=f"current {CURRENT_THRESHOLD:.2f} m -> {rc*100:.1f}%")
    ax2.set_xlabel("candidate proximity threshold tau (m)")
    ax2.set_ylabel("proximity-violation rate  P(min_sep < tau)")
    ax2.set_title("Violation rate vs threshold (read the knee)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, default=None, help="Policy checkpoint (omit for random).")
    p.add_argument("--task", default="saucepan_to_hob")
    p.add_argument("--disruption", default="coworker_train")
    p.add_argument("--obs-mode", choices=("off", "oracle", "noisy"), default="off")
    p.add_argument("--human-model", choices=("g1", "smplh"), default="g1")
    p.add_argument("--seeds", default="0")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--num-demos-for-stats", type=int, default=5)
    p.add_argument("--candidates", type=float, nargs="+", default=list(DEFAULT_CANDIDATES))
    p.add_argument("--out", type=Path, default=Path("results/prox_calib.png"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]

    seps = collect_separations(
        snapshot=args.snapshot, task=args.task, disruption=args.disruption,
        obs_mode=args.obs_mode, human_model=args.human_model, seeds=seeds,
        episodes=args.episodes, max_steps=args.max_steps,
        num_demos_for_stats=args.num_demos_for_stats,
    )
    if seps.size == 0:
        raise SystemExit("No separation samples collected.")

    pcts = {p: float(np.percentile(seps, p)) for p in (1, 5, 10, 25, 50)}
    logger.info("min_separation: min=%.3f  p1=%.3f  p5=%.3f  p10=%.3f  p25=%.3f  median=%.3f  max=%.3f",
                seps.min(), pcts[1], pcts[5], pcts[10], pcts[25], pcts[50], seps.max())
    logger.info("threshold -> proximity-violation rate (fraction of steps below):")
    rows = []
    for c in sorted(set(list(args.candidates) + [CURRENT_THRESHOLD])):
        r = float((seps < c).mean())
        rows.append({"threshold_m": c, "violation_rate": r, "n_steps": int(seps.size)})
        logger.info("  tau=%.2f m -> %.1f%%", c, r * 100)

    _plot(seps, args.candidates, args.out)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold_m", "violation_rate", "n_steps"])
        w.writeheader(); w.writerows(rows)
    logger.info("Wrote %s and %s", args.out, csv_path)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
