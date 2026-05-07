#!/usr/bin/env python
"""Evaluate the SVF safety filter on top of a target policy.

Phase 2 deliverable — measures intervention rate + residual violation rate
for a frozen :class:`SafetyCritic` wrapped around a target policy in a live
:class:`SafetyBiGymEnv` + :class:`BodySLAMWrapper`.

``--policy random`` works without any snapshot. ``--policy snapshot``
requires ``--snapshot-path`` and follows the same workspace.py-drift loading
path as ``svf_collect_dataset.py`` (Phase-0 ACT snapshots are still pending
GPU retrain; until then this branch fails fast).

``--smoke`` runs 1 episode × 50 steps (random policy) and writes one row of
metrics to stdout / CSV.

Usage:
    python scripts/svf_eval_filter.py --smoke --critic-path /tmp/svf_smoke/_smoke_critic.pt
    python scripts/svf_eval_filter.py \\
        --critic-path checkpoints/svf_v1.pt --threshold-R 50.0 \\
        --policy random --tasks reach_target_single --episodes-per-cell 10
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse env-builder + snapshot policy loader from the collector to keep the
# wrapping order identical between collection-time and eval-time.
from svf_collect_dataset import (  # noqa: E402
    DEFAULT_CLIPS,
    DEFAULT_DISRUPTIONS,
    TASK_REGISTRY,
    _build_live_env,
    load_snapshot_policy,
    random_policy,
)
from safety_bigym.filters.critic import SafetyCritic  # noqa: E402
from safety_bigym.filters.fallback import FallbackRegistry  # noqa: E402
from safety_bigym.filters.threshold_sweep import (  # noqa: E402
    ThresholdEvalResult,
    evaluate_threshold,
)

logger = logging.getLogger("svf_eval_filter")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--critic-path", type=Path, required=False)
    p.add_argument("--threshold-R", type=float, default=50.0)
    p.add_argument("--fallback", default="zero_velocity")
    p.add_argument(
        "--policy", choices=("random", "snapshot"), default="random"
    )
    p.add_argument("--snapshot-path", type=Path, default=None)
    p.add_argument(
        "--tasks", nargs="+", default=("reach_target_single",),
        choices=sorted(TASK_REGISTRY),
    )
    p.add_argument("--disruptions", nargs="+", default=list(DEFAULT_DISRUPTIONS))
    p.add_argument("--episodes-per-cell", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--bodyslam-mode", choices=("oracle", "noisy"), default="oracle")
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _load_critic(path: Path) -> SafetyCritic:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"--critic-path {path!r} not found")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SafetyCritic.from_checkpoint_payload(payload)


def _make_policy(name: str, env, snapshot_path: Optional[Path], rng: np.random.Generator):
    if name == "random":
        return random_policy(env, rng)
    if name == "snapshot":
        return load_snapshot_policy(snapshot_path, env)
    raise ValueError(f"Unknown policy {name!r}")


def run_eval(args: argparse.Namespace) -> list[ThresholdEvalResult]:
    rng = np.random.default_rng(args.seed)
    critic = _load_critic(args.critic_path)

    rows: list[ThresholdEvalResult] = []
    for task_key in args.tasks:
        for disruption in args.disruptions:
            logger.info(
                f"Evaluating filter on task={task_key} disruption={disruption} "
                f"(R={args.threshold_R}, policy={args.policy})"
            )
            env = _build_live_env(
                task_key, disruption, args.bodyslam_mode, DEFAULT_CLIPS
            )
            fallback = FallbackRegistry.build(args.fallback, env.action_space)
            policy = _make_policy(args.policy, env, args.snapshot_path, rng)

            result = evaluate_threshold(
                env=env,
                critic=critic,
                fallback=fallback,
                threshold_R=args.threshold_R,
                policy=policy,
                n_episodes=args.episodes_per_cell,
                max_steps=args.max_steps,
                seed=args.seed,
            )
            logger.info(
                f"  intervention_rate={result.intervention_rate:.3f} "
                f"residual_violation_rate={result.residual_violation_rate:.3f} "
                f"mean_q={result.mean_q_value:.2f}"
            )
            rows.append(result)
    return rows


def write_csv(rows: list[ThresholdEvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].to_dict().keys()) if rows else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.smoke:
        if args.critic_path is None:
            raise SystemExit(
                "--smoke requires --critic-path to a smoke critic produced by "
                "svf_train_critic.py --smoke"
            )
        # Smoke override: 1 episode × 50 steps × first task × first disruption
        args.tasks = (args.tasks[0],)
        args.disruptions = (args.disruptions[0],)
        args.episodes_per_cell = 1
        args.max_steps = 50
        args.policy = "random"

    rows = run_eval(args)

    if args.output_csv:
        write_csv(rows, args.output_csv)
        logger.info(f"Wrote {len(rows)} rows to {args.output_csv}")
    else:
        for r in rows:
            print(r.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
