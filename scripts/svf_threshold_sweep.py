#!/usr/bin/env python
"""Threshold sweep — Pareto curve for the SVF filter.

Loads a frozen :class:`SafetyCritic` and traces ``intervention_rate`` vs
``residual_violation_rate`` across ``--thresholds`` for one (task, disruption)
cell at a time. Writes a CSV; the GPU operator promotes that to a W&B plot.

``--smoke`` runs 2 thresholds × 1 episode × 50 steps.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    sweep_thresholds,
)

logger = logging.getLogger("svf_threshold_sweep")


DEFAULT_THRESHOLDS = (10.0, 25.0, 50.0, 75.0, 90.0)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--critic-path", type=Path, required=False)
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--fallback", default="zero_velocity")
    p.add_argument("--policy", choices=("random", "snapshot"), default="random")
    p.add_argument(
        "--snapshot-override",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "Override a snapshot path for one task; takes precedence over the "
            "SNAPSHOTS dict in safety_bigym/filters/snapshots.py. Repeatable."
        ),
    )
    p.add_argument(
        "--task", default="reach_target_single", choices=sorted(TASK_REGISTRY)
    )
    p.add_argument("--disruption", default=DEFAULT_DISRUPTIONS[0])
    p.add_argument("--episodes-per-R", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--bodyslam-mode", choices=("oracle", "noisy"), default="oracle")
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _parse_overrides(raw: Sequence[str]) -> dict:
    out: dict = {}
    for entry in raw:
        if "=" not in entry:
            raise SystemExit(f"--snapshot-override expects TASK=PATH; got {entry!r}")
        task, _, path = entry.partition("=")
        if not task or not path:
            raise SystemExit(f"--snapshot-override expects TASK=PATH; got {entry!r}")
        out[task] = path
    return out


def run_sweep(args: argparse.Namespace) -> List[ThresholdEvalResult]:
    from safety_bigym.filters.snapshots import resolve_snapshot

    if args.critic_path is None or not args.critic_path.is_file():
        raise FileNotFoundError(f"--critic-path {args.critic_path!r} not found")

    payload = torch.load(args.critic_path, map_location="cpu", weights_only=False)
    critic = SafetyCritic.from_checkpoint_payload(payload)
    rng = np.random.default_rng(args.seed)

    env = _build_live_env(args.task, args.disruption, args.bodyslam_mode, DEFAULT_CLIPS)
    fallback = FallbackRegistry.build(args.fallback, env.action_space)

    if args.policy == "random":
        policy = random_policy(env, rng)
    elif args.policy == "snapshot":
        overrides = _parse_overrides(getattr(args, "snapshot_override", []) or [])
        snapshot_path = resolve_snapshot(args.task, overrides=overrides)
        if snapshot_path is None:
            raise SystemExit(
                f"Snapshot policy requested but no snapshot configured for task "
                f"{args.task!r}. Set SNAPSHOTS[{args.task!r}] in "
                "safety_bigym/filters/snapshots.py or pass "
                f"--snapshot-override {args.task}=PATH."
            )
        policy = load_snapshot_policy(snapshot_path, env)
    else:
        raise ValueError(f"Unknown policy {args.policy!r}")

    rows = sweep_thresholds(
        env=env,
        critic=critic,
        fallback=fallback,
        thresholds=tuple(args.thresholds),
        policy=policy,
        n_episodes=args.episodes_per_R,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    for r in rows:
        logger.info(
            f"R={r.threshold_R:5.1f} intervention_rate={r.intervention_rate:.3f} "
            f"residual_violation_rate={r.residual_violation_rate:.3f}"
        )
    return rows


def write_csv(rows: List[ThresholdEvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].to_dict().keys()) if rows else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Reduce sweep scope to a 2 R × 1 episode × 50 step smoke run."""
    if args.critic_path is None:
        raise SystemExit(
            "--smoke requires --critic-path to a smoke critic produced by "
            "svf_train_critic.py --smoke"
        )
    args.thresholds = [10.0, 90.0]
    args.episodes_per_R = 1
    args.max_steps = 50
    args.policy = "random"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.smoke:
        apply_smoke_overrides(args)

    rows = run_sweep(args)

    if args.output_csv:
        write_csv(rows, args.output_csv)
        logger.info(f"Wrote {len(rows)} Pareto rows to {args.output_csv}")
    else:
        for r in rows:
            print(r.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
