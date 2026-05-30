#!/usr/bin/env python
"""SVF safety-critic training entrypoint (Phase 2 sub-branch 2).

Reads a sharded dataset produced by ``svf_collect_dataset.py`` and trains
a :class:`SafetyCritic` with offline CQL. Saves a checkpoint payload that
round-trips through :meth:`SafetyCritic.from_checkpoint_payload`.

``--smoke`` mode runs ~100 grad steps on the smoke shard, asserts that the
running mean loss decreases between the first and last quartile, and writes
``_smoke_critic.pt``. CPU-runnable.

Real GPU runs use the full ``cfgs/launch/svf_filter_train.yaml`` Hydra config
(promoted in a follow-up commit; this script is argparse-driven for v1).

Usage:
    python scripts/svf_train_critic.py --smoke --dataset-dir /tmp/svf_smoke
    python scripts/svf_train_critic.py --dataset-dir datasets/svf_v1 \\
        --output checkpoints/svf_v1.pt --num-steps 200000 --cql-alpha 5.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_bigym.filters.cql_trainer import CQLSafetyTrainer  # noqa: E402
from safety_bigym.filters.critic import SafetyCritic  # noqa: E402
from safety_bigym.filters.dataset import (  # noqa: E402
    SafetyTransitionDataset,
    make_oversampler,
)

logger = logging.getLogger("svf_train_critic")


@dataclass
class TrainPlan:
    dataset_dir: Path
    output: Path
    num_steps: int
    batch_size: int
    cql_alpha: float
    aux_weight: float
    target_tau: float
    lr: float
    gamma: float
    target_violation_rate: float
    log_every: int
    seed: int
    device: str
    # When set, relabel r_safe on the fly as (min_separation >= τ) from the
    # stored per-step min_separation — free re-threshold, no re-collection.
    proximity_threshold: Optional[float] = None

    @classmethod
    def smoke(cls, dataset_dir: Path, output: Path) -> "TrainPlan":
        return cls(
            dataset_dir=dataset_dir,
            output=output,
            num_steps=100,
            batch_size=32,
            cql_alpha=5.0,
            aux_weight=0.0,
            target_tau=5e-3,
            lr=3e-4,
            gamma=0.99,
            target_violation_rate=0.3,
            log_every=10,
            seed=0,
            device="cpu",
        )


def _build_action_space(dataset: SafetyTransitionDataset):
    """Reconstruct an action-space Box from the dataset.

    The collector recorded raw env actions (not post-tanh-rescaled). For the
    critic we only need ``low``/``high`` to seed CQL OOD sampling, so we
    extract them from the dataset itself: per-dim min/max across all stored
    actions, with a small margin so the box strictly contains the data.
    """
    import gymnasium as gym

    actions = []
    for shard in dataset._shards:  # noqa: SLF001 — read-only access
        with np.load(shard.path) as data:
            actions.append(np.asarray(data["action"]))
    actions = np.concatenate(actions, axis=0)
    low = actions.min(axis=0)
    high = actions.max(axis=0)
    span = (high - low).clip(min=1e-3)
    margin = 0.05 * span
    return gym.spaces.Box(
        low=(low - margin).astype(np.float32),
        high=(high + margin).astype(np.float32),
        dtype=np.float32,
    )


def run_training(plan: TrainPlan) -> Path:
    # ``force=True`` rebinds the root handler/level even if some earlier
    # import already configured one (in which case bare basicConfig is a
    # no-op and every logger.info silently dropped at root level=WARNING).
    # Without this, B5.1 smoke could complete successfully but produce
    # zero stdout/stderr, making it look like the script never ran.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    logger.info(f"Loading dataset from {plan.dataset_dir}")
    dataset = SafetyTransitionDataset(
        plan.dataset_dir, proximity_threshold=plan.proximity_threshold
    )
    if plan.proximity_threshold is not None:
        logger.info(
            f"Relabelling r_safe on the fly at proximity_threshold="
            f"{plan.proximity_threshold} m (from stored min_separation)."
        )
    logger.info(
        f"Dataset: {len(dataset)} transitions; "
        f"{len(dataset.violation_indices)} violating, "
        f"{len(dataset.safe_indices)} safe"
    )

    sampler = make_oversampler(
        dataset, target_violation_rate=plan.target_violation_rate, seed=plan.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=plan.batch_size,
        sampler=sampler,
        num_workers=0,  # mmap-backed shards; multi-worker is GPU-only
        drop_last=True,
    )

    action_space = _build_action_space(dataset)
    critic = SafetyCritic(spec=dataset.spec, gamma=plan.gamma)
    trainer = CQLSafetyTrainer(
        critic=critic,
        action_space=action_space,
        cql_alpha=plan.cql_alpha,
        aux_weight=plan.aux_weight,
        lr=plan.lr,
        target_tau=plan.target_tau,
        device=plan.device,
        seed=plan.seed,
    )

    losses: List[float] = []
    bellman_losses: List[float] = []
    step = 0
    iter_loader = iter(loader)
    while step < plan.num_steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
        info = trainer.train_step(batch)
        losses.append(info["loss"])
        bellman_losses.append(info["bellman_loss"])
        step += 1
        if step % plan.log_every == 0 or step == 1:
            logger.info(
                f"step={step:6d} loss={info['loss']:.4f} "
                f"bellman={info['bellman_loss']:.4f} cql={info['cql_term']:.4f} "
                f"q_mean={info['q_mean']:.3f}"
            )

    plan.output.parent.mkdir(parents=True, exist_ok=True)
    payload = critic.checkpoint_payload()
    q1 = max(1, len(losses) // 4)
    payload["training"] = {
        "num_steps": plan.num_steps,
        "cql_alpha": plan.cql_alpha,
        "lr": plan.lr,
        "target_tau": plan.target_tau,
        "loss_first": float(np.mean(losses[:q1])),
        "loss_last": float(np.mean(losses[-q1:])),
        # Bellman MSE is the load-bearing fit signal — total loss can grow
        # with CQL because the conservatism term reflects how distinguishable
        # Q is on data vs OOD actions, which is a learning signal not a fit.
        "bellman_first": float(np.mean(bellman_losses[:q1])),
        "bellman_last": float(np.mean(bellman_losses[-q1:])),
        "dataset_dir": str(plan.dataset_dir),
        "dataset_size": len(dataset),
        "proximity_threshold": plan.proximity_threshold,
    }
    torch.save(payload, plan.output)
    logger.info(f"Wrote checkpoint to {plan.output}")
    return plan.output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dataset-dir", type=Path, required=False)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--num-steps", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cql-alpha", type=float, default=5.0)
    p.add_argument("--aux-weight", type=float, default=0.0)
    p.add_argument("--target-tau", type=float, default=5e-3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-violation-rate", type=float, default=0.3)
    p.add_argument(
        "--proximity-threshold",
        type=float,
        default=None,
        help=(
            "Relabel r_safe on the fly as (min_separation >= τ) metres, from "
            "the stored per-step min_separation — free re-threshold, no "
            "re-collection. Unset uses the label baked at collection time."
        ),
    )
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.smoke:
        if args.dataset_dir is None:
            raise SystemExit(
                "--smoke requires --dataset-dir pointing at the smoke shard "
                "produced by `svf_collect_dataset.py --smoke`."
            )
        output = args.output or (args.dataset_dir / "_smoke_critic.pt")
        plan = TrainPlan.smoke(dataset_dir=args.dataset_dir, output=output)
    else:
        if args.dataset_dir is None or args.output is None:
            raise SystemExit("--dataset-dir and --output are required for non-smoke runs.")
        plan = TrainPlan(
            dataset_dir=args.dataset_dir,
            output=args.output,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            cql_alpha=args.cql_alpha,
            aux_weight=args.aux_weight,
            target_tau=args.target_tau,
            lr=args.lr,
            gamma=args.gamma,
            target_violation_rate=args.target_violation_rate,
            log_every=args.log_every,
            seed=args.seed,
            device=args.device,
            proximity_threshold=args.proximity_threshold,
        )

    out = run_training(plan)

    if args.smoke:
        # Smoke assertion: Bellman MSE must decrease — that's the only term
        # whose downward trend implies the critic is fitting r_safe correctly.
        payload = torch.load(out, weights_only=False)
        first = payload["training"]["bellman_first"]
        last = payload["training"]["bellman_last"]
        if not last < first:
            raise SystemExit(
                f"Smoke training did not reduce Bellman MSE: "
                f"first={first:.4f} last={last:.4f}"
            )
        logger.info(
            f"Smoke OK: bellman MSE reduced {first:.4f} -> {last:.4f} "
            f"({100 * (1 - last / max(first, 1e-9)):.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
