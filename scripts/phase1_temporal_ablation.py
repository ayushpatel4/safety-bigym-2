#!/usr/bin/env python
"""Phase-1 E1.3 — does the noise *structure* matter, or just the marginal σ?

Compares three temporal noise structures at matched marginal σ=0.05:
  iid     — α=0   (degenerate OU = i.i.d.), no latency, no dropout, no occlusion
  ou      — α=0.9 (correlated OU),         no latency, no dropout, no occlusion
  full    — α=0.9, latency=3, dropout=0.02, occlusion if available

If `iid` matches the policy's safety violation rate of `ou`/`full`, the
policy is using the perception channel as a noisy point estimate only and
not benefiting from the temporal structure of real perception. That's the
upper bound on how much value Phase 2 (an explicit filter) can add.

Usage:
    python scripts/phase1_temporal_ablation.py --method dp --task reach_target_single --train
    python scripts/phase1_temporal_ablation.py --method dp --task reach_target_single --eval
    python scripts/phase1_temporal_ablation.py --smoke
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADLESS_ENV = ("MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl")

METHODS = {
    "dp":  {"launch": "dp_pixel_safety_bigym",  "exp_dir": "dp_safety"},
    "act": {"launch": "act_pixel_safety_bigym", "exp_dir": "act_safety"},
}

# Variant -> Hydra overrides (applied on top of `bodyslam=noisy`).
VARIANTS: dict[str, list[str]] = {
    "iid": [
        "++env.bodyslam.ou_alpha=0.0",
        "++env.bodyslam.latency_steps=0",
        "++env.bodyslam.dropout_prob=0.0",
        "++env.bodyslam.use_occlusion=false",
    ],
    "ou": [
        "++env.bodyslam.ou_alpha=0.9",
        "++env.bodyslam.latency_steps=0",
        "++env.bodyslam.dropout_prob=0.0",
        "++env.bodyslam.use_occlusion=false",
    ],
    "full": [
        "++env.bodyslam.ou_alpha=0.9",
        "++env.bodyslam.latency_steps=3",
        "++env.bodyslam.dropout_prob=0.02",
    ],
}

DISRUPTIONS = (
    "INCIDENTAL", "SHARED_GOAL", "DIRECT", "OBSTRUCTION", "RANDOM_PERTURBED",
)

# Fill in after training. Keyed by (method, task, variant).
SNAPSHOTS: dict[tuple[str, str, str], str | None] = {}


def _require_amass():
    if not os.environ.get("AMASS_DATA_DIR"):
        sys.stderr.write("AMASS_DATA_DIR not set.\n")
        sys.exit(1)


def _resolved_snapshot(method: str, task: str, variant: str) -> Path | None:
    rel = SNAPSHOTS.get((method, task, variant))
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.is_file() else None


def _train_cmd(method: str, task: str, variant: str, seed: int) -> list[str]:
    run_name = f"phase1-temporal-train-{method}-{task}-{variant}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        "bodyslam=noisy",
        *VARIANTS[variant],
        f"seed={seed}",
        "save_snapshot=true",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-1","temporal-ablation","train","{method}","{task}","{variant}"]',
    ]


def _eval_cmd(
    method: str, task: str, variant: str, disruption: str, snapshot: Path,
    *, seed: int, num_eval_episodes: int, wandb_use: bool,
) -> list[str]:
    run_name = f"phase1-temporal-eval-{method}-{task}-{variant}-{disruption.lower()}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        "bodyslam=noisy",
        *VARIANTS[variant],
        f"+env.disruption_type={disruption}",
        f"+snapshot_path={snapshot}",
        "num_train_frames=0",
        "num_pretrain_steps=0",
        "demos=0",
        f"num_eval_episodes={num_eval_episodes}",
        "eval_every_steps=1",
        f"seed={seed}",
        f"wandb.use={'true' if wandb_use else 'false'}",
        f"wandb.name={run_name}",
        (
            f'+wandb.tags=["phase-1","temporal-ablation","eval","{method}",'
            f'"{task}","{variant}","{disruption.lower()}"]'
        ),
    ]


def _print_train(method: str, task: str, seed: int) -> int:
    print(f"# Phase-1 E1.3 — temporal ablation — {method}/{task} ({len(VARIANTS)} runs)")
    for variant in VARIANTS:
        cmd = _train_cmd(method, task, variant, seed)
        print(" ".join(shlex.quote(c) for c in cmd))
    return 0


def _print_eval(method: str, task: str, seed: int, num_eval_episodes: int) -> int:
    print(f"# Phase-1 E1.3 — temporal ablation eval — {method}/{task}")
    missing = []
    for variant in VARIANTS:
        snap = _resolved_snapshot(method, task, variant)
        if snap is None:
            missing.append(variant)
            print(f"# SKIP variant={variant}: no snapshot.")
            continue
        print(f"# --- variant={variant}  ({snap.relative_to(REPO_ROOT)}) ---")
        for disruption in DISRUPTIONS:
            cmd = _eval_cmd(
                method, task, variant, disruption, snap,
                seed=seed, num_eval_episodes=num_eval_episodes, wandb_use=True,
            )
            print(" ".join(shlex.quote(c) for c in cmd))
        print()
    return 0 if not missing else 2


def _smoke(method: str, task: str, variant: str, seed: int) -> int:
    cmd = [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        "bodyslam=noisy",
        *VARIANTS[variant],
        f"seed={seed}",
        "num_train_frames=100",
        "num_pretrain_steps=0",
        "demos=0",
        "num_eval_episodes=1",
        "wandb.use=false",
    ]
    print(">>> smoke:", " ".join(shlex.quote(c) for c in cmd))
    argv = list(cmd)
    env = os.environ.copy()
    while argv and "=" in argv[0] and not argv[0].startswith("-"):
        k, v = argv.pop(0).split("=", 1)
        env[k] = v
    return subprocess.run(argv, cwd=REPO_ROOT, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--eval", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", default="dp", choices=sorted(METHODS))
    parser.add_argument("--task", default="reach_target_single")
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    _require_amass()

    if args.smoke:
        return _smoke(args.method, args.task, args.variant, args.seed)
    if args.eval:
        return _print_eval(args.method, args.task, args.seed, args.num_eval_episodes)
    return _print_train(args.method, args.task, args.seed)


if __name__ == "__main__":
    sys.exit(main())
