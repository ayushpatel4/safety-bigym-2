#!/usr/bin/env python
"""Phase-1 E1.2 — at what noise level does the obs stop helping?

Sweeps `bodyslam.noise_std` across NOISE_STDS for one method/task pair
chosen from the strongest cell of E1.1. Holds OU α, latency, occlusion,
and dropout fixed at their `bodyslam=noisy` defaults — only σ moves.

Usage:
    python scripts/phase1_noise_sweep.py --method dp --task reach_target_single --train
    python scripts/phase1_noise_sweep.py --method dp --task reach_target_single --eval
    python scripts/phase1_noise_sweep.py --smoke
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

NOISE_STDS = (0.02, 0.05, 0.10, 0.15, 0.20)

DISRUPTIONS = (
    "INCIDENTAL", "SHARED_GOAL", "DIRECT", "OBSTRUCTION", "RANDOM_PERTURBED",
)

# Fill in after training. Keyed by (method, task, noise_std).
SNAPSHOTS: dict[tuple[str, str, float], str | None] = {}


def _require_amass():
    if not os.environ.get("AMASS_DATA_DIR"):
        sys.stderr.write("AMASS_DATA_DIR not set.\n")
        sys.exit(1)


def _resolved_snapshot(method: str, task: str, sigma: float) -> Path | None:
    rel = SNAPSHOTS.get((method, task, sigma))
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.is_file() else None


def _train_cmd(method: str, task: str, sigma: float, seed: int) -> list[str]:
    sigma_tag = f"sigma{sigma:.2f}".replace(".", "p")
    run_name = f"phase1-noise-train-{method}-{task}-{sigma_tag}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        "bodyslam=noisy",
        f"++env.bodyslam.noise_std={sigma}",
        f"seed={seed}",
        "save_snapshot=true",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-1","noise-sweep","train","{method}","{task}","{sigma_tag}"]',
    ]


def _eval_cmd(
    method: str, task: str, sigma: float, disruption: str, snapshot: Path,
    *, seed: int, num_eval_episodes: int, wandb_use: bool,
) -> list[str]:
    sigma_tag = f"sigma{sigma:.2f}".replace(".", "p")
    run_name = f"phase1-noise-eval-{method}-{task}-{sigma_tag}-{disruption.lower()}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        "bodyslam=noisy",
        f"++env.bodyslam.noise_std={sigma}",
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
            f'+wandb.tags=["phase-1","noise-sweep","eval","{method}",'
            f'"{task}","{sigma_tag}","{disruption.lower()}"]'
        ),
    ]


def _print_train(method: str, task: str, seed: int) -> int:
    print(f"# Phase-1 E1.2 — noise sweep — {method}/{task} ({len(NOISE_STDS)} runs)")
    for sigma in NOISE_STDS:
        cmd = _train_cmd(method, task, sigma, seed)
        print(" ".join(shlex.quote(c) for c in cmd))
    return 0


def _print_eval(method: str, task: str, seed: int, num_eval_episodes: int) -> int:
    print(
        f"# Phase-1 E1.2 — noise sweep eval — {method}/{task} "
        f"({num_eval_episodes} eps × {len(DISRUPTIONS)} disruptions)"
    )
    missing = []
    for sigma in NOISE_STDS:
        snap = _resolved_snapshot(method, task, sigma)
        if snap is None:
            missing.append(sigma)
            print(f"# SKIP sigma={sigma}: no snapshot.")
            continue
        print(f"# --- sigma={sigma}  ({snap.relative_to(REPO_ROOT)}) ---")
        for disruption in DISRUPTIONS:
            cmd = _eval_cmd(
                method, task, sigma, disruption, snap,
                seed=seed, num_eval_episodes=num_eval_episodes, wandb_use=True,
            )
            print(" ".join(shlex.quote(c) for c in cmd))
        print()
    return 0 if not missing else 2


def _smoke(method: str, task: str, sigma: float, seed: int) -> int:
    cmd = [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        "bodyslam=noisy",
        f"++env.bodyslam.noise_std={sigma}",
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
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    _require_amass()

    if args.smoke:
        return _smoke(args.method, args.task, args.sigma, args.seed)
    if args.eval:
        return _print_eval(args.method, args.task, args.seed, args.num_eval_episodes)
    return _print_train(args.method, args.task, args.seed)


if __name__ == "__main__":
    sys.exit(main())
