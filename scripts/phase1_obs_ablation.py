#!/usr/bin/env python
"""Phase-1 E1.1 — does giving the policy a human-pose obs help safety?

Sweeps `bodyslam=` ∈ {off, oracle, noisy} for each method ∈ {dp, act}
across PHASE1_TASKS. The 'off' arm is the no-perception baseline; 'oracle'
is the upper bound (clean human_pos, all flags zero); 'noisy' runs the full
OU + latency + dropout pipeline.

Eval reuses the baseline_sweep pattern: train, then evaluate each trained
policy across all 5 ISO 15066 disruption types. The bodyslam preset must
match between train and eval (the policy was trained with that obs key,
so eval needs it too).

Usage:
    # Print train commands (18 cells = 3 tasks × 2 methods × 3 obs modes)
    python scripts/phase1_obs_ablation.py --train

    # Print eval commands (after training; expects SNAPSHOTS to be filled)
    python scripts/phase1_obs_ablation.py --eval

    # Smoke (≤100 train frames, one cell, no W&B)
    python scripts/phase1_obs_ablation.py --smoke

Hand-off to GPU after the smoke completes locally.
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

PHASE1_TASKS = (
    "reach_target_single",
    "dishwasher_close",
    "drawers_open_all",
)

METHODS = {
    "dp":  {"launch": "dp_pixel_safety_bigym",  "exp_dir": "dp_safety"},
    "act": {"launch": "act_pixel_safety_bigym", "exp_dir": "act_safety"},
}

BODYSLAM_MODES = ("off", "oracle", "noisy")

DISRUPTIONS = (
    "INCIDENTAL",
    "SHARED_GOAL",
    "DIRECT",
    "OBSTRUCTION",
    "RANDOM_PERTURBED",
)

# Fill these in after training. Keyed by (method, task, bodyslam_mode).
# Path is relative to REPO_ROOT, pointing at the peak-by-W&B-curve snapshot.
SNAPSHOTS: dict[tuple[str, str, str], str | None] = {
    ("dp", "reach_target_single", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/reach_target_single_20260502122856/snapshots/40000_snapshot.pt"
    for m in METHODS
    for t in PHASE1_TASKS
    for b in BODYSLAM_MODES
}


def _require_amass() -> str:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        sys.stderr.write(
            "AMASS_DATA_DIR is not set. export it to the CMU AMASS root first.\n"
        )
        sys.exit(1)
    return amass


def _resolved_snapshot(method: str, task: str, mode: str) -> Path | None:
    rel = SNAPSHOTS.get((method, task, mode))
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.is_file() else None


def _train_cmd(method: str, task: str, mode: str, seed: int) -> list[str]:
    run_name = f"phase1-train-{method}-{task}-bs{mode}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        f"bodyslam={mode}",
        f"seed={seed}",
        "save_snapshot=true",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-1","obs-ablation","train","{method}","{task}","bs-{mode}"]',
    ]


def _eval_cmd(
    method: str, task: str, mode: str, disruption: str, snapshot: Path,
    *, seed: int, num_eval_episodes: int, wandb_use: bool,
) -> list[str]:
    run_name = f"phase1-eval-{method}-{task}-bs{mode}-{disruption.lower()}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        f"bodyslam={mode}",
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
            f'+wandb.tags=["phase-1","obs-ablation","eval","{method}",'
            f'"{task}","bs-{mode}","{disruption.lower()}"]'
        ),
    ]


def _print_train(seed: int) -> int:
    print(
        f"# Phase-1 E1.1 — obs-ablation training "
        f"({len(PHASE1_TASKS)} tasks × {len(METHODS)} methods × "
        f"{len(BODYSLAM_MODES)} modes = "
        f"{len(PHASE1_TASKS) * len(METHODS) * len(BODYSLAM_MODES)} runs)"
    )
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}\n")
    for method in METHODS:
        for task in PHASE1_TASKS:
            for mode in BODYSLAM_MODES:
                cmd = _train_cmd(method, task, mode, seed)
                print(" ".join(shlex.quote(c) for c in cmd))
            print()
    return 0


def _print_eval(seed: int, num_eval_episodes: int) -> int:
    print(f"# Phase-1 E1.1 — obs-ablation eval ({num_eval_episodes} eps each)")
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}\n")
    missing = []
    for method in METHODS:
        for task in PHASE1_TASKS:
            for mode in BODYSLAM_MODES:
                snap = _resolved_snapshot(method, task, mode)
                if snap is None:
                    missing.append((method, task, mode))
                    print(
                        f"# SKIP {method}/{task}/bs={mode}: no snapshot "
                        f"(rel={SNAPSHOTS.get((method, task, mode))!r})"
                    )
                    continue
                print(
                    f"# --- {method} / {task} / bs={mode} "
                    f"(snapshot: {snap.relative_to(REPO_ROOT)}) ---"
                )
                for disruption in DISRUPTIONS:
                    cmd = _eval_cmd(
                        method, task, mode, disruption, snap,
                        seed=seed,
                        num_eval_episodes=num_eval_episodes,
                        wandb_use=True,
                    )
                    print(" ".join(shlex.quote(c) for c in cmd))
                print()
    if missing:
        print(f"# {len(missing)} cell(s) missing snapshots.")
        print("# Run `python scripts/phase1_obs_ablation.py --train` first.")
    return 0 if not missing else 2


def _smoke(method: str, task: str, mode: str, seed: int) -> int:
    cmd = [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={METHODS[method]['launch']}",
        f"env=safety_bigym/{task}",
        f"bodyslam={mode}",
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
    mode.add_argument("--train", action="store_true",
                      help="Print training commands (18 cells).")
    mode.add_argument("--eval", action="store_true",
                      help="Print eval commands (requires SNAPSHOTS filled).")
    mode.add_argument("--smoke", action="store_true",
                      help="Run a 100-step train smoke locally (no W&B).")
    parser.add_argument("--method", default="dp", choices=sorted(METHODS))
    parser.add_argument("--task", default=PHASE1_TASKS[0])
    parser.add_argument("--bodyslam", default="noisy", choices=BODYSLAM_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    _require_amass()

    if args.smoke:
        return _smoke(args.method, args.task, args.bodyslam, args.seed)
    if args.eval:
        return _print_eval(args.seed, args.num_eval_episodes)
    return _print_train(args.seed)


if __name__ == "__main__":
    sys.exit(main())
