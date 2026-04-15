#!/usr/bin/env python
"""Phase-0 baseline sweep — eval-from-snapshot × 5 disruption types.

The Phase-0 deliverable is a baseline table: for each of 3 tasks, evaluate
one trained Diffusion Policy against each of the 5 ISO 15066 disruption
types, logging `info["episode_safety"]` to W&B under `phase-0-baseline`.

That means 1 DP per task (NOT one per (task, disruption) cell), and
15 short eval runs total — disruption is an *environment* knob applied via
`env.disruption_type=<NAME>` which forces the scenario sampler to emit that
single disruption on every episode (see SafetyBiGymEnvFactory).

Snapshots are expected at:
    exp_local/dp_safety/<task>_<ts>/snapshots/100000_snapshot.pt

If `dishwasher_close` has no snapshot, run `--train-missing` first (on GPU).

Usage:
    # 1. Print the training command for the one missing DP (run on GPU)
    python scripts/baseline_sweep.py --train-missing

    # 2. Print the 15 eval commands (run on GPU)
    python scripts/baseline_sweep.py --eval

    # 3. Local smoke test (≤1 eval episode, one cell) — verifies plumbing
    python scripts/baseline_sweep.py --smoke

Claude Code must not launch multi-hour jobs — it only prints / smokes.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

TASKS = (
    "reach_target_single",
    "dishwasher_load_plates",
    "dishwasher_close",
)

DISRUPTIONS = (
    "INCIDENTAL",
    "SHARED_GOAL",
    "DIRECT",
    "OBSTRUCTION",
    "RANDOM_PERTURBED",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Snapshots produced by prior training runs. Paths are relative to REPO_ROOT.
# Update these when new snapshots land.
SNAPSHOTS: dict[str, str | None] = {
    "reach_target_single":
        "exp_local/dp_safety/reach_target_single_20260218193921/snapshots/100000_snapshot.pt",
    "dishwasher_load_plates":
        "exp_local/dp_safety/dishwasher_load_plates_20260304161104/snapshots/100000_snapshot.pt",
    "dishwasher_close": None,  # not yet trained — see --train-missing
}


def _require_amass() -> str:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        sys.stderr.write(
            "AMASS_DATA_DIR is not set. Export it to the CMU AMASS root, e.g.\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU\n"
        )
        sys.exit(1)
    return amass


def _resolved_snapshot(task: str) -> Path | None:
    rel = SNAPSHOTS.get(task)
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.is_file() else None


def _train_cmd(task: str, seed: int = 0) -> list[str]:
    run_name = f"phase0-train-{task}-s{seed}"
    return [
        sys.executable,
        "train_safety.py",
        "launch=dp_pixel_safety_bigym",
        f"env=safety_bigym/{task}",
        f"seed={seed}",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-0","baseline","train","{task}"]',
    ]


def _eval_cmd(
    task: str,
    disruption: str,
    snapshot: Path,
    *,
    seed: int,
    num_eval_episodes: int,
    wandb_use: bool,
) -> list[str]:
    run_name = f"phase0-eval-{task}-{disruption.lower()}-s{seed}"
    return [
        sys.executable,
        "train_safety.py",
        "launch=dp_pixel_safety_bigym",
        f"env=safety_bigym/{task}",
        f"+env.disruption_type={disruption}",
        f"+snapshot_path={snapshot}",
        "num_train_frames=0",
        "num_pretrain_steps=0",
        f"num_eval_episodes={num_eval_episodes}",
        "eval_every_steps=1",
        f"seed={seed}",
        f"wandb.use={'true' if wandb_use else 'false'}",
        f"wandb.name={run_name}",
        (
            f'+wandb.tags=["phase-0-baseline","eval","{task}",'
            f'"{disruption.lower()}"]'
        ),
    ]


def _print_grid(seed: int, num_eval_episodes: int) -> int:
    print("# Phase-0 baseline eval sweep — run on GPU node")
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}")
    print(
        f"# {len(TASKS)} tasks × {len(DISRUPTIONS)} disruptions = "
        f"{len(TASKS) * len(DISRUPTIONS)} eval runs "
        f"({num_eval_episodes} episodes each)\n"
    )
    missing: list[str] = []
    for task in TASKS:
        snap = _resolved_snapshot(task)
        if snap is None:
            missing.append(task)
            print(f"# SKIP {task}: no snapshot on disk "
                  f"(rel={SNAPSHOTS.get(task)!r})")
            continue
        print(f"# --- {task}  (snapshot: {snap.relative_to(REPO_ROOT)}) ---")
        for disruption in DISRUPTIONS:
            cmd = _eval_cmd(
                task, disruption, snap,
                seed=seed,
                num_eval_episodes=num_eval_episodes,
                wandb_use=True,
            )
            print(" ".join(shlex.quote(c) for c in cmd))
        print()
    if missing:
        print(f"# {len(missing)} task(s) missing snapshots: {missing}")
        print("# Run `python scripts/baseline_sweep.py --train-missing` first.")
    return 0 if not missing else 2


def _print_train_missing(seed: int) -> int:
    missing = [t for t in TASKS if _resolved_snapshot(t) is None]
    if not missing:
        print("# All task snapshots already present. Nothing to train.")
        return 0
    print("# Phase-0 training for missing DPs — run on GPU")
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}")
    print(f"# Missing: {missing}\n")
    for task in missing:
        cmd = _train_cmd(task, seed=seed)
        print(" ".join(shlex.quote(c) for c in cmd))
    return 0


def _smoke(task: str, disruption: str, seed: int) -> int:
    snap = _resolved_snapshot(task)
    if snap is None:
        sys.stderr.write(
            f"No snapshot for task={task}. Pick one of: "
            f"{[t for t in TASKS if _resolved_snapshot(t)]}\n"
        )
        return 1
    cmd = _eval_cmd(
        task, disruption, snap,
        seed=seed,
        num_eval_episodes=1,
        wandb_use=False,
    )
    print(">>> smoke:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", action="store_true",
                      help="Print the 15 eval-from-snapshot commands.")
    mode.add_argument("--train-missing", action="store_true",
                      help="Print training commands for tasks without snapshots.")
    mode.add_argument("--smoke", action="store_true",
                      help="Run one eval episode locally (pipeline check).")
    parser.add_argument("--task", default=TASKS[0])
    parser.add_argument("--disruption", default=DISRUPTIONS[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    _require_amass()

    if args.train_missing:
        return _print_train_missing(args.seed)
    if args.smoke:
        return _smoke(args.task, args.disruption, args.seed)
    # Default action is --eval (the deliverable).
    return _print_grid(args.seed, args.num_eval_episodes)


if __name__ == "__main__":
    sys.exit(main())
