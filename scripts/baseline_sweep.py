#!/usr/bin/env python
"""Phase-0 baseline sweep over (task × disruption type).

Prints the command grid for a baseline DP run across the 3 Phase-0 tasks
and 5 ISO 15066 disruption types. Does NOT launch training itself — Claude
Code must not start multi-hour jobs. The human copies the printed commands
onto a GPU node.

Smoke test:
    python scripts/baseline_sweep.py --smoke
    # → runs train_safety.py for 100 frames on one (task, disruption) cell,
    #   verifying the pipeline + W&B episode_safety/* wiring end-to-end.

Full sweep (run on GPU):
    python scripts/baseline_sweep.py --print
    # → prints all 15 shell commands; redirect or eval them as needed.
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


def _require_amass() -> str:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        sys.stderr.write(
            "AMASS_DATA_DIR is not set. Export it to the CMU AMASS root, e.g.\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU\n"
        )
        sys.exit(1)
    return amass


def _build_cmd(
    task: str,
    disruption: str,
    *,
    num_frames: int,
    wandb_use: bool,
    seed: int,
) -> list[str]:
    run_name = f"phase0-{task}-{disruption.lower()}-s{seed}"
    args = [
        sys.executable,
        "train_safety.py",
        "launch=dp_pixel_safety_bigym",
        f"env=safety_bigym/{task}",
        f"num_train_frames={num_frames}",
        f"num_pretrain_steps={min(num_frames, 100)}",
        f"seed={seed}",
        f"wandb.use={'true' if wandb_use else 'false'}",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-0","baseline","{task}","{disruption.lower()}"]',
    ]
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a ≤100-step smoke test against one (task, disruption).",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the full sweep commands and exit (default).",
    )
    parser.add_argument(
        "--task", default=TASKS[0], help="Task for --smoke run."
    )
    parser.add_argument(
        "--disruption",
        default=DISRUPTIONS[0],
        help="Disruption type for --smoke run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    _require_amass()
    repo_root = Path(__file__).resolve().parent.parent

    if args.smoke:
        cmd = _build_cmd(
            args.task,
            args.disruption,
            num_frames=100,
            wandb_use=args.wandb,
            seed=args.seed,
        )
        print(">>> smoke:", " ".join(shlex.quote(c) for c in cmd))
        result = subprocess.run(cmd, cwd=repo_root)
        return result.returncode

    # Default: print full grid (user launches on GPU).
    print("# Phase-0 baseline sweep — run on GPU node")
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}")
    print(f"# {len(TASKS)} tasks × {len(DISRUPTIONS)} disruptions = "
          f"{len(TASKS) * len(DISRUPTIONS)} runs\n")
    for task in TASKS:
        for disruption in DISRUPTIONS:
            cmd = _build_cmd(
                task,
                disruption,
                num_frames=100_000,
                wandb_use=True,
                seed=args.seed,
            )
            print(" ".join(shlex.quote(c) for c in cmd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
