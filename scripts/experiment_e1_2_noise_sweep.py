#!/usr/bin/env python
"""Phase-1 E1.2 — BodySLAM++ noise-level sweep.

Once E1.1 establishes that human-state observations help, E1.2 asks:
how much position noise can the policy tolerate before the benefit
collapses? Sweep sigma across the BodySLAM++ ATE range (1-3 cm) plus
a stress-test tail (5, 10, 15, 20 cm), all other failure modes pinned
to their realistic defaults. 5 sigmas x 3 tasks = 15 runs.

This is a sensitivity study, not a hyperparameter search — freeze
policy hyperparameters to the E1.1 winner (expected: noisy mode).

Modes:
  --print-only (default)  Print + log commands, don't execute.
  --run                   Execute every command serially with EGL headless.
  --smoke                 Execute one 100-step run to verify plumbing.
"""

from __future__ import annotations

import argparse

from _experiment_runner import run_experiment

TASKS = (
    "reach_target_single",
    "dishwasher_load_plates",
    "dishwasher_close",
)

SIGMAS = (0.02, 0.05, 0.10, 0.15, 0.20)


def _extra_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))


def _build(args: argparse.Namespace, tag: str):
    for task in args.tasks:
        for sigma in args.sigmas:
            sigma_tag = f"{int(round(sigma * 100)):03d}cm"
            run_name = f"phase1_e1_2_{task}_sigma_{sigma_tag}_{tag}"
            cmd = [
                "python", "train_safety.py",
                "launch=dp_pixel_safety_bigym",
                f"env=safety_bigym/{task}",
                "env.body_slam.mode=noisy",
                f"env.body_slam.sigma={sigma}",
                "wandb.use=true",
                "wandb.project=safety-critic",
                f"wandb.name={run_name}",
                "+wandb.tags=[phase-1,e1.2,noise-sweep]",
            ]
            yield run_name, cmd


def main() -> None:
    run_experiment(
        name="E1.2 noise-sweep",
        log_subdir="e1_2",
        build_commands=_build,
        extra_args=_extra_args,
    )


if __name__ == "__main__":
    main()
