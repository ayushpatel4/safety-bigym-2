#!/usr/bin/env python
"""Phase-1 E1.3 — temporal-failure-mode ablation.

Isolate which class of BodySLAM++ failure mode matters most by pinning
sigma to 0.05 m and toggling the structural knobs. 3 configs x 3 tasks
= 9 runs.

  * iid  : alpha=0,   occlusion=off, dropout=0     (white-noise floor)
  * ou   : alpha=0.9, occlusion=off, dropout=0     (OU-correlated noise only)
  * full : alpha=0.9, occlusion=on,  dropout=0.02  (realistic regime)

Latency stays at the BodySLAM++ default of 2 steps across all three —
the comparison is strictly about noise structure, not reaction time.

Expected shape: iid < ou (OU models real drift); full tracks ou unless
occlusion/dropout drive a large gap, which would motivate explicit
occlusion handling in the downstream safety critic.

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

# (tag, alpha, use_occlusion, dropout_prob)
CONFIGS = (
    ("iid",  0.0, False, 0.00),
    ("ou",   0.9, False, 0.00),
    ("full", 0.9, True,  0.02),
)


def _extra_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))


def _build(args: argparse.Namespace, tag: str):
    for task in args.tasks:
        for name, alpha, use_occ, drop in CONFIGS:
            run_name = f"phase1_e1_3_{task}_{name}_{tag}"
            cmd = [
                "python", "train_safety.py",
                "launch=dp_pixel_safety_bigym",
                f"env=safety_bigym/{task}",
                "env.body_slam.mode=noisy",
                f"env.body_slam.alpha={alpha}",
                f"env.body_slam.use_occlusion={str(use_occ).lower()}",
                f"env.body_slam.dropout_prob={drop}",
                "wandb.use=true",
                "wandb.project=safety-critic",
                f"wandb.name={run_name}",
                "+wandb.tags=[phase-1,e1.3,temporal-ablation]",
            ]
            yield run_name, cmd


def main() -> None:
    run_experiment(
        name="E1.3 temporal-ablation",
        log_subdir="e1_3",
        build_commands=_build,
        extra_args=_extra_args,
    )


if __name__ == "__main__":
    main()
