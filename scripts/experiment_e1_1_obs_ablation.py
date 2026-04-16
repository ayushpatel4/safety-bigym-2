#!/usr/bin/env python
"""Phase-1 E1.1 — observation-mode ablation (baseline / oracle / noisy).

Hypothesis: giving the policy *any* human state information reduces SSM
violations relative to the Phase-0 blind baseline. Three variants of
BodySLAMConfig.mode across three tasks = 9 training runs.

  * baseline : mode=off     (no wrapper — Phase-0 control)
  * oracle   : mode=oracle  (sigma=0, no occlusion/dropout/latency — upper bound)
  * noisy    : mode=noisy   (full BodySLAM++ failure-mode cocktail)

Decision rule (HYBRID_SAFETY_CRITIC_PLAN): oracle reduces
`ep_ssm_violation_frac` by >=20% vs baseline => Phase 2 (Safety Value
Function) is justified. Otherwise, cost signal is the bottleneck —
jump to Phase 3.

Modes:
  --print-only (default)  Print + log commands, don't execute.
  --run                   Execute every command serially with EGL headless.
  --smoke                 Execute one 100-step run to verify plumbing.

  --continue-on-error     Don't abort the sweep when a run fails.
  --dry-run               With --run, print what would execute without invoking.
"""

from __future__ import annotations

import argparse

from _experiment_runner import run_experiment

TASKS = (
    "reach_target_single",
    "dishwasher_load_plates",
    "dishwasher_close",
)

MODES = ("off", "oracle", "noisy")


def _extra_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--modes", nargs="+", default=list(MODES))


def _build(args: argparse.Namespace, tag: str):
    for task in args.tasks:
        for mode in args.modes:
            run_name = f"phase1_e1_1_{task}_{mode}_{tag}"
            cmd = [
                "python", "train_safety.py",
                "launch=dp_pixel_safety_bigym",
                f"env=safety_bigym/{task}",
                f"env.body_slam.mode={mode}",
                "wandb.use=true",
                "wandb.project=safety-critic",
                f"wandb.name={run_name}",
                "+wandb.tags=[phase-1,e1.1,obs-ablation]",
            ]
            yield run_name, cmd


def main() -> None:
    run_experiment(
        name="E1.1 obs-ablation",
        log_subdir="e1_1",
        build_commands=_build,
        extra_args=_extra_args,
    )


if __name__ == "__main__":
    main()
