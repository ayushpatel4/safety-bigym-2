#!/usr/bin/env python
"""Phase-1 all-in-one driver — runs E1.1, E1.2, E1.3 end-to-end.

Default order: E1.1 first (observation-mode ablation is the decision
gate for Phase 2), then E1.2 (noise sweep) and E1.3 (temporal ablation)
in parallel intent but serial execution since each job uses the full
GPU.

Forwards --run / --print-only / --smoke / --continue-on-error to every
sub-experiment. Use this when you want a single fire-and-forget command
on the GPU box.

Usage:
    # Dry-run: print all 33 commands, log each experiment's file
    python scripts/run_phase1_all.py

    # Execute everything (expect many hours)
    python scripts/run_phase1_all.py --run

    # Smoke all three (100-step each) to confirm plumbing
    python scripts/run_phase1_all.py --smoke

    # Skip a stage if already complete
    python scripts/run_phase1_all.py --run --skip e1.2
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STAGES = {
    "e1.1": "experiment_e1_1_obs_ablation.py",
    "e1.2": "experiment_e1_2_noise_sweep.py",
    "e1.3": "experiment_e1_3_temporal_ablation.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--print-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip", nargs="+", default=[],
        choices=sorted(STAGES),
        help="Stages to skip (e.g. --skip e1.2 e1.3).",
    )
    parser.add_argument(
        "--only", nargs="+", default=[],
        choices=sorted(STAGES),
        help="If set, run only these stages.",
    )
    args = parser.parse_args()

    stages = args.only or [s for s in STAGES if s not in args.skip]
    if not stages:
        sys.stderr.write("No stages selected.\n")
        sys.exit(2)

    forwarded: list[str] = []
    if args.run:
        forwarded.append("--run")
    elif args.smoke:
        forwarded.append("--smoke")
    elif args.print_only:
        forwarded.append("--print-only")
    if args.continue_on_error:
        forwarded.append("--continue-on-error")
    if args.dry_run:
        forwarded.append("--dry-run")

    start = dt.datetime.now()
    results: list[tuple[str, int]] = []
    for stage in stages:
        script = REPO_ROOT / "scripts" / STAGES[stage]
        cmd = [sys.executable, str(script), *forwarded]
        print()
        print(f"###### {stage.upper()} — {script.name} ######")
        print(" ".join(cmd))
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        results.append((stage, rc))
        if rc != 0 and not args.continue_on_error:
            print(
                f"# {stage} exited {rc}; aborting. "
                f"Pass --continue-on-error to keep going."
            )
            break

    elapsed = dt.datetime.now() - start
    ok = sum(1 for _, rc in results if rc == 0)
    fail = len(results) - ok
    print()
    print(f"###### phase-1 summary: {ok} ok, {fail} failed, elapsed {elapsed}")
    for stage, rc in results:
        flag = "OK " if rc == 0 else f"FAIL({rc})"
        print(f"  {flag}  {stage}")

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
