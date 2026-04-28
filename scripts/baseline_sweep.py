#!/usr/bin/env python
"""Phase-0 baseline sweep — eval-from-snapshot × 5 disruption types.

The Phase-0 deliverable is a baseline table: for each task, evaluate one
trained imitation policy against each of the 5 ISO 15066 disruption types,
logging `info["episode_safety"]` to W&B under `phase-0-baseline`.

Two methods are supported via `--method {dp,act}`:
    dp  — Diffusion Policy   (launch=dp_pixel_safety_bigym,  exp_local/dp_safety/)
    act — Action Chunking Tx (launch=act_pixel_safety_bigym, exp_local/act_safety/)

That means 1 policy per (method, task) — NOT one per (task, disruption) cell —
and `len(TASKS) × len(DISRUPTIONS)` short eval runs per method. Disruption
is an *environment* knob applied via `env.disruption_type=<NAME>` which forces
the scenario sampler to emit that single disruption on every episode (see
SafetyBiGymEnvFactory).

Snapshots are produced at every pretrain-eval interval (default every 10k steps)
under:
    exp_local/<method-exp-dir>/<task>_<ts>/snapshots/<step>_snapshot.pt

After each retrain, inspect `pretrain_eval/episode_success` on W&B and update the
`SNAPSHOTS[method]` dict below to point at the peak step — NOT blindly 100000,
which on small pixel demo sets is usually past the over-fit inflection.

If any task has no snapshot, run `--train-missing` first (on GPU).

Usage:
    # 1. Print training commands for missing snapshots (default --method dp)
    python scripts/baseline_sweep.py --train-missing
    python scripts/baseline_sweep.py --method act --train-missing

    # 2. Print the eval commands
    python scripts/baseline_sweep.py --eval
    python scripts/baseline_sweep.py --method act --eval

    # 3. Local smoke test (≤1 eval episode, one cell) — verifies plumbing
    python scripts/baseline_sweep.py --smoke
    python scripts/baseline_sweep.py --method act --smoke

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
    "saucepan_to_hob",
    "dishwasher_close",
    "drawers_open_all",
    "drawer_top_close",
    "sandwich_remove",
)

DISRUPTIONS = (
    "INCIDENTAL",
    "SHARED_GOAL",
    "DIRECT",
    "OBSTRUCTION",
    "RANDOM_PERTURBED",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Headless rendering prefix — GPU nodes have no X display, so MuJoCo needs
# EGL (preferred) or osmesa. Prepended to every command this script emits.
HEADLESS_ENV = ("MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl")

# Per-method launch metadata. The launch name maps 1:1 to a Hydra config under
# cfgs/launch/, and `exp_dir` must match the `hydra.run.dir` prefix in that
# launch config so snapshot paths can be resolved consistently.
METHODS: dict[str, dict[str, str]] = {
    "dp":  {"launch": "dp_pixel_safety_bigym",  "exp_dir": "dp_safety",  "tag": "dp-pixel"},
    "act": {"launch": "act_pixel_safety_bigym", "exp_dir": "act_safety", "tag": "act-pixel"},
}

# Snapshots produced by prior training runs, keyed by method then task.
# Pick the peak-by-W&B-curve checkpoint for each task, not the final one.
# Expected shape:
#   exp_local/<method-exp-dir>/<task>_<YYYYMMDDHHMMSS>/snapshots/<step>_snapshot.pt
SNAPSHOTS: dict[str, dict[str, str | None]] = {
    "dp": {
        "reach_target_single": "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/reach_target_single_20260422193458/snapshots/50000_snapshot.pt",
        "saucepan_to_hob": None,
        "dishwasher_close": "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/dishwasher_close_20260422193511/snapshots/70000_snapshot.pt",
        "drawers_open_all": None,
        "drawer_top_close": None,
        "sandwich_remove": None,
    },
    "act": {
        "reach_target_single": None,
        "saucepan_to_hob": None,
        "dishwasher_close": None,
        "drawers_open_all": None,
        "drawer_top_close": None,
        "sandwich_remove": None,
    },
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


def _resolved_snapshot(task: str, method: str) -> Path | None:
    rel = SNAPSHOTS.get(method, {}).get(task)
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.is_file() else None


def _train_cmd(task: str, method: str, seed: int = 0) -> list[str]:
    run_name = f"phase0-train-{method}-{task}-s{seed}"
    launch = METHODS[method]["launch"]
    # save_snapshot=true is load-bearing: with the robobase drift patch,
    # Workspace._pretrain_on_demos writes `snapshots/<step>_snapshot.pt` at every
    # pretrain-eval interval. Without this flag only the final checkpoint persists,
    # and prior runs have shown DP over-fits past its eval-success peak on small
    # pixel demo sets.
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={launch}",
        f"env=safety_bigym/{task}",
        f"seed={seed}",
        "save_snapshot=true",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-0","baseline","train","{method}","{task}"]',
    ]


def _eval_cmd(
    task: str,
    disruption: str,
    snapshot: Path,
    *,
    method: str,
    seed: int,
    num_eval_episodes: int,
    wandb_use: bool,
) -> list[str]:
    run_name = f"phase0-eval-{method}-{task}-{disruption.lower()}-s{seed}"
    launch = METHODS[method]["launch"]
    # demos=0 is load-bearing: snapshots saved by the robobase drift patch carry
    # `action_stats`/`obs_stats` in-payload, and Workspace.__init__ seeds them onto
    # the env_factory before wrappers capture them by reference. Re-fetching 30-50
    # demos at eval just to recompute identical stats wastes ~1-3 min per cell.
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={launch}",
        f"env=safety_bigym/{task}",
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
            f'+wandb.tags=["phase-0-baseline","eval","{method}","{task}",'
            f'"{disruption.lower()}"]'
        ),
    ]


def _print_grid(seed: int, num_eval_episodes: int, method: str) -> int:
    print(f"# Phase-0 baseline eval sweep — method={method} — run on GPU node")
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}")
    print(
        f"# {len(TASKS)} tasks × {len(DISRUPTIONS)} disruptions = "
        f"{len(TASKS) * len(DISRUPTIONS)} eval runs "
        f"({num_eval_episodes} episodes each)\n"
    )
    missing: list[str] = []
    for task in TASKS:
        snap = _resolved_snapshot(task, method)
        if snap is None:
            missing.append(task)
            print(f"# SKIP {task}: no snapshot on disk "
                  f"(rel={SNAPSHOTS.get(method, {}).get(task)!r})")
            continue
        print(f"# --- {task}  (snapshot: {snap.relative_to(REPO_ROOT)}) ---")
        for disruption in DISRUPTIONS:
            cmd = _eval_cmd(
                task, disruption, snap,
                method=method,
                seed=seed,
                num_eval_episodes=num_eval_episodes,
                wandb_use=True,
            )
            print(" ".join(shlex.quote(c) for c in cmd))
        print()
    if missing:
        print(f"# {len(missing)} task(s) missing snapshots: {missing}")
        print(
            f"# Run `python scripts/baseline_sweep.py --method {method} "
            "--train-missing` first."
        )
    return 0 if not missing else 2


def _run_grid(seed: int, num_eval_episodes: int, method: str) -> int:
    import json
    import tempfile
    print(
        f"# Running {len(TASKS) * len(DISRUPTIONS)} eval runs sequentially "
        f"(method={method})..."
    )
    missing: list[str] = []

    # Check for missing tasks first
    for task in TASKS:
        if _resolved_snapshot(task, method) is None:
            missing.append(task)

    if missing:
        print(f"# {len(missing)} task(s) missing snapshots: {missing}")
        print(
            f"# Run `python scripts/baseline_sweep.py --method {method} "
            "--train-missing` first."
        )
        return 2

    # Loop through and run
    env = os.environ.copy()
    results = {}

    for task in TASKS:
        snap = _resolved_snapshot(task, method)
        results[task] = {}
        print(f"\n# === Running Task: {task} ===")
        for disruption in DISRUPTIONS:
            print(f"\n# --- Disruption: {disruption} ---")

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_out = tmp.name

            cmd = _eval_cmd(
                task, disruption, snap,
                method=method,
                seed=seed,
                num_eval_episodes=num_eval_episodes,
                wandb_use=True,
            )
            cmd.append(f"+eval_output_path={tmp_out}")
            
            argv = list(cmd)
            run_env = env.copy()
            # Extract headless env variables to add to subprocess runtime env
            while argv and "=" in argv[0] and not argv[0].startswith("-"):
                k, v = argv.pop(0).split("=", 1)
                run_env[k] = v
                
            print(">>>", " ".join(shlex.quote(c) for c in argv))
            ret = subprocess.run(argv, cwd=REPO_ROOT, env=run_env).returncode
            if ret != 0:
                print(f"Command failed with code {ret}")
                try: os.remove(tmp_out)
                except: pass
                return ret
                
            try:
                with open(tmp_out, "r") as f:
                    metrics = json.load(f)
                results[task][disruption] = metrics
            except Exception as e:
                print(f"Failed to read metrics for {task}/{disruption}: {e}")
            finally:
                try: os.remove(tmp_out)
                except: pass
    
    # Dump all results locally
    out_file = REPO_ROOT / f"baseline_sweep_results_{method}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n# Saved full JSON results to {out_file}")

    # Print summary table
    print("\n\n" + "="*100)
    print(f"BASELINE SWEEP RESULTS SUMMARY — method={method}")
    print("="*100)
    
    headers = ["Task", "Disruption", "Success Rate", "SSM Viol Rate", "Max Force (N)", "1st Viol Step"]
    row_format = "{:<25} | {:<20} | {:<12} | {:<13} | {:<13} | {:<13}"
    print(row_format.format(*headers))
    print("-" * 100)
    
    for task in TASKS:
        for disruption in DISRUPTIONS:
            m = results[task].get(disruption, {})
            succ_val = m.get("episode_success")
            succ = f"{succ_val:.2f}" if succ_val is not None else "N/A"
            
            safety = m.get("env_info/episode_safety", {})
            ssm = f"{safety.get('ep_ssm_violation_rate', 0.0):.2f}"
            force = f"{safety.get('ep_max_contact_force', 0.0):.2f}"
            first = str(safety.get('ep_time_to_first_violation', -1))
            
            print(row_format.format(task, disruption, succ, ssm, force, first))
        print("-" * 100)

    return 0


def _print_train_missing(seed: int, method: str) -> int:
    missing = [t for t in TASKS if _resolved_snapshot(t, method) is None]
    if not missing:
        print(f"# All {method} task snapshots already present. Nothing to train.")
        return 0
    print(f"# Phase-0 training for missing {method} policies — run on GPU")
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}")
    print(f"# Missing: {missing}\n")
    for task in missing:
        cmd = _train_cmd(task, method=method, seed=seed)
        print(" ".join(shlex.quote(c) for c in cmd))
    return 0


def _smoke(task: str, disruption: str, seed: int, method: str) -> int:
    snap = _resolved_snapshot(task, method)
    if snap is None:
        available = [t for t in TASKS if _resolved_snapshot(t, method)]
        sys.stderr.write(
            f"No {method} snapshot for task={task}. Pick one of: {available}\n"
        )
        return 1
    cmd = _eval_cmd(
        task, disruption, snap,
        method=method,
        seed=seed,
        num_eval_episodes=1,
        wandb_use=False,
    )
    print(">>> smoke:", " ".join(shlex.quote(c) for c in cmd))
    env = os.environ.copy()
    argv = list(cmd)
    while argv and "=" in argv[0] and not argv[0].startswith("-"):
        k, v = argv.pop(0).split("=", 1)
        env[k] = v
    return subprocess.run(argv, cwd=REPO_ROOT, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", action="store_true",
                      help="Print the eval-from-snapshot commands.")
    mode.add_argument("--run", action="store_true",
                      help="Run the eval-from-snapshot commands sequentially.")
    mode.add_argument("--train-missing", action="store_true",
                      help="Print training commands for tasks without snapshots.")
    mode.add_argument("--smoke", action="store_true",
                      help="Run one eval episode locally (pipeline check).")
    parser.add_argument("--method", default="dp", choices=sorted(METHODS),
                        help="Imitation method to sweep (default: dp).")
    parser.add_argument("--task", default=TASKS[0])
    parser.add_argument("--disruption", default=DISRUPTIONS[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    _require_amass()

    if args.train_missing:
        return _print_train_missing(args.seed, args.method)
    if args.smoke:
        return _smoke(args.task, args.disruption, args.seed, args.method)
    if args.run:
        return _run_grid(args.seed, args.num_eval_episodes, args.method)
    # Default action is --eval (the deliverable).
    return _print_grid(args.seed, args.num_eval_episodes, args.method)


if __name__ == "__main__":
    sys.exit(main())
