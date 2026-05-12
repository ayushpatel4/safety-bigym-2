#!/usr/bin/env python
"""Phase-1.4 — reward-on pilot. Does a reward gradient turn the
`human_pos_estimate` channel from ignored to used?

E1.1 (BC) was a negative result, but BC ignores env reward — flipping the
SSM/PFL penalty for ACT/DP is mechanistically inert. This pilot uses
DrQ-V2+ (demo-driven online pixel RL) so the safety penalty becomes a
learning gradient.

Scope: two tasks (reach_target_single, saucepan_to_hob) × three
`bodyslam` modes = 6 cells. reach is the cleanest baseline (E1.1 task
success 0.88); saucepan is where E1.1 showed the most interesting
oracle behaviour (task success jumped 0.46 → 0.64 in oracle without
reducing SSM — the policy was using human state for task progress, not
safety).

Decision rule (off → mode SSM-violation-rate reduction, averaged across
the 5 disruption types):
  - ≥ +20% reduction on either task → channel + reward together work;
    greenlight Phase 3.
  - ≈ 0 on both tasks → channel doesn't help even with reward; pivot
    to Phase 2 (filter) and/or alternate reward shaping.
  - off wins on both → reward shaping alone suffices; retire the channel.

Usage:
    python scripts/phase1_reward_pilot.py --train  # 6 train commands
    python scripts/phase1_reward_pilot.py --eval   # 30 eval commands
    python scripts/phase1_reward_pilot.py --run    # execute evals + table
    python scripts/phase1_reward_pilot.py --smoke  # ≤100 frames, no W&B
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

TASKS = ("reach_target_single", "saucepan_to_hob")
LAUNCH = "drqv2plus_pixel_safety_bigym"
EXP_DIR = "drqv2plus_safety"

BODYSLAM_MODES = ("off", "oracle", "noisy")
DISRUPTIONS = (
    "INCIDENTAL", "SHARED_GOAL", "DIRECT", "OBSTRUCTION", "RANDOM_PERTURBED",
)

# Fill after training. Peak-by-W&B-curve snapshot per (task, mode).
SNAPSHOTS: dict[tuple[str, str], str | None] = {
    (t, m): None for t in TASKS for m in BODYSLAM_MODES
}


def _require_amass() -> str:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        sys.stderr.write("AMASS_DATA_DIR not set.\n")
        sys.exit(1)
    return amass


def _resolved_snapshot(task: str, mode: str) -> Path | None:
    rel = SNAPSHOTS.get((task, mode))
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.is_file() else None


def _train_cmd(task: str, mode: str, seed: int) -> list[str]:
    run_name = f"phase1r-train-{task}-bs{mode}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={LAUNCH}",
        f"env=safety_bigym/{task}",
        f"bodyslam={mode}",
        f"seed={seed}",
        "save_snapshot=true",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-1-reward","train","drqv2plus","{task}","bs-{mode}"]',
    ]


def _eval_cmd(
    task: str, mode: str, disruption: str, snapshot: Path,
    *, seed: int, num_eval_episodes: int, wandb_use: bool,
) -> list[str]:
    run_name = f"phase1r-eval-{task}-bs{mode}-{disruption.lower()}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={LAUNCH}",
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
            f'+wandb.tags=["phase-1-reward","eval","drqv2plus","{task}",'
            f'"bs-{mode}","{disruption.lower()}"]'
        ),
    ]


def _print_train(seed: int) -> int:
    n_cells = len(TASKS) * len(BODYSLAM_MODES)
    print(f"# Phase-1.4 reward pilot — {n_cells} train cells "
          f"({len(TASKS)} tasks × {len(BODYSLAM_MODES)} modes)")
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}\n")
    for task in TASKS:
        print(f"# --- {task} ---")
        for mode in BODYSLAM_MODES:
            print(" ".join(shlex.quote(c) for c in _train_cmd(task, mode, seed)))
        print()
    return 0


def _print_eval(seed: int, num_eval_episodes: int) -> int:
    n_cells = len(TASKS) * len(BODYSLAM_MODES) * len(DISRUPTIONS)
    print(f"# Phase-1.4 reward pilot — eval "
          f"({n_cells} cells, {num_eval_episodes} eps each)")
    missing: list[tuple[str, str]] = []
    for task in TASKS:
        for mode in BODYSLAM_MODES:
            snap = _resolved_snapshot(task, mode)
            if snap is None:
                missing.append((task, mode))
                print(f"# SKIP {task}/bs={mode}: no snapshot "
                      f"(rel={SNAPSHOTS.get((task, mode))!r})")
                continue
            print(f"# --- {task} / bs={mode}  "
                  f"({snap.relative_to(REPO_ROOT)}) ---")
            for disruption in DISRUPTIONS:
                cmd = _eval_cmd(
                    task, mode, disruption, snap,
                    seed=seed,
                    num_eval_episodes=num_eval_episodes,
                    wandb_use=True,
                )
                print(" ".join(shlex.quote(c) for c in cmd))
            print()
    if missing:
        print(f"# {len(missing)} cell(s) missing snapshots: {missing}")
        print("# Run `python scripts/phase1_reward_pilot.py --train` first.")
    return 0 if not missing else 2


def _safety_lookup(metrics: dict, key: str, default=0.0):
    nested = metrics.get("env_info/episode_safety")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    flat = metrics.get(f"env_info/episode_safety/{key}")
    return flat if flat is not None else default


def _run_grid(seed: int, num_eval_episodes: int) -> int:
    import json
    import tempfile

    missing = [
        (t, m) for t in TASKS for m in BODYSLAM_MODES
        if _resolved_snapshot(t, m) is None
    ]
    if missing:
        print(f"# missing snapshots: {missing}; run --train first.")
        return 2

    base_env = os.environ.copy()
    results: dict = {}
    n_cells = len(TASKS) * len(BODYSLAM_MODES) * len(DISRUPTIONS)
    cell_idx = 0
    for task in TASKS:
        results.setdefault(task, {})
        for mode in BODYSLAM_MODES:
            results[task].setdefault(mode, {})
            snap = _resolved_snapshot(task, mode)
            for disruption in DISRUPTIONS:
                cell_idx += 1
                print(f"\n# [{cell_idx}/{n_cells}] {task} / bs={mode} / "
                      f"{disruption}")
                with tempfile.NamedTemporaryFile(
                    suffix=".json", delete=False
                ) as tmp:
                    tmp_out = tmp.name
                cmd = _eval_cmd(
                    task, mode, disruption, snap,
                    seed=seed,
                    num_eval_episodes=num_eval_episodes,
                    wandb_use=True,
                )
                cmd.append(f"+eval_output_path={tmp_out}")
                argv = list(cmd)
                run_env = base_env.copy()
                while argv and "=" in argv[0] and not argv[0].startswith("-"):
                    k, v = argv.pop(0).split("=", 1)
                    run_env[k] = v
                print(">>>", " ".join(shlex.quote(c) for c in argv))
                rc = subprocess.run(argv, cwd=REPO_ROOT, env=run_env).returncode
                if rc != 0:
                    print(f"# Cell failed (rc={rc}); aborting.")
                    try: os.remove(tmp_out)
                    except: pass
                    return rc
                try:
                    with open(tmp_out) as f:
                        results[task][mode][disruption] = json.load(f)
                except Exception as exc:
                    print(f"# Failed to read {tmp_out}: {exc}")
                finally:
                    try: os.remove(tmp_out)
                    except: pass

    out_path = REPO_ROOT / "phase1_reward_pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n# Full JSON: {out_path}")

    print("\n" + "=" * 96)
    print(f"PHASE-1.4 REWARD PILOT  (avg over {len(DISRUPTIONS)} disruptions)")
    print("=" * 96)
    print(f"{'task':<22} {'mode':<8} {'ssm_viol':>10} {'pfl_viol':>10} "
          f"{'success':>9} {'redux':>10} {'criterion':>10}")
    print("-" * 96)

    any_pass = False
    per_task_pass: dict[str, bool] = {}
    for task in TASKS:
        task_pass = False
        rows = {}
        for mode in BODYSLAM_MODES:
            runs = results.get(task, {}).get(mode, {})
            if not runs:
                rows[mode] = None
                continue
            ssm = [_safety_lookup(m, "ep_ssm_violation_rate")
                   for m in runs.values()]
            pfl = [_safety_lookup(m, "ep_pfl_violation_rate")
                   for m in runs.values()]
            succ = [m.get("episode_success", 0.0) for m in runs.values()]
            rows[mode] = {
                "ssm": sum(ssm) / len(ssm),
                "pfl": sum(pfl) / len(pfl),
                "success": sum(succ) / len(succ),
            }
        off = rows.get("off")
        for mode in BODYSLAM_MODES:
            r = rows[mode]
            if r is None:
                print(f"{task:<22} {mode:<8} {'-':>10} {'-':>10} {'-':>9} "
                      f"{'-':>10} {'-':>10}")
                continue
            if mode == "off" or off is None or off["ssm"] == 0:
                redux, crit = "-", "-"
            else:
                r_pct = (off["ssm"] - r["ssm"]) / off["ssm"]
                redux = f"{r_pct*100:+.1f}%"
                crit = "PASS" if r_pct >= 0.20 else ""
                if r_pct >= 0.20:
                    task_pass = True
                    any_pass = True
            print(f"{task:<22} {mode:<8} {r['ssm']:>10.3f} "
                  f"{r['pfl']:>10.3f} {r['success']:>9.2f} "
                  f"{redux:>10} {crit:>10}")
        print("-" * 96)
        per_task_pass[task] = task_pass

    print()
    print("Success criterion: oracle (or noisy) ≥ 20% reduction in SSM "
          "violation rate vs `off`, on at least one task.")
    for t, ok in per_task_pass.items():
        print(f"  {t}: {'PASS' if ok else 'FAIL'}")
    print(f"\nDecision: {'PASS — greenlight Phase 3 with obs channel' if any_pass else 'FAIL — pivot to Phase 2 filter / alternate reward shaping'}")
    return 0


def _smoke(task: str, mode: str, seed: int) -> int:
    cmd = [
        *HEADLESS_ENV,
        sys.executable,
        "train_safety.py",
        f"launch={LAUNCH}",
        f"env=safety_bigym/{task}",
        f"bodyslam={mode}",
        f"seed={seed}",
        "num_train_frames=100",
        "num_pretrain_steps=0",
        "demos=2",
        "num_eval_episodes=0",
        "replay_size_before_train=25000",
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
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--task", default=TASKS[0], choices=TASKS)
    parser.add_argument("--bodyslam", default="oracle", choices=BODYSLAM_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    _require_amass()

    if args.smoke:
        return _smoke(args.task, args.bodyslam, args.seed)
    if args.run:
        return _run_grid(args.seed, args.num_eval_episodes)
    if args.eval:
        return _print_eval(args.seed, args.num_eval_episodes)
    return _print_train(args.seed)


if __name__ == "__main__":
    sys.exit(main())
