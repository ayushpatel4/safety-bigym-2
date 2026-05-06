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
    "saucepan_to_hob"
)

METHODS = {
    # "dp":  {"launch": "dp_pixel_safety_bigym",  "exp_dir": "dp_safety"},
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
    # ("dp", "reach_target_single", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/reach_target_single_20260502122856/snapshots/40000_snapshot.pt"
    # ("dp", "reach_target_single", "noisy"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/reach_target_single_20260502181921/snapshots/50000_snapshot.pt"
    # ("dp", "reach_target_single", "off"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/reach_target_single_20260502181132/snapshots/50000_snapshot.pt",

    # ("dp", "dishwasher_close", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/dishwasher_close_20260503022632/snapshots/10000_snapshot.pt"
    # ("dp", "dishwasher_close", "noisy"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/dishwasher_close_20260503122249/snapshots/20000_snapshot.pt"
    # ("dp", "dishwasher_close", "off"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/dishwasher_close_20260503021247/snapshots/10000_snapshot.pt",
    
    # ("dp", "drawers_open_all", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/drawers_open_all_20260503043644/snapshots/20000_snapshot.pt"
    # ("dp", "drawers_open_all", "noisy"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/drawers_open_all_20260503045446/snapshots/20000_snapshot.pt"
    # ("dp", "drawers_open_all", "off"): "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/drawers_open_all_20260503044952/snapshots/20000_snapshot.pt",
    ("act", "reach_target_single", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/reach_target_single_20260504000016/snapshots/20000_snapshot.pt",
    ("act", "reach_target_single", "noisy"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/reach_target_single_20260504000103/snapshots/20000_snapshot.pt",
    ("act", "reach_target_single", "off"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/reach_target_single_20260503235934/snapshots/50000_snapshot.pt",
    ("act", "dishwasher_close", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/dishwasher_close_20260504133104/snapshots/20000_snapshot.pt",
    ("act", "dishwasher_close", "noisy"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/dishwasher_close_20260504133124/snapshots/30000_snapshot.pt",
    ("act", "dishwasher_close", "off"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/dishwasher_close_20260504133029/snapshots/40000_snapshot.pt",
    ("act", "drawers_open_all", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/drawers_open_all_20260505031240/snapshots/70000_snapshot.pt",
    ("act", "drawers_open_all", "noisy"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/drawers_open_all_20260505031217/snapshots/20000_snapshot.pt",
    ("act", "drawers_open_all", "off"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/drawers_open_all_20260504232537/snapshots/70000_snapshot.pt",
    ("act", "saucepan_to_hob", "oracle"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/saucepan_to_hob_20260505105023/snapshots/80000_snapshot.pt",
    ("act", "saucepan_to_hob", "noisy"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/saucepan_to_hob_20260505105056/snapshots/50000_snapshot.pt",
    ("act", "saucepan_to_hob", "off"): "/home/ap2322/Documents/safety_bigym/exp_local/act_safety/saucepan_to_hob_20260505105006/snapshots/80000_snapshot.pt",
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


def _safety_lookup(metrics: dict, key: str, default=0.0):
    """eval_metrics may flatten nested keys with '/' or keep them nested.
    Try both shapes."""
    nested = metrics.get("env_info/episode_safety")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    flat = metrics.get(f"env_info/episode_safety/{key}")
    if flat is not None:
        return flat
    return default


def _run_grid(seed: int, num_eval_episodes: int) -> int:
    """Execute eval cells sequentially, capture per-cell JSON, print a
    summary table with the 20%-SSM-reduction success-criterion check."""
    import json
    import tempfile

    missing = [
        (m, t, b)
        for m in METHODS for t in PHASE1_TASKS for b in BODYSLAM_MODES
        if _resolved_snapshot(m, t, b) is None
    ]
    if missing:
        print(f"# {len(missing)} cell(s) missing snapshots: {missing}")
        print("# Run `python scripts/phase1_obs_ablation.py --train` first.")
        return 2

    base_env = os.environ.copy()
    results: dict = {}

    n_cells = len(METHODS) * len(PHASE1_TASKS) * len(BODYSLAM_MODES) * len(DISRUPTIONS)
    print(f"# Running {n_cells} eval cells sequentially "
          f"({num_eval_episodes} eps each)...")

    cell_idx = 0
    for method in METHODS:
        results.setdefault(method, {})
        for task in PHASE1_TASKS:
            results[method].setdefault(task, {})
            snap = _resolved_snapshot(method, task, BODYSLAM_MODES[0])  # any
            for mode_ in BODYSLAM_MODES:
                snap = _resolved_snapshot(method, task, mode_)
                results[method][task].setdefault(mode_, {})
                for disruption in DISRUPTIONS:
                    cell_idx += 1
                    print(f"\n# [{cell_idx}/{n_cells}] {method} / {task} / "
                          f"bs={mode_} / {disruption}")
                    with tempfile.NamedTemporaryFile(
                        suffix=".json", delete=False
                    ) as tmp:
                        tmp_out = tmp.name
                    cmd = _eval_cmd(
                        method, task, mode_, disruption, snap,
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
                            results[method][task][mode_][disruption] = json.load(f)
                    except Exception as exc:
                        print(f"# Failed to read {tmp_out}: {exc}")
                    finally:
                        try: os.remove(tmp_out)
                        except: pass

    out_path = REPO_ROOT / "phase1_obs_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n# Full JSON: {out_path}")

    _print_summary(results)
    return 0


def _print_summary(results: dict) -> None:
    """Per (method, task), average across disruptions for each mode and
    print: ep_ssm_violation_rate, ep_pfl_violation_rate, episode_success,
    and the off→oracle / off→noisy reduction. Flag the 20%-cut criterion.
    """
    print("\n" + "=" * 110)
    print("PHASE-1 E1.1 OBS-ABLATION SUMMARY  "
          "(averaged across {} disruption types)".format(len(DISRUPTIONS)))
    print("=" * 110)

    header = (
        f"{'method':<6} {'task':<25} {'mode':<8} "
        f"{'ssm_viol':>10} {'pfl_viol':>10} {'success':>9} "
        f"{'ssm_redux':>11} {'criterion':>11}"
    )
    print(header)
    print("-" * 110)

    any_pass = False
    per_method_pass: dict[str, bool] = {}

    for method in METHODS:
        method_pass = False
        for task in PHASE1_TASKS:
            cell = results.get(method, {}).get(task, {})
            avgs: dict[str, dict[str, float]] = {}
            for mode_ in BODYSLAM_MODES:
                runs = cell.get(mode_, {})
                if not runs:
                    avgs[mode_] = {}
                    continue
                ssm = [_safety_lookup(m, "ep_ssm_violation_rate") for m in runs.values()]
                pfl = [_safety_lookup(m, "ep_pfl_violation_rate") for m in runs.values()]
                succ = [m.get("episode_success", 0.0) for m in runs.values()]
                avgs[mode_] = {
                    "ssm": sum(ssm) / len(ssm),
                    "pfl": sum(pfl) / len(pfl),
                    "success": sum(succ) / len(succ),
                }

            off_ssm = avgs.get("off", {}).get("ssm", float("nan"))
            for mode_ in BODYSLAM_MODES:
                a = avgs.get(mode_, {})
                if not a:
                    print(f"{method:<6} {task:<25} {mode_:<8} "
                          f"{'-':>10} {'-':>10} {'-':>9} {'-':>11} {'-':>11}")
                    continue
                if mode_ == "off" or off_ssm == 0 or off_ssm != off_ssm:
                    redux_str = "-"
                    crit = "-"
                else:
                    redux = (off_ssm - a["ssm"]) / off_ssm
                    redux_str = f"{redux*100:+.1f}%"
                    if redux >= 0.20:
                        crit = "PASS"
                        method_pass = True
                        any_pass = True
                    else:
                        crit = ""
                print(f"{method:<6} {task:<25} {mode_:<8} "
                      f"{a['ssm']:>10.3f} {a['pfl']:>10.3f} {a['success']:>9.2f} "
                      f"{redux_str:>11} {crit:>11}")
            print("-" * 110)
        per_method_pass[method] = method_pass

    print()
    print("Success criterion: oracle ≥ 20% reduction in SSM violation rate vs "
          "`off`, on at least one of DP / ACT.")
    for m, ok in per_method_pass.items():
        print(f"  {m}: {'PASS' if ok else 'FAIL'}")
    print(f"  overall: {'PASS' if any_pass else 'FAIL — Phase 2/3 contingency triggers'}")


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
    mode.add_argument("--run", action="store_true",
                      help="Run all 90 eval cells, dump JSON, print summary table.")
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
    if args.run:
        return _run_grid(args.seed, args.num_eval_episodes)
    if args.eval:
        return _print_eval(args.seed, args.num_eval_episodes)
    return _print_train(args.seed)


if __name__ == "__main__":
    sys.exit(main())
