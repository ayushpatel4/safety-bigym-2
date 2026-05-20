#!/usr/bin/env python
"""E1.4 — CQN-AS observation ablation on saucepan_to_hob (COWORKER).

Re-runs the Phase 1.4 reward-on pilot with the CQN-AS agent in place of
DrQ-V2+. Same hypothesis: does an SSM/PFL violation penalty turn the
``human_pos_estimate`` channel from ignored to used? The previous DrQ-V2+
sweep (E1.1) showed `oracle` boosted task success 0.22 → 0.58 on
saucepan_to_hob while *worsening* SSM — strongly suggesting the actor
uses the channel for progress, not safety. CQN-AS is the actor we plan
to ship in Phase 3, so we re-run the ablation on the new architecture
before locking the Phase-3 obs config.

Scope: anchor task ``saucepan_to_hob``, COWORKER train space for
training, COWORKER eval space for evaluation, three ``bodyslam`` modes
(off / oracle / noisy) = 3 train cells. ~10–15 GPU-h total at
``num_train_frames=200000``.

Decision rule (per .claude/IMPLEMENTATION_STATUS.md C3):
  - channel helps under RL → ``bodyslam=noisy`` for Phase 3 actor
  - channel doesn't help → ``bodyslam=off`` (filter consumes channel only)
  - oracle helps, noisy doesn't → ``bodyslam=oracle`` + noise-model dig

Usage:
    # Print the 3 train commands (hand to GPU box):
    python scripts/phase1_reward_pilot_cqn_as.py --train

    # Print the eval commands once snapshots land in SNAPSHOTS dict:
    python scripts/phase1_reward_pilot_cqn_as.py --eval

    # Print all 6 (train + eval) at once:
    python scripts/phase1_reward_pilot_cqn_as.py --all

    # 2000-frame smoke for a single cell, no W&B:
    python scripts/phase1_reward_pilot_cqn_as.py --smoke --cell oracle

Depends on A6 being green (train_cqn_as.py end-to-end validated under
``disruption=coworker_train`` + ``bodyslam=oracle`` + COWORKER episode
boundaries). Smoke gate validates composition; the full sweep is
launched only after A6.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADLESS_ENV = ("MUJOCO_GL=egl", "MUJOCO_EGL_DEVICE_ID=0")

# E1.4 anchor (per IMPLEMENTATION_STATUS.md decision log, 2026-05-15):
# single task, not the 3-task grid. saucepan_to_hob is the cell where
# the legacy E1.1 finding was strongest — oracle boosted success without
# improving SSM. Most diagnostic for whether an RL gradient redirects
# the channel toward safety.
ANCHOR_TASK = "saucepan_to_hob"
TRAIN_DISRUPTION = "coworker_train"
EVAL_DISRUPTION = "coworker_eval"
BODYSLAM_MODES = ("off", "oracle", "noisy")

NUM_TRAIN_FRAMES = 200_000
NUM_EVAL_EPISODES = 20
EVAL_SEEDS = (0, 1, 2)

# Fill after training: per-cell snapshot path (peak by W&B eval/episode_reward
# curve, or final snapshot if the curve is monotone). Relative to REPO_ROOT.
SNAPSHOTS: dict[str, str | None] = {
    "off": "~/Documents/safety_bigym/exp_local/cqn_as_safety/saucepan_to_hob_20260519110325/snapshot_80000.pt",
    "oracle": "~/Documents/safety_bigym/exp_local/cqn_as_safety/saucepan_to_hob_20260519110356/snapshot_80000.pt",
    "noisy": "~/Documents/safety_bigym/exp_local/cqn_as_safety/saucepan_to_hob_20260519110409/snapshot_80000.pt",
}


def _require_amass() -> str:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        sys.stderr.write(
            "AMASS_DATA_DIR not set; export it before running:\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU\n"
        )
        sys.exit(1)
    return amass


def _resolved_snapshot(mode: str) -> Path | None:
    rel = SNAPSHOTS.get(mode)
    if rel is None:
        return None
    p = REPO_ROOT / rel
    return p if p.is_file() else None


def _train_cmd(mode: str, seed: int) -> list[str]:
    run_name = f"phase1r-cqnas-train-{ANCHOR_TASK}-bs{mode}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_cqn_as.py",
        f"env=safety_bigym/{ANCHOR_TASK}",
        f"disruption={TRAIN_DISRUPTION}",
        f"bodyslam={mode}",
        f"seed={seed}",
        f"num_train_frames={NUM_TRAIN_FRAMES}",
        "num_demos=0",
        "save_snapshot=true",
        "save_video=true",
        "env.safety.add_violation_penalty=true",
        "env.safety.violation_penalty=0.05",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-1-reward","train","cqn-as","{ANCHOR_TASK}","bs-{mode}","e1.4"]',
    ]


def _eval_cmd(mode: str, snapshot: Path, seed: int) -> list[str]:
    """Eval-only run: load snapshot, COWORKER eval space, no training.

    train_cqn_as.py's eval path runs inside the train loop; for a
    snapshot-only eval we set num_train_frames=0 and eval_every_frames=0
    so the workspace evaluates the loaded policy on the eval space and
    exits. Snapshot loading is wired via +snapshot_path (mirrors
    train_safety.py's convention; train_cqn_as.py needs to honour it —
    if it doesn't yet, this is the eval-side follow-up to A8).
    """
    run_name = f"phase1r-cqnas-eval-{ANCHOR_TASK}-bs{mode}-s{seed}"
    return [
        *HEADLESS_ENV,
        sys.executable,
        "train_cqn_as.py",
        f"env=safety_bigym/{ANCHOR_TASK}",
        f"disruption={EVAL_DISRUPTION}",
        f"bodyslam={mode}",
        f"+snapshot_path={snapshot}",
        "num_train_frames=0",
        "num_demos=0",
        f"num_eval_episodes={NUM_EVAL_EPISODES}",
        "eval_every_frames=1",
        f"seed={seed}",
        "wandb.use=true",
        f"wandb.name={run_name}",
        f'+wandb.tags=["phase-1-reward","eval","cqn-as","{ANCHOR_TASK}","bs-{mode}","{EVAL_DISRUPTION}","e1.4"]',
    ]


def _print_train(seed: int) -> int:
    print(
        f"# E1.4 CQN-AS observation ablation — {len(BODYSLAM_MODES)} train cells "
        f"(anchor task: {ANCHOR_TASK}; train disruption: {TRAIN_DISRUPTION})"
    )
    print(f"# AMASS_DATA_DIR={os.environ['AMASS_DATA_DIR']}\n")
    print(f"# cd safety_bigym && source venv/bin/activate")
    print(f"# (run sequentially or distribute by hand across GPUs)\n")
    for mode in BODYSLAM_MODES:
        print(f"# --- {ANCHOR_TASK} / bodyslam={mode} ---")
        print(" ".join(shlex.quote(c) for c in _train_cmd(mode, seed)))
        print()
    return 0


def _print_eval(num_eval_episodes: int) -> int:
    n_cells = len(BODYSLAM_MODES) * len(EVAL_SEEDS)
    print(
        f"# E1.4 CQN-AS eval — {n_cells} cells "
        f"({len(BODYSLAM_MODES)} modes × {len(EVAL_SEEDS)} seeds, "
        f"{num_eval_episodes} eps each, eval disruption: {EVAL_DISRUPTION})"
    )
    missing: list[str] = []
    for mode in BODYSLAM_MODES:
        snap = _resolved_snapshot(mode)
        if snap is None:
            missing.append(mode)
            print(
                f"# SKIP bs={mode}: no snapshot "
                f"(rel={SNAPSHOTS.get(mode)!r}). Run --train first."
            )
            continue
        print(f"\n# --- bs={mode}  ({snap.relative_to(REPO_ROOT)}) ---")
        for seed in EVAL_SEEDS:
            print(" ".join(shlex.quote(c) for c in _eval_cmd(mode, snap, seed)))
    if missing:
        print(f"\n# {len(missing)} cell(s) missing snapshots: {missing}")
        print("# Run `python scripts/phase1_reward_pilot_cqn_as.py --train` first,")
        print("# then paste peak-by-eval snapshot paths into SNAPSHOTS at the top of this script.")
        return 2
    return 0


def _smoke(mode: str, seed: int) -> int:
    """2000-frame validation. No W&B; default num_demos=0; bodyslam=mode.

    Smoke checks: composition of env_adapter + factory + COWORKER scenario
    + violation_penalty doesn't crash, and the train loop logs at least
    one episode-end safety payload. Wall-clock budget: a few minutes on
    the GPU box once A6 is green.
    """
    run_name = f"phase1r-cqnas-smoke-{ANCHOR_TASK}-bs{mode}-s{seed}"
    cmd = [
        *HEADLESS_ENV,
        sys.executable,
        "train_cqn_as.py",
        f"env=safety_bigym/{ANCHOR_TASK}",
        f"disruption={TRAIN_DISRUPTION}",
        f"bodyslam={mode}",
        f"seed={seed}",
        "num_train_frames=2000",
        "num_demos=0",
        "save_snapshot=false",
        "env.safety.add_violation_penalty=true",
        "env.safety.violation_penalty=0.05",
        "wandb.use=false",
        f"wandb.name={run_name}",
    ]
    print(">>> smoke:", " ".join(shlex.quote(c) for c in cmd))
    argv = list(cmd)
    env = os.environ.copy()
    while argv and "=" in argv[0] and not argv[0].startswith("-"):
        k, v = argv.pop(0).split("=", 1)
        env[k] = v
    return subprocess.run(argv, cwd=REPO_ROOT, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"E1.4 CQN-AS obs ablation — 3 cells × {ANCHOR_TASK} "
            f"× COWORKER (train={TRAIN_DISRUPTION}, eval={EVAL_DISRUPTION})"
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="print 3 train commands")
    mode.add_argument("--eval", action="store_true", help="print eval commands (needs SNAPSHOTS filled)")
    mode.add_argument("--all", action="store_true", help="print train + eval commands")
    mode.add_argument("--smoke", action="store_true", help="run a 2000-frame smoke for one cell")
    parser.add_argument(
        "--cell", default="oracle", choices=BODYSLAM_MODES,
        help="cell selector for --smoke (default: oracle)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-eval-episodes", type=int, default=NUM_EVAL_EPISODES,
        help=f"eval episodes per (mode, seed) cell (default: {NUM_EVAL_EPISODES})",
    )
    args = parser.parse_args()

    _require_amass()

    if args.smoke:
        return _smoke(args.cell, args.seed)
    if args.all:
        rc = _print_train(args.seed)
        print()
        rc |= _print_eval(args.num_eval_episodes)
        return rc
    if args.eval:
        return _print_eval(args.num_eval_episodes)
    return _print_train(args.seed)


if __name__ == "__main__":
    sys.exit(main())
