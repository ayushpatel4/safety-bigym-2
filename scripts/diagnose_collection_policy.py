#!/usr/bin/env python3
"""Diagnose WHY the SVF collection-path policy runs full-length while the same
snapshot succeeds (~449 steps) in the benchmark/deployment path.

Context (2026-06-01): after fixing the de-norm, execution-mode, and episode-length
mismatches, the collection-path sweep STILL diverges from deployment — episodes
run the full max_steps (never terminate) and filterless proximity is 0.0516 vs the
benchmark's 0.296. Two hypotheses:

  H1 (degraded policy): the collection env/obs/de-norm differs enough that the
      policy produces poor actions and never completes the task. Episode reward
      will be far below deployment's (row-1 mean reward -4.27). -> needs the UNIFY
      fix (collect through the benchmark adapter+runner).
  H2 (termination wiring): the policy works fine but the collection env doesn't
      fire `terminated` on success, so episodes run to max_steps. Episode reward
      will be ~ deployment's (-4.27). -> a smaller fix (wire success-termination).

This rolls out the COLLECTION path (svf_collect._build_live_env + load_snapshot_policy
— the exact path the SVF dataset is built from) and logs the decisive signals:
episode reward, length, whether `terminated` fired, min-separation, proximity rate,
and an action-saturation check. Compare the mean reward to the deployment reference
printed at the end.

Usage (GPU box):
  export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 AMASS_DATA_DIR=~/Documents/CMU/CMU
  python scripts/diagnose_collection_policy.py \
      --snapshot exp_local/.../stage2_full/snapshot_28203.pt \
      --episodes 5 --max-steps 1000
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

# Import the collection module by its real name (scripts/ on sys.path) so its
# @dataclass definitions resolve normally (a path-loaded alias breaks them).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import svf_collect_dataset as C  # noqa: E402

# Deployment reference — E4.1 row-1 (policy-only, noisy, coworker_train, g1).
_DEPLOY = dict(reward=-4.27, ep_len=449.0, success=0.85, proximity=0.296)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=pathlib.Path)
    ap.add_argument("--task", default="saucepan_to_hob")
    ap.add_argument("--disruption", default="coworker_train")
    ap.add_argument("--human-model", default="g1")
    ap.add_argument("--bodyslam-mode", default="noisy", choices=["oracle", "noisy", "off"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cameras, resolution = C.peek_snapshot_cameras(args.snapshot)
    snap_bs = C.peek_snapshot_bodyslam_mode(args.snapshot)
    print(f"# snapshot trained bodyslam={snap_bs}, cameras={list(cameras) or 'none'}")
    print(f"# eval: task={args.task} disruption={args.disruption} "
          f"bodyslam={args.bodyslam_mode} human={args.human_model} "
          f"max_steps={args.max_steps}\n")

    env = C._build_live_env(
        args.task, args.disruption, args.bodyslam_mode, motion_clips=[],
        cameras=cameras, camera_resolution=resolution, human_model=args.human_model,
    )
    policy = C.load_snapshot_policy(args.snapshot, env, rollout_max_steps=args.max_steps)

    rewards, lengths, terms, min_seps, prox_rates = [], [], [], [], []
    act_abs_max = 0.0
    print(f"{'ep':>3} {'len':>5} {'term':>5} {'term_step':>9} {'reward':>9} "
          f"{'max_r':>7} {'min_sep':>8} {'prox_rate':>9}")
    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        policy.reset()
        total_r, max_r, min_sep, prox = 0.0, -math.inf, math.inf, 0
        term_step, terminated, steps = None, False, 0
        for t in range(args.max_steps):
            action = policy(obs)
            if ep == 0:
                act_abs_max = max(act_abs_max, float(np.max(np.abs(action))))
            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            max_r = max(max_r, float(r))
            safety = info.get("safety", {}) if isinstance(info, dict) else {}
            sep = safety.get("min_separation", math.inf)
            if sep is not None and sep < min_sep:
                min_sep = float(sep)
            if safety.get("proximity_violation"):
                prox += 1
            steps = t + 1
            if terminated and term_step is None:
                term_step = t
            if terminated or truncated:
                break
        rewards.append(total_r)
        lengths.append(steps)
        terms.append(bool(terminated))
        min_seps.append(min_sep)
        prox_rates.append(prox / max(steps, 1))
        print(f"{ep:>3} {steps:>5} {str(bool(terminated)):>5} "
              f"{str(term_step):>9} {total_r:>9.2f} {max_r:>7.2f} "
              f"{min_sep:>8.3f} {prox/max(steps,1):>9.3f}")

    mean_r = float(np.mean(rewards))
    succ = float(np.mean([1.0 if t else 0.0 for t in terms]))
    print("\n=== COLLECTION PATH (svf_collect._build_live_env + snapshot policy) ===")
    print(f"  mean episode reward : {mean_r:8.2f}")
    print(f"  mean episode length : {np.mean(lengths):8.1f}")
    print(f"  success (terminated): {succ:8.2f}")
    print(f"  mean proximity rate : {np.mean(prox_rates):8.3f}")
    print(f"  max |action| (ep0)  : {act_abs_max:8.3f}   (>> demo range => saturated/garbage)")
    print("\n=== DEPLOYMENT REFERENCE (E4.1 row-1, benchmark path) ===")
    print(f"  mean episode reward : {_DEPLOY['reward']:8.2f}")
    print(f"  mean episode length : {_DEPLOY['ep_len']:8.1f}")
    print(f"  success rate        : {_DEPLOY['success']:8.2f}")
    print(f"  mean proximity rate : {_DEPLOY['proximity']:8.3f}")

    print("\n=== VERDICT ===")
    gap = mean_r - _DEPLOY["reward"]
    if mean_r <= _DEPLOY["reward"] - 5.0:
        print(f"  H1 (DEGRADED POLICY): collection reward {mean_r:.1f} << deployment "
              f"{_DEPLOY['reward']:.1f} (gap {gap:.1f}). The policy mis-behaves in the")
        print("  collection env -> obs/de-norm divergence. FIX: UNIFY (collect via the")
        print("  benchmark adapter+runner so the policy behaves identically).")
    elif succ >= 0.5 or abs(gap) < 3.0:
        print(f"  H2 (TERMINATION/ENV WIRING): collection reward {mean_r:.1f} ~ deployment "
              f"{_DEPLOY['reward']:.1f} and/or success fires. The policy WORKS in the")
        print("  collection env; the full-length runs were a termination artifact. FIX is")
        print("  smaller (wire success-termination / match the eval horizon).")
    else:
        print(f"  AMBIGUOUS: collection reward {mean_r:.1f} vs deployment "
              f"{_DEPLOY['reward']:.1f} (gap {gap:.1f}), success {succ:.2f}. Inspect the")
        print("  per-episode rows above (does reward climb? does min_sep get close?).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
