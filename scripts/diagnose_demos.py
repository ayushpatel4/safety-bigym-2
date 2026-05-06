#!/usr/bin/env python
"""Test eval WITH demos loaded to check if stats-from-snapshot is the issue."""

import os, sys, json, subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("AMASS_DATA_DIR", os.environ.get("AMASS_DATA_DIR", "/home/ap2322/Documents/datasets/AMASS/CMU/CMU"))

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK = "dishwasher_close"
SNAP = str(REPO_ROOT / "exp_local/dp_safety/dishwasher_close_20260422193511/snapshots/90000_snapshot.pt")

# Test 1: demos=0 (current failing config)
# Test 2: demos=30 (match training config - compute stats from demos)
# Test 3: demos=0, skip cfg restore via a patched load (test cfg overwrite theory)

tests = {
    "demos_0_current": ["demos=0"],
    "demos_30_match_train": [],  # default demos=50 from env config, override to 30
}

for label, extra_args in tests.items():
    print(f"\n{'='*60}")
    print(f"  Test: {label}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, str(REPO_ROOT / "train_safety.py"),
        "launch=dp_pixel_safety_bigym",
        f"env=safety_bigym/{TASK}",
        f"+snapshot_path={SNAP}",
        "num_train_frames=0",
        "num_pretrain_steps=0",
        "num_eval_episodes=5",
        "eval_every_steps=1",
        "seed=0",
        "wandb.use=false",
        f"+eval_output_path=/tmp/eval_{label}.json",
    ] + extra_args
    
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    
    print(f"  CMD: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=900)
    
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-15:]:
            print(f"  {line}")
        continue
    
    try:
        with open(f"/tmp/eval_{label}.json") as f:
            metrics = json.load(f)
        print(f"  episode_success: {metrics.get('episode_success', 'N/A')}")
        print(f"  episode_reward:  {metrics.get('episode_reward', 'N/A')}")
        print(f"  episode_length:  {metrics.get('episode_length', 'N/A')}")
    except Exception as e:
        print(f"  Could not read results: {e}")

print("\n\nIf demos_30 succeeds but demos_0 fails: stats-from-snapshot is wrong")
print("If both fail: the issue is elsewhere (env construction, wrapper chain, etc.)")
