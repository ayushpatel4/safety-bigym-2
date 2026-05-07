#!/usr/bin/env python
"""Quick smoke test: compare eval with 70k vs 90k snapshot."""

import os, sys, json
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("AMASS_DATA_DIR", os.environ.get("AMASS_DATA_DIR", "/home/ap2322/Documents/datasets/AMASS/CMU/CMU"))

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK = "dishwasher_close"

snapshots = {
    "70k": str(REPO_ROOT / "exp_local/dp_safety/dishwasher_close_20260422193511/snapshots/70000_snapshot.pt"),
    "90k": str(REPO_ROOT / "exp_local/dp_safety/dishwasher_close_20260422193511/snapshots/90000_snapshot.pt"),
}

for label, snap_path in snapshots.items():
    print(f"\n{'='*60}")
    print(f"  Testing {label} snapshot: {Path(snap_path).name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, str(REPO_ROOT / "train_safety.py"),
        "launch=dp_pixel_safety_bigym",
        f"env=safety_bigym/{TASK}",
        f"+snapshot_path={snap_path}",
        "num_train_frames=0",
        "num_pretrain_steps=0",
        "demos=0",
        "num_eval_episodes=5",
        "eval_every_steps=1",
        "seed=0",
        "wandb.use=false",
        f"+eval_output_path=/tmp/eval_{label}.json",
    ]
    
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        # Print last 10 lines of stderr
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-10:]:
            print(f"  {line}")
        continue
    
    try:
        with open(f"/tmp/eval_{label}.json") as f:
            metrics = json.load(f)
        
        success = metrics.get("episode_success", "N/A")
        reward = metrics.get("episode_reward", "N/A")
        ep_len = metrics.get("episode_length", "N/A")
        print(f"  episode_success: {success}")
        print(f"  episode_reward: {reward}")
        print(f"  episode_length: {ep_len}")
    except Exception as e:
        print(f"  Could not read results: {e}")

print("\n\nIf both show 0.0 success, the issue is NOT checkpoint selection.")
print("If 90k shows >0, the issue IS checkpoint selection (use peak checkpoint).")
