#!/usr/bin/env python
"""Decisive test: run a TRAINING run but with snapshot loaded,
to replicate the EXACT training eval path that showed 0.8 success.

The key insight: during training, _eval() is called from _pretrain_on_demos().
Let's replicate that by doing a tiny pretrain (1 step) after loading snapshot,
then evaluating.
"""

import os, sys, json, subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("AMASS_DATA_DIR", os.environ.get("AMASS_DATA_DIR", "/home/ap2322/Documents/datasets/AMASS/CMU/CMU"))

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK = "dishwasher_close"
SNAP = str(REPO_ROOT / "exp_local/dp_safety/dishwasher_close_20260422193511/snapshots/90000_snapshot.pt")

# Test: Run "training" with snapshot loaded BUT ALSO load demos 
# and do a tiny N pretrain steps, then eval like training does.
# This should reproduce the 0.8 success from training.
# num_pretrain_steps=10 means: load demos, do 10 train steps, then eval at step 10
# BUT with snapshot loaded, the agent starts from learned weights
# eval_every_steps=1 means eval at every pretrain step

# Actually, the simplest test: DON'T use snapshot_path at all.
# Just re-do the full training with 100k steps and snapshot save.
# But that takes hours.

# Better: use a custom script that loads the snapshot manually
# and runs eval in the EXACT same way as training does.

test_script = REPO_ROOT / "scripts" / "_diagnose_train_eval_match.py"
test_script.write_text('''#!/usr/bin/env python
"""Hydra-decorated test that loads snapshot and runs eval 
in the exact same way as the pretrain loop does."""
import sys, json
import numpy as np
import torch
from pathlib import Path

import hydra

@hydra.main(config_path="../cfgs", config_name="safety_config", version_base=None)
def main(cfg):
    from robobase.workspace import Workspace
    from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory

    factory = SafetyBiGymEnvFactory()
    workspace = Workspace(cfg, env_factory=factory)

    snap_path = cfg.get("snapshot_path", None)
    if snap_path:
        print(f"Loading snapshot: {snap_path}")
        workspace.load_snapshot()

    # Print the cfg AFTER load_snapshot to see if it was overwritten
    print(f"\\nAfter load_snapshot:")
    print(f"  workspace.cfg.num_eval_episodes = {workspace.cfg.num_eval_episodes}")
    print(f"  workspace.cfg.num_pretrain_steps = {workspace.cfg.get('num_pretrain_steps', 'N/A')}")
    print(f"  workspace.cfg.num_train_frames = {workspace.cfg.num_train_frames}")
    print(f"  workspace.cfg.demos = {workspace.cfg.demos}")
    print(f"  workspace.cfg.action_sequence = {workspace.cfg.action_sequence}")
    print(f"  workspace.cfg.log_eval_video = {workspace.cfg.log_eval_video}")
    print(f"  workspace._pretrain_step = {workspace._pretrain_step}")
    print(f"  workspace._main_loop_iterations = {workspace._main_loop_iterations}")
    print(f"  workspace.global_env_steps = {workspace.global_env_steps}")

    # Now the KEY question: what does _eval() see?
    # Let's check the wrappers on the eval env
    env = workspace.eval_env
    wrapper_chain = []
    while hasattr(env, 'env'):
        wrapper_chain.append(type(env).__name__)
        env = env.env
    wrapper_chain.append(type(env).__name__)
    print(f"\\nEval env wrapper chain:")
    for w in wrapper_chain:
        print(f"  -> {w}")
    
    # Check if RecedingHorizonControl params match
    from robobase.envs.wrappers import RecedingHorizonControl
    env = workspace.eval_env
    while env and not isinstance(env, RecedingHorizonControl):
        env = getattr(env, 'env', None)
    if env and isinstance(env, RecedingHorizonControl):
        print(f"\\nRecedingHorizonControl settings:")
        print(f"  action_sequence: {env._action_sequence}")
        print(f"  max_episode_steps: {env._max_episode_steps}")
        print(f"  execution_length: {env._execution_length}")
        print(f"  temporal_ensemble: {env._temporal_ensemble}")
        print(f"  gain: {env._gain}")
    
    # Run eval
    print(f"\\nRunning eval with {workspace.cfg.num_eval_episodes} episodes...")
    eval_metrics = workspace._eval()
    
    print(f"\\nEval results:")
    print(f"  episode_success: {eval_metrics.get('episode_success', 'N/A')}")
    print(f"  episode_reward: {eval_metrics.get('episode_reward', 'N/A')}")
    print(f"  episode_length: {eval_metrics.get('episode_length', 'N/A')}")
    
    # Cleanup
    workspace.eval_env.close()
    workspace.train_envs.close()

if __name__ == "__main__":
    main()
''')

# Run test with demos loaded (matching training)
cmd = [
    sys.executable, str(test_script),
    "launch=dp_pixel_safety_bigym",
    f"env=safety_bigym/{TASK}",
    f"+snapshot_path={SNAP}",
    "num_train_frames=0",
    "num_pretrain_steps=0",
    "num_eval_episodes=5",
    "eval_every_steps=1",
    "seed=0",
    "wandb.use=false",
    "demos=0",
]

env = os.environ.copy()
env["MUJOCO_GL"] = "egl"
env["PYOPENGL_PLATFORM"] = "egl"

print("Running train/eval match diagnostic...")
result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)

print("\n--- STDOUT ---")
print(result.stdout)
if result.returncode != 0:
    stderr_lines = result.stderr.strip().split("\n")
    print(f"\n--- STDERR (last 20 lines) ---")
    for line in stderr_lines[-20:]:
        print(line)
print(f"\nReturn code: {result.returncode}")
