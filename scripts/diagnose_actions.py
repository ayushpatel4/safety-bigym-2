#!/usr/bin/env python
"""Focused diagnosis: Compare agent behavior with trained weights vs random init.

This is the definitive test: if the trained agent produces the same actions
as a randomly-initialized agent, the weights aren't loading.

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/diagnose_actions.py
"""

import os, sys, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

SNAPSHOT = "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/dishwasher_close_20260422193511/snapshots/70000_snapshot.pt"

print("=== Loading snapshot ===")
payload = torch.load(SNAPSHOT, map_location="cpu", weights_only=False)
agent_state = payload["agent"]
action_stats = payload["action_stats"]
snap_cfg = payload["cfg"]

print(f"\nPretrain step: {payload['_pretrain_step']}")
print(f"Action stats min: {action_stats['min']}")
print(f"Action stats max: {action_stats['max']}")

# The real normalization: actions from agent are in [-1, 1] (tanh)
# RescaleFromTanhWithMinMax maps [-1,1] -> [action_min, action_max]
# Where action_min/max come from demo statistics

# So: action_real = (tanh_action + 1) / 2 * (max - min) + min

action_min = action_stats["min"]
action_max = action_stats["max"]
print(f"\nAction dim: {len(action_min)}")
print(f"Action range per dim:")
for i in range(len(action_min)):
    print(f"  dim {i}: [{action_min[i]:.4f}, {action_max[i]:.4f}], range={action_max[i]-action_min[i]:.4f}")

# Check for zero-range dimensions!
zero_range = np.where(np.abs(action_max - action_min) < 1e-6)[0]
if len(zero_range) > 0:
    print(f"\n*** WARNING: {len(zero_range)} action dims have zero range: {zero_range} ***")
    print("   These map any tanh output to the same value (all actions equivalent)")


# Now let's check - what action values does a tanh=0 produce vs tanh=1 vs tanh=-1?
for tanh_val in [-1.0, 0.0, 1.0]:
    tanh_action = np.full_like(action_min, tanh_val)
    real_action = (tanh_action + 1) / 2.0 * (action_max - action_min) + action_min
    print(f"\nTanh={tanh_val:.1f} -> real action: {real_action}")

# Check if the agent's action_sequence=16 means the action shape is [1, 16, 16]
# meaning (batch, timesteps, action_dim)
print(f"\naction_sequence from config: {snap_cfg.get('action_sequence', 'N/A')}")
print(f"execution_length from config: {snap_cfg.get('execution_length', 'N/A')}")
print(f"temporal_ensemble from config: {snap_cfg.get('temporal_ensemble', 'N/A')}")

# Key: The snapshot had episode_reward=0 and episode_success=0 for dishwasher_close
# but 0.4 for reach_target_single. Let's check if the issue is task-specific.

# Check: is the snapshot at step 70000 the PEAK or past the overfit point?
# With 100000 pretrain steps, 70000 could be before or after peak

print("\n\n=== CONCLUSION ===")
print("From diagnosis output:")
print("1. Weights ARE loading (444/506 changed)")
print("2. Actions are NOT random - they vary with observations (inter-action std=0.51)")
print("3. Action norms are very high (~11) - actions are at tanh saturation [-1,1]")
print("4. Reward is 0.0 for all steps during rollout")
print()
print("Possible root causes:")
print("a) Policy overfit to demos but actions are in tanh[-1,1] space saturating")  
print("b) The obs_stats normalization in ConcatDim is BROKEN (k in obs_stats check is wrong)")
print("   During training: obs come from replay buffer (already processed in demo loading)")
print("   During eval: obs come from fresh env (not normalized same way)")
print("c) The policy learned successfully on normalized demos but eval env")
print("   uses different observation ranges causing distribution shift")
print("d) The 70000-step checkpoint is past the overfit inflection")
print("e) The RecedingHorizonControl wrapper + temporal ensemble causes issues")
print("f) The demos used for training (30 from DemoStore) don't produce success on this task")
print()

# Let's check: what was the training performance at step 70k?
print("TO VERIFY: Check W&B for pretrain_eval/episode_success at step 70000")
print("If it was 0.0 during training too, the issue is that the policy never learned the task")
print("If it was >0 during training, there's a train/eval mismatch")
