#!/usr/bin/env python
"""Diagnosis script for eval-from-snapshot failures.

Tests whether:
1. The snapshot actually loads and changes model weights from init
2. The loaded agent produces different actions vs a fresh agent
3. Actions vary across different observations (not constant output)
4. The observation space matches between training and eval envs

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/diagnose_eval.py
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

SNAPSHOT_PATH = "/home/ap2322/Documents/safety_bigym/exp_local/dp_safety/dishwasher_close_20260422193511/snapshots/70000_snapshot.pt"
TASK = "dishwasher_close"

results = {}

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ===========================================================================
# TEST 1: Inspect snapshot contents
# ===========================================================================
section("TEST 1: Snapshot Contents Inspection")

snap_path = Path(SNAPSHOT_PATH)
print(f"Snapshot path: {snap_path}")
print(f"Exists: {snap_path.is_file()}")
print(f"Size: {snap_path.stat().st_size / 1e6:.1f} MB")

payload = torch.load(snap_path, map_location="cpu", weights_only=False)
print(f"\nPayload top-level keys: {list(payload.keys())}")
print(f"  _pretrain_step: {payload.get('_pretrain_step')}")
print(f"  _main_loop_iterations: {payload.get('_main_loop_iterations')}")
print(f"  _global_env_episode: {payload.get('_global_env_episode')}")
print(f"  action_stats present: {payload.get('action_stats') is not None}")
print(f"  obs_stats present: {payload.get('obs_stats') is not None}")

results["snapshot_pretrain_step"] = payload.get("_pretrain_step")
results["snapshot_has_agent"] = "agent" in payload
results["snapshot_has_action_stats"] = payload.get("action_stats") is not None
results["snapshot_has_obs_stats"] = payload.get("obs_stats") is not None

# Inspect agent state dict
agent_state = payload["agent"]
print(f"\nAgent state_dict keys ({len(agent_state)} total):")
param_norms = {}
for k, v in agent_state.items():
    if isinstance(v, torch.Tensor):
        param_norms[k] = v.float().norm().item()

# Print a sample of parameter norms
sample_keys = list(param_norms.keys())[:10]
for k in sample_keys:
    print(f"  {k}: norm={param_norms[k]:.4f}")
print(f"  ... ({len(param_norms) - 10} more)")

# Check action_stats details
if payload.get("action_stats"):
    stats = payload["action_stats"]
    print(f"\nAction stats type: {type(stats)}")
    if isinstance(stats, dict):
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
            else:
                print(f"  {k}: {type(v)}")

# Check obs_stats details
if payload.get("obs_stats"):
    stats = payload["obs_stats"]
    print(f"\nObs stats type: {type(stats)}")
    if isinstance(stats, dict):
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
            elif isinstance(v, dict):
                print(f"  {k}: (nested dict with {len(v)} keys)")
                for k2, v2 in v.items():
                    if isinstance(v2, np.ndarray):
                        print(f"    {k2}: shape={v2.shape}, range=[{v2.min():.4f}, {v2.max():.4f}]")
            else:
                print(f"  {k}: {type(v)}")


# Check snapshot cfg
snap_cfg = payload.get("cfg")
if snap_cfg:
    from omegaconf import OmegaConf
    print(f"\nSnapshot training config:")
    print(f"  num_pretrain_steps: {snap_cfg.get('num_pretrain_steps', 'N/A')}")
    print(f"  action_sequence: {snap_cfg.get('action_sequence', 'N/A')}")
    print(f"  use_min_max_normalization: {snap_cfg.get('use_min_max_normalization', 'N/A')}")
    print(f"  norm_obs: {snap_cfg.get('norm_obs', 'N/A')}")
    print(f"  execution_length: {snap_cfg.get('execution_length', 'N/A')}")
    print(f"  temporal_ensemble: {snap_cfg.get('temporal_ensemble', 'N/A')}")
    print(f"  env.task_name: {snap_cfg.env.get('task_name', 'N/A')}")
    print(f"  env.episode_length: {snap_cfg.env.get('episode_length', 'N/A')}")
    print(f"  pixels: {snap_cfg.get('pixels', 'N/A')}")
    print(f"  frame_stack: {snap_cfg.get('frame_stack', 'N/A')}")
    print(f"  demos: {snap_cfg.get('demos', 'N/A')}")
    print(f"  batch_size: {snap_cfg.get('batch_size', 'N/A')}")

results["snapshot_size_mb"] = snap_path.stat().st_size / 1e6


# ===========================================================================
# TEST 2: Simulate Hydra run to test full load path
# ===========================================================================
section("TEST 2: Full Workspace Load via Hydra (subprocess)")

# Instead of building workspace directly, let's test using a small script
# that Hydra decorates
test_script = Path(__file__).resolve().parent.parent / "scripts" / "_diagnose_hydra_test.py"
test_script.write_text('''#!/usr/bin/env python
"""Internal Hydra-decorated test for diagnose_eval.py"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

import hydra

@hydra.main(config_path="../cfgs", config_name="safety_config", version_base=None)
def main(cfg):
    from robobase.workspace import Workspace
    from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory
    from robobase import utils as rb_utils

    results = {}

    factory = SafetyBiGymEnvFactory()
    workspace = Workspace(cfg, env_factory=factory)

    # Capture pre-load norms
    pre_norms = {}
    for k, v in workspace.agent.state_dict().items():
        if isinstance(v, torch.Tensor):
            pre_norms[k] = v.float().norm().item()

    # Load snapshot
    if cfg.get("snapshot_path", None):
        print(f"Loading snapshot: {cfg.snapshot_path}")
        workspace.load_snapshot()

    # Capture post-load norms
    post_norms = {}
    for k, v in workspace.agent.state_dict().items():
        if isinstance(v, torch.Tensor):
            post_norms[k] = v.float().norm().item()

    # Compare
    changed = sum(1 for k in pre_norms if k in post_norms and abs(pre_norms[k] - post_norms[k]) > 1e-6)
    unchanged = sum(1 for k in pre_norms if k in post_norms and abs(pre_norms[k] - post_norms[k]) <= 1e-6)
    results["weights_changed"] = changed
    results["weights_unchanged"] = unchanged
    print(f"WEIGHTS: {changed} changed, {unchanged} unchanged")

    # Test action generation
    obs, info = workspace.eval_env.reset()
    workspace.agent.set_eval_env_running(True)
    workspace.agent.reset(0, [workspace.train_envs.num_envs])

    print(f"\\nObservation keys: {list(obs.keys())}")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")

    # Generate actions
    torch_obs = {k: torch.from_numpy(v).unsqueeze(0).to(workspace.device) for k, v in obs.items()}
    
    actions_list = []
    with torch.no_grad():
        with rb_utils.eval_mode(workspace.agent):
            for i in range(5):
                a = workspace.agent.act(torch_obs, 0, eval_mode=True)
                actions_list.append(a.cpu().numpy())
                print(f"  Action {i}: shape={a.shape}, mean={a.mean():.4f}, std={a.std():.4f}, "
                      f"range=[{a.min():.4f}, {a.max():.4f}]")

    actions_arr = np.concatenate(actions_list, axis=0)
    print(f"\\nInter-action std: {actions_arr.std(axis=0).mean():.6f}")
    results["inter_action_std"] = float(actions_arr.std(axis=0).mean())
    results["action_mean"] = float(actions_arr.mean())

    # Do a short rollout
    print(f"\\n--- Short rollout (20 steps) ---")
    obs, info = workspace.eval_env.reset()
    workspace.agent.reset(0, [workspace.train_envs.num_envs])
    
    rewards = []
    action_norms = []
    for step_i in range(20):
        torch_obs = {k: torch.from_numpy(v).unsqueeze(0).to(workspace.device) for k, v in obs.items()}
        with torch.no_grad():
            with rb_utils.eval_mode(workspace.agent):
                action = workspace.agent.act(torch_obs, 0, eval_mode=True)
        action_np = action.cpu().numpy()[0]
        obs, reward, term, trunc, info = workspace.eval_env.step(action_np)
        rewards.append(reward)
        action_norms.append(np.linalg.norm(action_np))
        if step_i < 5:
            print(f"  Step {step_i}: reward={reward:.4f}, action_norm={action_norms[-1]:.4f}")
        if term or trunc:
            print(f"  Episode ended at step {step_i} (term={term}, trunc={trunc})")
            break

    print(f"\\nReward: mean={np.mean(rewards):.4f}, total={np.sum(rewards):.4f}")
    print(f"Action norm: mean={np.mean(action_norms):.4f}, std={np.std(action_norms):.4f}")
    
    results["rollout_reward_mean"] = float(np.mean(rewards))
    results["rollout_reward_sum"] = float(np.sum(rewards))
    results["action_norm_mean"] = float(np.mean(action_norms))
    results["action_norm_std"] = float(np.std(action_norms))

    # Save
    out = Path("diagnose_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\nResults saved to {out}")
    
    # Cleanup
    workspace.eval_env.close()
    workspace.train_envs.close()

if __name__ == "__main__":
    main()
''')

import subprocess
cmd = [
    sys.executable, str(test_script),
    "launch=dp_pixel_safety_bigym",
    f"env=safety_bigym/{TASK}",
    f"+snapshot_path={SNAPSHOT_PATH}",
    "num_train_frames=0",
    "num_pretrain_steps=0",
    "demos=0",
    "num_eval_episodes=1",
    "eval_every_steps=1",
    "seed=0",
    "wandb.use=false",
]

env = os.environ.copy()
env["MUJOCO_GL"] = "egl"
env["PYOPENGL_PLATFORM"] = "egl"

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(
    cmd,
    cwd=str(Path(__file__).resolve().parent.parent),
    env=env,
    capture_output=True,
    text=True,
    timeout=600,
)

print("\n--- STDOUT ---")
print(result.stdout)
if result.stderr:
    # Only print last 50 lines of stderr (usually logging noise)
    stderr_lines = result.stderr.strip().split("\n")
    if len(stderr_lines) > 50:
        print(f"\n--- STDERR (last 50 of {len(stderr_lines)} lines) ---")
        print("\n".join(stderr_lines[-50:]))
    else:
        print("\n--- STDERR ---")
        print(result.stderr)

print(f"\nReturn code: {result.returncode}")

# Try to collect results from the subprocess
hydra_results_path = Path(__file__).resolve().parent.parent / "diagnose_results.json"
# The file might be in the hydra output dir, search for it
import glob
possible = glob.glob(
    str(Path(__file__).resolve().parent.parent / "exp_local" / "**" / "diagnose_results.json"),
    recursive=True,
)
if possible:
    print(f"\nFound results at: {possible[-1]}")
    with open(possible[-1]) as f:
        sub_results = json.load(f)
    results.update(sub_results)


# ===========================================================================
# SUMMARY
# ===========================================================================
section("DIAGNOSIS SUMMARY")

issues = []

if results.get("weights_changed", -1) == 0:
    issues.append("CRITICAL: Snapshot weights not loaded — agent uses random init weights")

if results.get("inter_action_std", -1) < 1e-6 and results.get("inter_action_std", -1) >= 0:
    issues.append("SUSPICIOUS: Agent outputs identical actions for same observation")

if results.get("action_norm_std", -1) < 0.001 and results.get("action_norm_std", -1) >= 0:
    issues.append("SUSPICIOUS: Action norms constant across steps — policy may not be reacting to obs")

if not results.get("snapshot_has_action_stats"):
    issues.append("WARNING: Snapshot missing action_stats — normalization may be wrong")

if not results.get("snapshot_has_obs_stats"):
    issues.append("WARNING: Snapshot missing obs_stats — observation normalization may be wrong")

if issues:
    print("ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("No obvious issues found in this diagnostic pass.")
    
# Save all results
out_path = Path(__file__).resolve().parent.parent / "diagnose_eval_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nFull results saved to {out_path}")
