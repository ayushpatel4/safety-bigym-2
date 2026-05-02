#!/usr/bin/env python
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

    print(f"\nObservation keys: {list(obs.keys())}")
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
    print(f"\nInter-action std: {actions_arr.std(axis=0).mean():.6f}")
    results["inter_action_std"] = float(actions_arr.std(axis=0).mean())
    results["action_mean"] = float(actions_arr.mean())

    # Do a short rollout
    print(f"\n--- Short rollout (20 steps) ---")
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

    print(f"\nReward: mean={np.mean(rewards):.4f}, total={np.sum(rewards):.4f}")
    print(f"Action norm: mean={np.mean(action_norms):.4f}, std={np.std(action_norms):.4f}")
    
    results["rollout_reward_mean"] = float(np.mean(rewards))
    results["rollout_reward_sum"] = float(np.sum(rewards))
    results["action_norm_mean"] = float(np.mean(action_norms))
    results["action_norm_std"] = float(np.std(action_norms))

    # Save
    out = Path("diagnose_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    
    # Cleanup
    workspace.eval_env.close()
    workspace.train_envs.close()

if __name__ == "__main__":
    main()
