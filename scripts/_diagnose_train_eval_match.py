#!/usr/bin/env python
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
    print(f"\nAfter load_snapshot:")
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
    print(f"\nEval env wrapper chain:")
    for w in wrapper_chain:
        print(f"  -> {w}")
    
    # Check if RecedingHorizonControl params match
    from robobase.envs.wrappers import RecedingHorizonControl
    env = workspace.eval_env
    while env and not isinstance(env, RecedingHorizonControl):
        env = getattr(env, 'env', None)
    if env and isinstance(env, RecedingHorizonControl):
        print(f"\nRecedingHorizonControl settings:")
        print(f"  action_sequence: {env._action_sequence}")
        print(f"  max_episode_steps: {env._max_episode_steps}")
        print(f"  execution_length: {env._execution_length}")
        print(f"  temporal_ensemble: {env._temporal_ensemble}")
        print(f"  gain: {env._gain}")
    
    # Run eval
    print(f"\nRunning eval with {workspace.cfg.num_eval_episodes} episodes...")
    eval_metrics = workspace._eval()
    
    print(f"\nEval results:")
    print(f"  episode_success: {eval_metrics.get('episode_success', 'N/A')}")
    print(f"  episode_reward: {eval_metrics.get('episode_reward', 'N/A')}")
    print(f"  episode_length: {eval_metrics.get('episode_length', 'N/A')}")
    
    # Cleanup
    workspace.eval_env.close()
    workspace.train_envs.close()

if __name__ == "__main__":
    main()
