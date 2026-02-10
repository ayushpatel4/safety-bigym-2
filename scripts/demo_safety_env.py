#!/usr/bin/env python
"""
Demo: SafetyBiGymEnv with Human

Visual demo of the integrated safety environment.
Run with: mjpython scripts/demo_safety_env.py

Shows:
- BiGym robot + SMPL-H human in same scene
- Safety monitoring (SSM/PFL) in real-time
"""

import numpy as np
import mujoco.viewer
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

from bigym.action_modes import JointPositionActionMode
from safety_bigym import SafetyBiGymEnv, SafetyConfig, HumanConfig


def main():
    print("=" * 60)
    print("SafetyBiGymEnv Demo")
    print("=" * 60)
    
    # Create action mode
    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
    )
    
    # Create safety config - log violations
    safety_config = SafetyConfig(
        log_violations=True,
        terminate_on_violation=False,
    )
    
    # Create human config with AMASS motion clip
    cmu_clips_dir = "/Users/ayushpatel/Documents/FYP3/CMU/CMU"
    human_config = HumanConfig(
        motion_clip_dir=cmu_clips_dir,
        motion_clip_paths=["74/74_01_poses.npz"],  # Walking motion
    )
    
    # Create environment
    print("Creating SafetyBiGymEnv...")
    env = SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=safety_config,
        human_config=human_config,
        inject_human=True,
    )
    
    print(f"✅ Environment created: {env.task_name}")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Human pelvis ID: {env._human_pelvis_id}")
    
    # Reset
    obs, info = env.reset()
    print(f"✅ Environment reset")
    print(f"   Scenario: {info.get('scenario', {})}")
    
    # Get model and data for viewer
    model = env._mojo.model
    data = env._mojo.data
    
    print("\nOpening viewer...")
    print("Press ESC to close")
    print("-" * 60)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            # Take random action
            action = np.zeros(env.action_space.shape)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get safety info
            safety = info.get("safety", {})
            
            # Print status every 100 steps
            if step % 100 == 0:
                human_pos = data.xpos[env._human_pelvis_id] if env._human_pelvis_id else [0,0,0]
                sep = safety.get('min_separation', float('inf'))
                sep_str = f"{sep:.2f}m" if sep != float('inf') else "inf"
                print(f"Step {step:4d} | "
                      f"Sep: {sep_str} | "
                      f"SSM: {'⚠️' if safety.get('ssm_violation') else '✓'} | "
                      f"PFL: {'⚠️' if safety.get('pfl_violation') else '✓'} | "
                      f"Force: {safety.get('max_contact_force', 0):.1f}N | "
                      f"Human: ({human_pos[0]:.2f}, {human_pos[1]:.2f}, {human_pos[2]:.2f})")
            
            # Check termination
            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, info = env.reset()
                step = 0
            
            # Sync viewer
            viewer.sync()
            step += 1
    
    print("\nViewer closed. Cleaning up...")
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
