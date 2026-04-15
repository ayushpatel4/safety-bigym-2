#!/usr/bin/env python
"""
Demo: SafetyBiGymEnv with Human

Visual demo of the integrated safety environment.
Run with: mjpython scripts/demo_safety_env.py

Shows:
- BiGym robot + SMPL-H human in same scene
- Safety monitoring (SSM/PFL) in real-time
"""

import os
import numpy as np
import mujoco.viewer
import logging
import argparse

# Enable logging
logging.basicConfig(level=logging.INFO)

from bigym.action_modes import JointPositionActionMode
from bigym.bigym_env import BiGymEnv
from safety_bigym import SafetyConfig, HumanConfig, make_safety_env

# Available tasks (import on demand)
TASK_MAP = {
    "default": "bigym.bigym_env:BiGymEnv",
    "reach": "bigym.envs.reach_target:ReachTargetSingle",
    "reach_dual": "bigym.envs.reach_target:ReachTargetDual",
    "pick_box": "bigym.envs.pick_and_place:PickBox",
    "saucepan": "bigym.envs.pick_and_place:SaucepanToHob",
    "flip_cup": "bigym.envs.manipulation:FlipCup",
    "stack_blocks": "bigym.envs.manipulation:StackBlocks",
    "dishwasher_open": "bigym.envs.dishwasher:DishwasherOpen",
    "dishwasher_close": "bigym.envs.dishwasher:DishwasherClose",
    "cupboard_open": "bigym.envs.cupboards:CupboardsOpenAll",
    "drawer_open": "bigym.envs.cupboards:DrawerTopOpen",
    "move_plate": "bigym.envs.move_plates:MovePlate",
    "groceries": "bigym.envs.groceries:GroceriesStoreLower",
    "take_cups": "bigym.envs.pick_and_place:TakeCups",
    "put_cups": "bigym.envs.pick_and_place:PutCups",
}


def load_task_cls(task_key: str) -> type:
    """Load a task class by key from TASK_MAP."""
    if task_key not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_key}. Choose from: {list(TASK_MAP.keys())}")
    module_path, cls_name = TASK_MAP[task_key].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def main():
    parser = argparse.ArgumentParser(description="SafetyBiGymEnv Demo")
    parser.add_argument(
        "--task", default="default", choices=list(TASK_MAP.keys()),
        help="BiGym task to run (default: %(default)s)",
    )
    args = parser.parse_args()

    task_cls = load_task_cls(args.task)

    print("=" * 60)
    print(f"SafetyBiGymEnv Demo — Task: {task_cls.__name__}")
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
    cmu_clips_dir = os.environ.get("AMASS_DATA_DIR", "/home/ap2322/Documents/CMU/CMU")
    human_config = HumanConfig(
        motion_clip_dir=cmu_clips_dir,
        motion_clip_paths=["74/74_01_poses.npz"],  # Walking motion
    )
    
    # Create environment using the factory
    print(f"Creating safety env with task: {task_cls.__name__}...")
    env = make_safety_env(
        task_cls=task_cls,
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
