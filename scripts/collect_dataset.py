#!/usr/bin/env python
"""
Dataset Collection Script for Safety BiGym

Iterates through specified robot tasks and generates diverse human-robot interaction scenarios.
Collects episode data including observations, actions, rewards, and detailed safety information.
Saves the data to an HDF5 file.

Usage:
  python scripts/collect_dataset.py --tasks reach pick_box --episodes-per-task 10 --output dataset.h5
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import h5py
import numpy as np
import mujoco

from bigym.action_modes import JointPositionActionMode
from bigym.bigym_env import BiGymEnv
from safety_bigym import make_safety_env, SafetyConfig, HumanConfig, get_amass_data_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Task Mapping (same as demo_safety_env.py)
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

def collect_episode(
    env, 
    task_name: str, 
    episode_idx: int, 
    h5_group: h5py.Group,
    random_actions: bool = False,
    viewer = None
):
    """
    Run a single episode and save data to HDF5 group.
    """
    # Reset environment
    seed = int(time.time() * 1000) % 2**32
    obs, info = env.reset(seed=seed)
    
    # Create episode group
    ep_group = h5_group.create_group(f"episode_{episode_idx:04d}")
    
    # Save scenario metadata as attributes
    scenario = info.get("scenario", {})
    ep_group.attrs["task"] = task_name
    ep_group.attrs["seed"] = seed
    for k, v in scenario.items():
        if isinstance(v, (str, int, float, bool)):
            ep_group.attrs[f"scenario_{k}"] = v
    
    # Data buffers
    observations = []  # Assuming obs is a dict, we might flatten or save keys
    actions = []
    rewards = []
    
    # Safety data buffers
    safety_ssm_violation = []
    safety_pfl_violation = []
    safety_min_separation = []
    safety_max_force = []
    
    # Human state buffer (pos, quat)
    human_states = []

    terminated = False
    truncated = False
    step_count = 0
    max_steps = 200 # Limit steps per episode for dataset collection time

    while not (terminated or truncated) and step_count < max_steps:
        # Select action
        if random_actions:
            action = env.action_space.sample()
        else:
            action = np.zeros(env.action_space.shape)
            
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        # Check observation structure - likely a dict with 'proprioception', 'visual', etc.
        # For simplicity in this generic script, we'll try to save the 'proprioception' if available,
        # or try to save the whole obs if it's a flat array (unlikely in BiGym).
        # We'll save 'proprioception' from obs dict if present.
        if isinstance(obs, dict):
            if 'proprioception' in obs:
                 observations.append(obs['proprioception'])
            else:
                # Fallback: flatten values? Or just skip observing if structure unknown?
                # Let's try to grab 'qpos' if available as proxy
                pass
        elif isinstance(obs, (np.ndarray, list)):
            observations.append(obs)
            
        actions.append(action)
        rewards.append(reward)
        
        # Safety info
        safety = info.get("safety", {})
        safety_ssm_violation.append(safety.get("ssm_violation", False))
        safety_pfl_violation.append(safety.get("pfl_violation", False))
        safety_min_separation.append(safety.get("min_separation", -1.0))
        safety_max_force.append(safety.get("max_contact_force", 0.0))
        
        # Human state (if human exists)
        if env._human_pelvis_id is not None:
            # Get root position and orientation from MuJoCo data
            # Typically root body or joint. Let's use pelvis body pos/quat if possible.
            # Using env._human_pelvis_id which is a body ID.
            pos = env._mojo.data.xpos[env._human_pelvis_id].copy()
            quat = env._mojo.data.xquat[env._human_pelvis_id].copy()
            human_states.append(np.concatenate([pos, quat]))
        else:
            human_states.append(np.zeros(7))
            
        step_count += 1
        
        if viewer is not None:
            viewer.sync()

    # Save buffers to HDF5
    # Compress data to save space
    compression = "gzip"
    
    if observations:
        ep_group.create_dataset("obs", data=np.array(observations), compression=compression)
    ep_group.create_dataset("actions", data=np.array(actions), compression=compression)
    ep_group.create_dataset("rewards", data=np.array(rewards), compression=compression)
    
    # Safety group
    saf_group = ep_group.create_group("safety")
    saf_group.create_dataset("ssm_violation", data=np.array(safety_ssm_violation), compression=compression)
    saf_group.create_dataset("pfl_violation", data=np.array(safety_pfl_violation), compression=compression)
    saf_group.create_dataset("min_separation", data=np.array(safety_min_separation), compression=compression)
    saf_group.create_dataset("max_contact_force", data=np.array(safety_max_force), compression=compression)
    
    ep_group.create_dataset("human_state", data=np.array(human_states), compression=compression)
    
    logger.info(f"    Collected episode {episode_idx}: {step_count} steps, "
                f"SSM: {sum(safety_ssm_violation)}, PFL: {sum(safety_pfl_violation)}")

def main():
    parser = argparse.ArgumentParser(description="Collect Safety BiGym Dataset")
    parser.add_argument("--tasks", nargs="+", default=["reach"], 
                        help=f"List of tasks to run (choices: {', '.join(TASK_MAP.keys())} or 'all')")
    parser.add_argument("--episodes-per-task", type=int, default=5, 
                        help="Number of episodes to collect per task")
    parser.add_argument("--output", type=str, default="safety_bigym_dataset.h5", 
                        help="Output HDF5 filename")
    parser.add_argument("--random-actions", action="store_true", 
                        help="Use random actions instead of zero actions (robot holds position)")
    parser.add_argument("--render", action="store_true",
                        help="Visualize collection in MuJoCo viewer")
    parser.add_argument("--amass-dir", type=str, default=None,
                        help="Path to AMASS CMU clip root (overrides $AMASS_DATA_DIR)")

    args = parser.parse_args()

    cmu_dir = get_amass_data_dir(args.amass_dir)

    # Handle 'all' tasks
    if "all" in args.tasks:
        tasks_to_run = list(TASK_MAP.keys())
    else:
        tasks_to_run = args.tasks
        
    logger.info(f"Starting dataset collection for tasks: {tasks_to_run}")
    logger.info(f"Output file: {args.output}")
    
    # Open HDF5 file
    with h5py.File(args.output, "w") as f:
        f.attrs["created_at"] = time.ctime()
        f.attrs["version"] = "0.1.0"
        
        # Iterate tasks
        for task_key in tasks_to_run:
            logger.info(f"Initializing task: {task_key}")
            try:
                task_cls = load_task_cls(task_key)
                
                # Setup environment
                action_mode = JointPositionActionMode(floating_base=True, absolute=True)
                
                safety_config = SafetyConfig(
                    log_violations=False,
                    terminate_on_violation=False,
                )
                
                human_config = HumanConfig(
                    motion_clip_dir=str(cmu_dir),
                    motion_clip_paths=[],
                )
                
                env = make_safety_env(
                    task_cls=task_cls,
                    action_mode=action_mode,
                    safety_config=safety_config,
                    human_config=human_config,
                    inject_human=True,
                )
                
                # Create task group
                task_group = f.create_group(task_key)
                
                # Setup viewer context if rendering requested
                if args.render:
                    model = env._mojo.model
                    data = env._mojo.data
                    logger.info("Opening viewer... (close window to stop collection)")
                    viewer_ctx = mujoco.viewer.launch_passive(model, data)
                else:
                    from contextlib import nullcontext
                    viewer_ctx = nullcontext()

                # Run episodes
                with viewer_ctx as viewer:
                    for i in range(args.episodes_per_task):
                        if args.render and not viewer.is_running():
                            logger.info("Viewer closed by user. Stopping collection.")
                            break
                        
                        collect_episode(
                            env, task_key, i, task_group, 
                            random_actions=args.random_actions,
                            viewer=viewer if args.render else None
                        )
                    
                env.close()
                
            except Exception as e:
                logger.error(f"Failed to collect data for task {task_key}: {e}", exc_info=True)
                
    logger.info("Dataset collection complete!")

if __name__ == "__main__":
    main()


#mjpython scripts/collect_dataset.py --tasks all --episodes-per-task 5 --output full_dataset.h5 --random-actions --render