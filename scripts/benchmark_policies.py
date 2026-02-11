#!/usr/bin/env python
"""
Safety Benchmark CLI

Runs safety evaluation for visuomotor policies across BiGym tasks and scenario sampling.
Outputs a summary table and detailed JSON report.

Usage:
  python scripts/benchmark_policies.py --tasks reach pick_box --episodes 10
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import h5py # Not needed for running, but maybe for logging? No, using json.

from safety_bigym import make_safety_env, SafetyConfig, HumanConfig
from safety_bigym.benchmark.safety_benchmark import SafetyBenchmark
from safety_bigym.benchmark.policy import RandomPolicy
from bigym.action_modes import JointPositionActionMode

# Check for rich for pretty printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Task Mapping
TASK_MAP = {
    "default": "bigym.bigym_env:BiGymEnv",
    "reach": "bigym.envs.reach_target:ReachTargetSingle",
    "reach_dual": "bigym.envs.reach_target:ReachTargetDual",
    "pick_box": "bigym.envs.pick_and_place:PickBox",
    "pick_box_real": "bigym.envs.pick_and_place:PickBoxReal", # Maybe?
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



def load_task_cls(task_key: str) -> Any:
    """Load a task class by key from TASK_MAP."""
    if task_key not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_key}. Choose from: {list(TASK_MAP.keys())}")
    module_path, cls_name = TASK_MAP[task_key].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)

def print_summary_table(all_results: Dict[str, Any], comprehensive: bool = False):
    """Print summary table using rich if available."""
    if not HAVE_RICH:
        print("\n--- Benchmark Summary ---")
        for task, res in all_results.items():
            metrics = res["metrics"]
            print(f"Task: {task}")
            print(f"  SSM Violation Rate: {metrics['ssm_violation_rate']:.1%} (Steps: {metrics['ssm_step_rate']:.1%})")
            print(f"  PFL Violation Rate: {metrics['pfl_violation_rate']:.1%} (Steps: {metrics['pfl_step_rate']:.1%})")
            print(f"  Avg Contact Force: {metrics['avg_contact_force']:.1f} N")
            print(f"  Max Force: {metrics['max_force_severity']:.1f} N")
            if comprehensive:
                print(f"  Time to 1st SSM: {metrics['avg_time_to_ssm']:.2f}s")
                print(f"  SSM Events: {metrics['avg_ssm_events']:.1f}")
        return

    console = Console()
    table = Table(title="Safety Benchmark Results")
    
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Episodes", justify="right")
    table.add_column("SSM Rate", justify="right", style="magenta")
    table.add_column("SSM Step %", justify="right", style="magenta dim")
    table.add_column("PFL Rate", justify="right", style="red")
    table.add_column("PFL Step %", justify="right", style="red dim")
    table.add_column("Collision", justify="right")
    table.add_column("Avg Force (N)", justify="right")
    table.add_column("Max Force (N)", justify="right")
    
    if comprehensive:
        table.add_column("1st SSM (s)", justify="right", style="yellow")
        table.add_column("SSM Events", justify="right", style="yellow dim")
        table.add_column("1st PFL (s)", justify="right", style="orange1")
    
    for task_name, res in all_results.items():
        metrics = res["metrics"]
        num = res["num_episodes"]
        
        row = [
            task_name,
            str(num),
            f"{metrics['ssm_violation_rate']:.1%}",
            f"{metrics['ssm_step_rate']:.1%}",
            f"{metrics['pfl_violation_rate']:.1%}",
            f"{metrics['pfl_step_rate']:.1%}",
            f"{metrics['collision_rate']:.1%}",
            f"{metrics['avg_contact_force']:.1f}",
            f"{metrics['max_force_severity']:.1f}"
        ]
        
        if comprehensive:
            ts = metrics['avg_time_to_ssm']
            tp = metrics['avg_time_to_pfl']
            row.append(f"{ts:.2f}" if ts >= 0 else "-")
            row.append(f"{metrics['avg_ssm_events']:.1f}")
            row.append(f"{tp:.2f}" if tp >= 0 else "-")
            
        table.add_row(*row)
        
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Safety Benchmark Runner")
    parser.add_argument("--tasks", nargs="+", default=["reach"], 
                        help=f"List of tasks to evaluate (choices: {', '.join(TASK_MAP.keys())} or 'all')")
    parser.add_argument("--policy", type=str, default="random", choices=["random"],
                        help="Policy to evaluate")
    parser.add_argument("--episodes", type=int, default=10, 
                        help="Number of episodes per task")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Master random seed")
    parser.add_argument("--render", action="store_true", 
                        help="Visualize evaluation in MuJoCo viewer")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Show detailed safety metrics (time to violation, event counts)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    if "all" in args.tasks:
        tasks_to_run = list(TASK_MAP.keys())
    else:
        tasks_to_run = args.tasks
        
    all_results = {}
    
    # Common config
    action_mode = JointPositionActionMode(floating_base=True, absolute=True)
    # Default human config (auto-discovers clips from CMU dir)
    # We assume CMU_DIR is in standard location or handled by default logic
    # For benchmark reproducibility, maybe we should fix the clip set?
    # For now, let scenariosampler handle it deterministically given seed.
    # Create human config with AMASS motion clip
    cmu_clips_dir = "/Users/ayushpatel/Documents/FYP3/CMU/CMU"
    try:
        human_config = HumanConfig(
            motion_clip_dir=cmu_clips_dir,
        motion_clip_paths=[
            "74/74_01_poses.npz",  # Walking
            "74/74_02_poses.npz",  # Walking
            "74/74_03_poses.npz",  # Walking
            "49/49_01_poses.npz",  # General motion
            "49/49_02_poses.npz",  # General motion
            "25/25_01_poses.npz",  # Diverse motion
        ],
        )
    except Exception:
        # Fallback if specific clip not found
        human_config = HumanConfig(motion_clip_paths=[])
    
    for task_key in tasks_to_run:
        logger.info(f"Evaluating task: {task_key}")
        try:
            task_cls = load_task_cls(task_key)
            
            # Initialize benchmark
            benchmark = SafetyBenchmark(
                task_cls=task_cls,
                action_mode=action_mode,
                human_config=human_config,
                render=args.render
            )
            
            # Initialize policy
            # We need to instantiate env once to get action space? 
            # Or SafetyBenchmark exposes an env property?
            # Or make_safety_env just to get space.
            # Efficient way: instantiate one env, get space, close.
            # But RandomPolicy just needs shape. BiGym default shape is often 7+gripper.
            # Let's instantiate a temp env to get correct space.
            
            temp_env = make_safety_env(task_cls, action_mode=action_mode, human_config=human_config, inject_human=False)
            action_space = temp_env.action_space
            temp_env.close()
            
            if args.policy == "random":
                policy = RandomPolicy(action_space)
            else:
                raise ValueError("Unknown policy")
                
            # Run evaluation
            logger.info("  Running evaluation loop...")
            results = benchmark.evaluate(
                policy=policy,
                num_episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps
            )
            
            all_results[task_key] = results
            
            # Print intermediate result
            m = results["metrics"]
            logger.info(f"  -> SSM Violated: {m['ssm_violation_rate']:.1%}, Max Force: {m['max_force_severity']:.1f}N")
            
        except Exception as e:
            logger.error(f"Failed to evaluate task {task_key}: {e}", exc_info=True)
            
    # Save results
    with open(args.output, 'w') as f:
        # Convert numpy types to native python for JSON serialization
        def convert(o):
            if isinstance(o, np.bool_): return bool(o)
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError(f"Unserializable type: {type(o)}")
            
        json.dump(all_results, f, indent=2, default=convert)
        
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    print_summary_table(all_results, comprehensive=args.comprehensive)

if __name__ == "__main__":
    main()
