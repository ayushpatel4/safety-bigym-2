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
from safety_bigym.benchmark.policy import RandomPolicy, SafePolicy, DiffusionPolicyWrapper
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
    "dishwasher_load_plates": "bigym.envs.dishwasher_plates:DishwasherLoadPlates",
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

def create_results_table(data: Dict[str, Dict], title: str, key_name: str, comprehensive: bool = False) -> Table:
    """Helper to create a rich table from results dictionary."""
    table = Table(title=title)
    
    table.add_column(key_name, style="cyan", no_wrap=True)
    table.add_column("Episodes", justify="right")
    table.add_column("Success", justify="right", style="green")
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
    
    # Sort by key for consistency
    for name in sorted(data.keys()):
        metrics = data[name]
        # Handle case where metrics might be nested or direct
        # Check if 'metrics' key exists (task level) or if it's direct (breakdown level)
        if "metrics" in metrics:
            m = metrics["metrics"]
            num = metrics["num_episodes"]
        else:
            m = metrics
            # We don't have num_episodes per group readily available in metrics dict
            # unless we add it to _compute_aggregate_metrics.
            # But the 'metrics' dict returned by _compute_aggregate_metrics doesn't have 'num_episodes'
            # Let's assume we can derive it or ignore it?
            # Actually, _compute_aggregate_metrics returns just metrics.
            # We can't easily get num_episodes from it without modifying it.
            # Let's just print "-" for episodes in breakdown or modify _compute_aggregate_metrics to include count.
            num = "-" 
            
        row = [
            name,
            str(num),
            f"{m.get('success_rate', 0):.1%}",
            f"{m['ssm_violation_rate']:.1%}",
            f"{m['ssm_step_rate']:.1%}",
            f"{m['pfl_violation_rate']:.1%}",
            f"{m['pfl_step_rate']:.1%}",
            f"{m['collision_rate']:.1%}",
            f"{m['avg_contact_force']:.1f}",
            f"{m['max_force_severity']:.1f}"
        ]
        
        if comprehensive:
            ts = m['avg_time_to_ssm']
            tp = m['avg_time_to_pfl']
            row.append(f"{ts:.2f}" if ts >= 0 else "-")
            row.append(f"{m['avg_ssm_events']:.1f}")
            row.append(f"{tp:.2f}" if tp >= 0 else "-")
            
        table.add_row(*row)
    return table

def print_summary_table(all_results: Dict[str, Any], comprehensive: bool = False, report_file: str = None):
    """Print summary tables using rich if available. Optionally save to file."""
    if not HAVE_RICH:
        print("Rich not installed, basic output only.")
        return

    # Prepare outputs
    outputs = []
    # Standard console output (let rich determine width)
    outputs.append(Console())
    
    # File output if requested (force width to avoid truncation)
    if report_file:
        file_console = Console(file=open(report_file, "w"), width=300)
        outputs.append(file_console)
    
    for console in outputs:
        # Main Task Table
        console.print(create_results_table(all_results, "Benchmark Results by Task", "Task", comprehensive))
        
        if comprehensive:
            for task, res in all_results.items():
                if "by_scenario" in res and res["by_scenario"]:
                    console.print(create_results_table(res["by_scenario"], f"Breakdown: Scenario Type ({task})", "Scenario", comprehensive))
                if "by_trajectory" in res and res["by_trajectory"]:
                    console.print(create_results_table(res["by_trajectory"], f"Breakdown: Trajectory Type ({task})", "Trajectory", comprehensive))
                if "by_motion" in res and res["by_motion"]:
                    console.print(create_results_table(res["by_motion"], f"Breakdown: Motion Clip ({task})", "Motion", comprehensive))
            
    if report_file:
        print(f"Full report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Safety Benchmark Runner")
    parser.add_argument("--tasks", nargs="+", default=["reach"], 
                        help=f"List of tasks to evaluate (choices: {', '.join(TASK_MAP.keys())} or 'all')")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "safe", "diffusion"],
                        help="Policy to evaluate")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Path to RoboBase snapshot .pt file (required for --policy diffusion)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for diffusion policy (cpu, mps, cuda)")
    parser.add_argument("--inference-steps", type=int, default=None,
                        help="Override diffusion inference steps (default: use training value, e.g. 50). Lower = faster (try 10)")
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
    parser.add_argument("--report-file", type=str, default="benchmark_report.txt",
                        help="Output text file for formatted tables (avoids truncation)")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Directory to save episode videos for headless review (e.g. 'recordings/')")
    parser.add_argument("--record-resolution", type=int, nargs=2, default=[640, 480],
                        metavar=("W", "H"),
                        help="Video resolution width height (default: 640 480)")
    
    args = parser.parse_args()
    
    if "all" in args.tasks:
        tasks_to_run = list(TASK_MAP.keys())
    else:
        tasks_to_run = args.tasks
        
    all_results = {}
    
    # Common config
    action_mode = JointPositionActionMode(floating_base=True, absolute=True)
    # Auto-discover motion clips from CMU directory for diverse human motions.
    # Picks one clip per subject folder to maximize diversity across subjects.
    cmu_clips_dir = os.environ.get("AMASS_DATA_DIR")
    if not cmu_clips_dir:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it to the CMU AMASS root, e.g.\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
        )
    try:
        import glob
        all_clips = sorted(glob.glob(f"{cmu_clips_dir}/*/*_poses.npz"))
        # Pick first clip from each subject folder for diversity
        seen_subjects = set()
        diverse_clips = []
        for clip in all_clips:
            rel = str(Path(clip).relative_to(cmu_clips_dir))  # e.g. "74/74_01_poses.npz"
            subject = rel.split("/")[0]
            if subject not in seen_subjects:
                seen_subjects.add(subject)
                diverse_clips.append(rel)
        logger.info(f"Discovered {len(diverse_clips)} motion clips from {len(seen_subjects)} subjects in {cmu_clips_dir}")
        human_config = HumanConfig(
            motion_clip_dir=cmu_clips_dir,
            motion_clip_paths=diverse_clips,
        )
    except Exception as e:
        logger.warning(f"Failed to auto-discover clips: {e}. Using empty clip list.")
        human_config = HumanConfig(motion_clip_paths=[])
    
    for task_key in tasks_to_run:
        logger.info(f"Evaluating task: {task_key}")
        try:
            task_cls = load_task_cls(task_key)

            # For diffusion policy, construct wrapper first so we can use
            # its action_mode and observation_config (matching the training
            # config) for the benchmark env.
            task_action_mode = action_mode
            env_kwargs = {}
            if args.policy == "diffusion":
                if args.snapshot is None:
                    raise ValueError("--snapshot is required when using --policy diffusion")
                policy = DiffusionPolicyWrapper(
                    snapshot_path=args.snapshot,
                    action_space=None,  # will be set below after env creation
                    device=args.device,
                    num_inference_steps=args.inference_steps,
                    motion_clip_dir=cmu_clips_dir,
                )
                task_action_mode = policy.action_mode
                env_kwargs["observation_config"] = policy.observation_config

            # Initialize benchmark with the correct action mode and obs config
            benchmark = SafetyBenchmark(
                task_cls=task_cls,
                action_mode=task_action_mode,
                human_config=human_config,
                render=args.render,
                record_dir=args.record_dir,
                record_resolution=tuple(args.record_resolution),
                env_kwargs=env_kwargs,
            )

            temp_env = make_safety_env(task_cls, action_mode=task_action_mode, human_config=human_config, inject_human=False, **env_kwargs)
            action_space = temp_env.action_space
            temp_env.close()

            if args.policy == "random":
                policy = RandomPolicy(action_space)
            elif args.policy == "safe":
                policy = SafePolicy(action_space)
            elif args.policy == "diffusion":
                policy._raw_action_space = action_space
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
    print_summary_table(all_results, comprehensive=args.comprehensive, report_file=args.report_file)

if __name__ == "__main__":
    main()
