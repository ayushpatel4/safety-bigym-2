#!/usr/bin/env python
"""
Dataset Generator for Safety BiGym

Generates diverse human-robot interaction episodes across multiple tasks
and human behaviour scenarios for safety evaluation.

Usage:
    # Generate dataset with all tasks and stratified human behaviours
    mjpython scripts/generate_dataset.py --output datasets/my_dataset

    # Specific tasks only
    mjpython scripts/generate_dataset.py --tasks reach pick_box dishwasher_open --output datasets/subset

    # Visualization mode (watch episodes in viewer)
    mjpython scripts/generate_dataset.py --view --tasks reach --n-per-type 1

    # Stratified vs random sampling
    mjpython scripts/generate_dataset.py --sampling stratified --n-per-type 5
    mjpython scripts/generate_dataset.py --sampling random --total-episodes 100

    # Save full observations (larger files)
    mjpython scripts/generate_dataset.py --save-obs --output datasets/full_obs
"""

import argparse
import json
import csv
import importlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import numpy as np

from bigym.action_modes import JointPositionActionMode
from safety_bigym import (
    make_safety_env,
    SafetyConfig,
    HumanConfig,
    ScenarioSampler,
    DisruptionType,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Path to AMASS motion clips
CMU_DIR = Path("/Users/ayushpatel/Documents/FYP3/CMU/CMU")

# Available tasks (same as demo_safety_env.py)
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
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def scenario_to_dict(scenario) -> dict:
    """Convert ScenarioParams to a JSON-serializable dict."""
    return {
        'clip_path': scenario.clip_path,
        'disruption_type': scenario.disruption_type.name,
        'trigger_time': scenario.trigger_time,
        'blend_duration': scenario.blend_duration,
        'speed_multiplier': scenario.speed_multiplier,
        'human_height_percentile': scenario.human_height_percentile,
        'approach_angle': scenario.approach_angle,
        'spawn_distance': scenario.spawn_distance,
        'reaching_arm': scenario.reaching_arm,
        'seed': scenario.seed,
    }


def run_episode(env, max_steps: int, save_obs: bool = False, viewer=None) -> dict:
    """
    Run a single episode and collect data.

    Args:
        env: The safety environment
        max_steps: Maximum steps per episode
        save_obs: Whether to save full observations
        viewer: Optional MuJoCo viewer for visualization

    Returns:
        dict with episode data
    """
    obs, info = env.reset()

    # Initialize data collection
    safety_data = []
    human_positions = []
    observations = [] if save_obs else None
    actions = [] if save_obs else None
    rewards = []

    # Safety counters
    ssm_violations = 0
    pfl_violations = 0
    min_separation = float('inf')
    max_contact_force = 0.0
    contact_regions = set()

    step = 0
    terminated = False
    truncated = False
    total_reward = 0.0

    while step < max_steps and not (terminated or truncated):
        # Take zero action (or could use a policy here)
        action = np.zeros(env.action_space.shape)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Get safety info
        safety = info.get("safety", {})

        # Track violations
        if safety.get("ssm_violation", False):
            ssm_violations += 1
        if safety.get("pfl_violation", False):
            pfl_violations += 1

        # Track min/max values
        sep = safety.get("min_separation", float('inf'))
        if sep < min_separation:
            min_separation = sep

        force = safety.get("max_contact_force", 0.0)
        if force > max_contact_force:
            max_contact_force = force

        # Track contact regions
        region = safety.get("contact_region")
        if region:
            contact_regions.add(region)

        # Store per-step data
        safety_data.append({
            'ssm_violation': safety.get("ssm_violation", False),
            'pfl_violation': safety.get("pfl_violation", False),
            'ssm_margin': safety.get("ssm_margin", 0.0),
            'pfl_force_ratio': safety.get("pfl_force_ratio", 0.0),
            'min_separation': safety.get("min_separation", float('inf')),
            'max_contact_force': safety.get("max_contact_force", 0.0),
        })

        # Track human position
        if hasattr(env, '_human_pelvis_id') and env._human_pelvis_id:
            human_pos = env._mojo.data.xpos[env._human_pelvis_id].copy()
            human_positions.append(human_pos)

        # Save full observations if requested
        if save_obs:
            observations.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
            actions.append(action.copy())
        rewards.append(reward)

        # Sync viewer if present
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break

        step += 1

    # Build result dict
    result = {
        'episode_length': step,
        'total_reward': total_reward,
        'terminated': terminated,
        'truncated': truncated,

        # Safety summary
        'ssm_violations': ssm_violations,
        'pfl_violations': pfl_violations,
        'min_separation': min_separation if min_separation != float('inf') else -1.0,
        'max_contact_force': max_contact_force,
        'contact_regions': list(contact_regions),

        # Per-step data
        'safety_data': safety_data,
        'human_positions': np.array(human_positions) if human_positions else np.array([]),
        'rewards': np.array(rewards),
    }

    if save_obs:
        result['observations'] = np.array(observations) if observations else np.array([])
        result['actions'] = np.array(actions) if actions else np.array([])

    return result


def save_episode(output_dir: Path, task_key: str, episode_idx: int,
                 scenario, episode_data: dict, save_obs: bool):
    """Save episode data to .npz file."""
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{task_key}_{episode_idx:04d}.npz"
    filepath = episodes_dir / filename

    # Prepare data for saving
    save_data = {
        'task': task_key,
        'episode_idx': episode_idx,
        'scenario_seed': scenario.seed,
        'scenario_disruption_type': scenario.disruption_type.name,
        'scenario_clip_path': scenario.clip_path,
        'scenario_trigger_time': scenario.trigger_time,
        'scenario_blend_duration': scenario.blend_duration,
        'scenario_speed_multiplier': scenario.speed_multiplier,
        'scenario_height_percentile': scenario.human_height_percentile,
        'scenario_approach_angle': scenario.approach_angle,
        'scenario_spawn_distance': scenario.spawn_distance,
        'scenario_reaching_arm': scenario.reaching_arm,

        'episode_length': episode_data['episode_length'],
        'total_reward': episode_data['total_reward'],
        'terminated': episode_data['terminated'],
        'truncated': episode_data['truncated'],

        'ssm_violations': episode_data['ssm_violations'],
        'pfl_violations': episode_data['pfl_violations'],
        'min_separation': episode_data['min_separation'],
        'max_contact_force': episode_data['max_contact_force'],

        'human_positions': episode_data['human_positions'],
        'rewards': episode_data['rewards'],
    }

    # Save per-step safety data as structured arrays
    safety_arrays = {
        'ssm_violation': np.array([s['ssm_violation'] for s in episode_data['safety_data']]),
        'pfl_violation': np.array([s['pfl_violation'] for s in episode_data['safety_data']]),
        'ssm_margin': np.array([s['ssm_margin'] for s in episode_data['safety_data']]),
        'pfl_force_ratio': np.array([s['pfl_force_ratio'] for s in episode_data['safety_data']]),
        'step_min_separation': np.array([s['min_separation'] for s in episode_data['safety_data']]),
        'step_max_force': np.array([s['max_contact_force'] for s in episode_data['safety_data']]),
    }
    save_data.update(safety_arrays)

    if save_obs and 'observations' in episode_data:
        save_data['observations'] = episode_data['observations']
        save_data['actions'] = episode_data['actions']

    np.savez_compressed(filepath, **save_data)
    return filepath


def generate_summary_csv(output_dir: Path, all_episodes: list):
    """Generate a CSV summary of all episodes."""
    csv_path = output_dir / "summary.csv"

    fieldnames = [
        'task', 'episode_idx', 'disruption_type', 'seed',
        'episode_length', 'total_reward',
        'ssm_violations', 'pfl_violations',
        'min_separation', 'max_contact_force',
        'trigger_time', 'speed_multiplier', 'approach_angle', 'spawn_distance'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in all_episodes:
            writer.writerow({
                'task': ep['task'],
                'episode_idx': ep['episode_idx'],
                'disruption_type': ep['scenario'].disruption_type.name,
                'seed': ep['scenario'].seed,
                'episode_length': ep['data']['episode_length'],
                'total_reward': ep['data']['total_reward'],
                'ssm_violations': ep['data']['ssm_violations'],
                'pfl_violations': ep['data']['pfl_violations'],
                'min_separation': ep['data']['min_separation'],
                'max_contact_force': ep['data']['max_contact_force'],
                'trigger_time': ep['scenario'].trigger_time,
                'speed_multiplier': ep['scenario'].speed_multiplier,
                'approach_angle': ep['scenario'].approach_angle,
                'spawn_distance': ep['scenario'].spawn_distance,
            })

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse human-robot interaction dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--output", "-o", type=str, default="datasets/safety_dataset",
        help="Output directory for dataset (default: datasets/safety_dataset)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        choices=list(TASK_MAP.keys()),
        help="Tasks to include (default: all tasks)"
    )
    parser.add_argument(
        "--sampling", choices=["stratified", "random"], default="stratified",
        help="Sampling strategy: stratified (balanced disruption types) or random"
    )
    parser.add_argument(
        "--n-per-type", type=int, default=5,
        help="Episodes per disruption type per task (stratified mode, default: 5)"
    )
    parser.add_argument(
        "--total-episodes", type=int, default=100,
        help="Total episodes per task (random mode, default: 100)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per episode (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)"
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Enable MuJoCo viewer for visualization"
    )
    parser.add_argument(
        "--save-obs", action="store_true",
        help="Save full observations (creates larger files)"
    )
    parser.add_argument(
        "--motion-dir", type=str, default=str(CMU_DIR),
        help=f"Path to AMASS motion clips (default: {CMU_DIR})"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_dir = Path(args.motion_dir)

    tasks = args.tasks if args.tasks else list(TASK_MAP.keys())

    logger.info("=" * 70)
    logger.info("Safety BiGym Dataset Generator")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Sampling: {args.sampling}")
    if args.sampling == "stratified":
        logger.info(f"Episodes per disruption type: {args.n_per_type}")
        logger.info(f"Total episodes per task: {args.n_per_type * 5}")
    else:
        logger.info(f"Total episodes per task: {args.total_episodes}")
    logger.info(f"Max steps per episode: {args.max_steps}")
    logger.info(f"Viewer: {'enabled' if args.view else 'disabled'}")
    logger.info(f"Save observations: {'yes' if args.save_obs else 'no'}")
    logger.info("=" * 70)

    # Initialize scenario sampler
    sampler = ScenarioSampler(motion_dir=motion_dir)
    logger.info(f"Discovered {len(sampler.params.clip_paths)} motion clips")

    # Action mode for all environments
    action_mode = JointPositionActionMode(floating_base=True, absolute=True)

    # Safety config - log violations but don't terminate
    safety_config = SafetyConfig(
        log_violations=False,  # We track violations ourselves
        terminate_on_violation=False,
    )

    # Track all episodes for summary
    all_episodes = []

    # Import viewer if needed
    viewer_module = None
    if args.view:
        import mujoco.viewer
        viewer_module = mujoco.viewer

    # Process each task
    for task_idx, task_key in enumerate(tasks):
        logger.info(f"\n{'─' * 70}")
        logger.info(f"Task {task_idx + 1}/{len(tasks)}: {task_key}")
        logger.info(f"{'─' * 70}")

        task_cls = load_task_cls(task_key)

        # Get scenarios based on sampling strategy
        if args.sampling == "stratified":
            stratified = sampler.get_stratified_sample(
                n_per_type=args.n_per_type,
                base_seed=args.seed + task_idx * 1000  # Different seed per task
            )
            scenarios = []
            for dtype in DisruptionType:
                scenarios.extend(stratified.get(dtype, []))
        else:
            scenarios = sampler.sample_batch(
                n=args.total_episodes,
                base_seed=args.seed + task_idx * 1000
            )

        logger.info(f"Generated {len(scenarios)} scenarios")

        # Print disruption type distribution
        type_counts = {}
        for s in scenarios:
            name = s.disruption_type.name
            type_counts[name] = type_counts.get(name, 0) + 1
        logger.info(f"Distribution: {type_counts}")

        # Run episodes for this task
        for ep_idx, scenario in enumerate(scenarios):
            logger.info(f"\n  Episode {ep_idx + 1}/{len(scenarios)} - "
                       f"{scenario.disruption_type.name} (seed={scenario.seed})")

            # Create human config for this scenario
            human_config = HumanConfig(
                motion_clip_dir=str(motion_dir),
                motion_clip_paths=[scenario.clip_path],
            )

            # Create environment
            env = make_safety_env(
                task_cls=task_cls,
                action_mode=action_mode,
                safety_config=safety_config,
                human_config=human_config,
                inject_human=True,
            )

            try:
                # Reset with scenario seed
                env.reset(seed=scenario.seed)

                if args.view and viewer_module is not None:
                    # Run with viewer
                    model = env._mojo.model
                    data = env._mojo.data

                    with viewer_module.launch_passive(model, data) as viewer:
                        logger.info(f"    Viewer open - close window to continue...")
                        episode_data = run_episode(
                            env, args.max_steps, args.save_obs, viewer
                        )
                else:
                    # Run headless
                    episode_data = run_episode(
                        env, args.max_steps, args.save_obs, viewer=None
                    )

                # Save episode
                filepath = save_episode(
                    output_dir, task_key, ep_idx, scenario, episode_data, args.save_obs
                )

                # Track for summary
                all_episodes.append({
                    'task': task_key,
                    'episode_idx': ep_idx,
                    'scenario': scenario,
                    'data': episode_data,
                    'filepath': str(filepath),
                })

                logger.info(f"    Steps: {episode_data['episode_length']} | "
                           f"SSM: {episode_data['ssm_violations']} | "
                           f"PFL: {episode_data['pfl_violations']} | "
                           f"Min sep: {episode_data['min_separation']:.3f}m | "
                           f"Max force: {episode_data['max_contact_force']:.1f}N")

            finally:
                env.close()

    # Generate summary CSV
    csv_path = generate_summary_csv(output_dir, all_episodes)
    logger.info(f"\nSummary saved to: {csv_path}")

    # Save dataset metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'tasks': tasks,
        'sampling': args.sampling,
        'n_per_type': args.n_per_type if args.sampling == 'stratified' else None,
        'total_episodes': args.total_episodes if args.sampling == 'random' else None,
        'max_steps': args.max_steps,
        'base_seed': args.seed,
        'save_obs': args.save_obs,
        'motion_dir': str(motion_dir),
        'total_episodes_generated': len(all_episodes),
        'disruption_types': [d.name for d in DisruptionType],
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("Dataset Generation Complete")
    logger.info("=" * 70)
    logger.info(f"Total episodes: {len(all_episodes)}")
    logger.info(f"Output directory: {output_dir}")

    # Aggregate stats
    total_ssm = sum(ep['data']['ssm_violations'] for ep in all_episodes)
    total_pfl = sum(ep['data']['pfl_violations'] for ep in all_episodes)
    logger.info(f"Total SSM violations: {total_ssm}")
    logger.info(f"Total PFL violations: {total_pfl}")

    # Per-task breakdown
    logger.info("\nPer-task breakdown:")
    for task_key in tasks:
        task_eps = [ep for ep in all_episodes if ep['task'] == task_key]
        task_ssm = sum(ep['data']['ssm_violations'] for ep in task_eps)
        task_pfl = sum(ep['data']['pfl_violations'] for ep in task_eps)
        logger.info(f"  {task_key}: {len(task_eps)} episodes, "
                   f"SSM: {task_ssm}, PFL: {task_pfl}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
