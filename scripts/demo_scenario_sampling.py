#!/usr/bin/env python
"""
Demo: Parameterised Scenario Sampling

Shows how the ScenarioSampler creates diverse human disruption scenarios
for safety evaluation. Demonstrates:
  - Sampling individual scenarios with reproducible seeds
  - Batch sampling for dataset creation
  - Stratified sampling by disruption type
  - Running sampled scenarios in the environment with a viewer

Usage:
  # Print sampled scenarios (no viewer needed):
    python scripts/demo_scenario_sampling.py

  # Run a specific scenario visually (needs mjpython):
    mjpython scripts/demo_scenario_sampling.py --run --seed 42

  # Run episodes cycling through disruption types:
    mjpython scripts/demo_scenario_sampling.py --run --stratified
"""

import argparse
import numpy as np
from collections import Counter
from pathlib import Path

from safety_bigym import get_amass_data_dir
from safety_bigym.scenarios import (
    ScenarioSampler,
    ScenarioParams,
    ParameterSpace,
    DisruptionType,
)


# ── Pretty printing ──────────────────────────────────────────────────────────

def print_scenario(s: ScenarioParams, idx: int = 0):
    """Print a single scenario in a readable format."""
    clip_name = Path(s.clip_path).name if s.clip_path else "N/A"
    print(f"  Scenario {idx:3d}  seed={s.seed}")
    print(f"    Disruption:  {s.disruption_type.name:<20s}  "
          f"(IK: {'yes' if s.disruption_config and s.disruption_config.requires_ik() else 'no'})")
    print(f"    Clip:        {clip_name}")
    print(f"    Timing:      trigger={s.trigger_time:.2f}s  blend={s.blend_duration:.2f}s")
    print(f"    Motion:      speed={s.speed_multiplier:.2f}x  height_pct={s.human_height_percentile:.2f}")
    print(f"    Spatial:     angle={s.approach_angle:.0f}°  distance={s.spawn_distance:.2f}m  arm={s.reaching_arm}")
    print()


# ── Demo 1: Basic sampling ───────────────────────────────────────────────────

def demo_basic_sampling(sampler: ScenarioSampler):
    print("=" * 70)
    print("DEMO 1: Basic Scenario Sampling")
    print("=" * 70)
    print()

    print("Sampling 5 scenarios with seeds 0-4:")
    print("-" * 50)
    for i in range(5):
        s = sampler.sample_scenario(seed=i)
        print_scenario(s, i)

    # Demonstrate reproducibility
    print("Reproducibility check:")
    a = sampler.sample_scenario(seed=42)
    b = sampler.sample_scenario(seed=42)
    match = (a.clip_path == b.clip_path and
             a.disruption_type == b.disruption_type and
             a.trigger_time == b.trigger_time)
    print(f"  seed=42 sampled twice → identical: {'✅ yes' if match else '❌ no'}")
    print()


# ── Demo 2: Batch sampling statistics ────────────────────────────────────────

def demo_batch_stats(sampler: ScenarioSampler):
    print("=" * 70)
    print("DEMO 2: Batch Sampling Statistics (100 scenarios)")
    print("=" * 70)
    print()

    batch = sampler.sample_batch(n=100, base_seed=0)

    # Disruption type distribution
    type_counts = Counter(s.disruption_type.name for s in batch)
    print("  Disruption type distribution:")
    for name, count in sorted(type_counts.items()):
        bar = "█" * count
        print(f"    {name:<20s} {count:3d}  {bar}")
    print()

    # Parameter ranges actually sampled
    triggers = [s.trigger_time for s in batch]
    speeds = [s.speed_multiplier for s in batch]
    distances = [s.spawn_distance for s in batch]
    angles = [s.approach_angle for s in batch]
    heights = [s.human_height_percentile for s in batch]

    print("  Parameter ranges (min → max):")
    print(f"    Trigger time:    {min(triggers):.2f}s → {max(triggers):.2f}s")
    print(f"    Speed:           {min(speeds):.2f}x → {max(speeds):.2f}x")
    print(f"    Spawn distance:  {min(distances):.2f}m → {max(distances):.2f}m")
    print(f"    Approach angle:  {min(angles):.0f}° → {max(angles):.0f}°")
    print(f"    Height pct:      {min(heights):.2f} → {max(heights):.2f}")
    print()

    # Unique clips used
    unique_clips = len(set(s.clip_path for s in batch))
    print(f"  Unique motion clips used: {unique_clips} / {len(sampler.params.clip_paths)}")

    # Arm distribution
    arm_counts = Counter(s.reaching_arm for s in batch)
    print(f"  Arm selection: right={arm_counts.get('right_arm', 0)}, left={arm_counts.get('left_arm', 0)}")
    print()


# ── Demo 3: Stratified sampling ──────────────────────────────────────────────

def demo_stratified(sampler: ScenarioSampler):
    print("=" * 70)
    print("DEMO 3: Stratified Sampling (5 per disruption type)")
    print("=" * 70)
    print()

    stratified = sampler.get_stratified_sample(n_per_type=5, base_seed=0)

    for dtype, scenarios in stratified.items():
        print(f"  ── {dtype.name} ({len(scenarios)} scenarios) ──")
        for i, s in enumerate(scenarios):
            clip_name = Path(s.clip_path).name if s.clip_path else "?"
            print(f"    [{i}] clip={clip_name:<25s} speed={s.speed_multiplier:.1f}x  "
                  f"trigger={s.trigger_time:.1f}s  dist={s.spawn_distance:.1f}m  "
                  f"angle={s.approach_angle:.0f}°")
        print()


# ── Demo 4: Custom parameter space ───────────────────────────────────────────

def demo_custom_space(sampler: ScenarioSampler):
    print("=" * 70)
    print("DEMO 4: Custom Parameter Space")
    print("=" * 70)
    print()

    # Create a narrow parameter space for close-range, fast scenarios
    custom_space = ParameterSpace(
        clip_paths=sampler.params.clip_paths[:20],  # Only first 20 clips
        disruption_weights={
            DisruptionType.DIRECT: 0.5,       # Bias toward direct interaction
            DisruptionType.OBSTRUCTION: 0.3,
            DisruptionType.INCIDENTAL: 0.1,
            DisruptionType.SHARED_GOAL: 0.1,
            DisruptionType.RANDOM_PERTURBED: 0.0,  # Exclude this type
        },
        trigger_time_range=(0.5, 1.5),    # Fast triggers
        speed_range=(1.5, 2.0),           # Fast motion
        spawn_distance_range=(1.0, 1.5),  # Close spawns
    )

    custom_sampler = ScenarioSampler(parameter_space=custom_space)
    batch = custom_sampler.sample_batch(n=20, base_seed=100)

    type_counts = Counter(s.disruption_type.name for s in batch)
    print("  Custom space: close-range, fast, direct-biased")
    print()
    print("  Disruption distribution (20 scenarios):")
    for name, count in sorted(type_counts.items()):
        print(f"    {name:<20s} {count}")
    print()

    triggers = [s.trigger_time for s in batch]
    speeds = [s.speed_multiplier for s in batch]
    distances = [s.spawn_distance for s in batch]
    print(f"  Trigger: {min(triggers):.2f}–{max(triggers):.2f}s  "
          f"Speed: {min(speeds):.2f}–{max(speeds):.2f}x  "
          f"Distance: {min(distances):.2f}–{max(distances):.2f}m")
    print()


# ── Demo 5: Run scenarios in environment (needs mjpython) ────────────────────

def demo_run_scenario(sampler: ScenarioSampler, seed: int, cmu_dir: Path):
    """Run a single scenario in the MuJoCo viewer."""
    import mujoco.viewer
    from bigym.action_modes import JointPositionActionMode
    from safety_bigym import make_safety_env, SafetyConfig, HumanConfig

    scenario = sampler.sample_scenario(seed)

    print("=" * 70)
    print(f"RUNNING SCENARIO seed={seed}")
    print("=" * 70)
    print_scenario(scenario, seed)

    human_config = HumanConfig(
        motion_clip_dir=str(cmu_dir),
        motion_clip_paths=[scenario.clip_path],
    )

    env = make_safety_env(
        task_cls=__import__('bigym.bigym_env', fromlist=['BiGymEnv']).BiGymEnv,
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        safety_config=SafetyConfig(log_violations=True, terminate_on_violation=False),
        human_config=human_config,
        inject_human=True,
    )

    obs, info = env.reset(seed=seed)
    model = env._mojo.model
    data = env._mojo.data

    print(f"Scenario: {scenario.disruption_type.name}")
    print(f"Opening viewer... (close window to stop)")
    print("-" * 50)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            action = np.zeros(env.action_space.shape)
            obs, rew, term, trunc, info = env.step(action)
            safety = info.get("safety", {})
            step += 1

            if step % 200 == 0:
                sep = safety.get('min_separation', float('inf'))
                sep_str = f"{sep:.2f}m" if sep != float('inf') else "inf"
                print(f"  Step {step:5d} | Sep: {sep_str} | "
                      f"SSM: {'⚠️' if safety.get('ssm_violation') else '✓'} | "
                      f"PFL: {'⚠️' if safety.get('pfl_violation') else '✓'} | "
                      f"Force: {safety.get('max_contact_force', 0):.1f}N")

            if term or trunc:
                break

            viewer.sync()

    env.close()
    print("Done!")


def demo_run_stratified(sampler: ScenarioSampler, cmu_dir: Path):
    """Run one scenario per disruption type sequentially."""
    import mujoco.viewer
    from bigym.action_modes import JointPositionActionMode
    from safety_bigym import make_safety_env, SafetyConfig, HumanConfig

    stratified = sampler.get_stratified_sample(n_per_type=1, base_seed=0)

    print("=" * 70)
    print("RUNNING STRATIFIED SCENARIOS (one per disruption type)")
    print("=" * 70)
    print()

    for dtype, scenarios in stratified.items():
        scenario = scenarios[0]
        print(f"\n{'─' * 50}")
        print(f"  {dtype.name}")
        print(f"{'─' * 50}")
        print_scenario(scenario, scenario.seed)

        human_config = HumanConfig(
            motion_clip_dir=str(cmu_dir),
            motion_clip_paths=[scenario.clip_path],
        )

        env = make_safety_env(
            task_cls=__import__('bigym.bigym_env', fromlist=['BiGymEnv']).BiGymEnv,
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            safety_config=SafetyConfig(log_violations=True, terminate_on_violation=False),
            human_config=human_config,
            inject_human=True,
        )

        obs, info = env.reset(seed=scenario.seed)
        model = env._mojo.model
        data = env._mojo.data

        print(f"  Viewer open — close to advance to next type...")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            while viewer.is_running():
                action = np.zeros(env.action_space.shape)
                obs, rew, term, trunc, info = env.step(action)
                step += 1
                if step % 500 == 0:
                    safety = info.get("safety", {})
                    sep = safety.get('min_separation', float('inf'))
                    print(f"    Step {step:5d} | Sep: {sep:.2f}m | "
                          f"SSM: {'⚠️' if safety.get('ssm_violation') else '✓'} | "
                          f"PFL: {'⚠️' if safety.get('pfl_violation') else '✓'}")
                if term or trunc:
                    break
                viewer.sync()

        env.close()

    print("\nAll disruption types demonstrated!")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Parameterised Scenario Sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", action="store_true",
                        help="Run scenario(s) in MuJoCo viewer (requires mjpython)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for single scenario run (default: 42)")
    parser.add_argument("--stratified", action="store_true",
                        help="With --run, cycle through one scenario per disruption type")
    parser.add_argument("--amass-dir", type=str, default=None,
                        help="Path to AMASS CMU clip root (overrides $AMASS_DATA_DIR)")
    args = parser.parse_args()

    cmu_dir = get_amass_data_dir(args.amass_dir)

    # Create sampler with auto-discovered clips
    sampler = ScenarioSampler(motion_dir=cmu_dir)
    print(f"Discovered {len(sampler.params.clip_paths)} motion clips in {cmu_dir}\n")

    if args.run:
        if args.stratified:
            demo_run_stratified(sampler, cmu_dir)
        else:
            demo_run_scenario(sampler, seed=args.seed, cmu_dir=cmu_dir)
    else:
        # Print-only demos (no viewer needed)
        demo_basic_sampling(sampler)
        demo_batch_stats(sampler)
        demo_stratified(sampler)
        demo_custom_space(sampler)

        print("=" * 70)
        print("To run scenarios visually:")
        print("  mjpython scripts/demo_scenario_sampling.py --run --seed 42")
        print("  mjpython scripts/demo_scenario_sampling.py --run --stratified")
        print("=" * 70)


if __name__ == "__main__":
    main()
