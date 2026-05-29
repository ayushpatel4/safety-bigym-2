#!/usr/bin/env python
"""
Visual Demo: COWORKER Disruption

Spawns an H1 robot + SMPL-H human in a BiGym task scene and forces every
reset to use ``DisruptionType.COWORKER``. The human parks near the robot
and cycles through extend/hold/retract/idle phases with its active arm,
reaching toward the robot end-effector or the task object.

Usage::

    export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
    cd safety_bigym
    mjpython scripts/demo_coworker.py --spawn walk_in --reach-target alternate
    mjpython scripts/demo_coworker.py --spawn in_place --arm left --reach-target ee

    # G1 patrol with stage-2 train disruption knobs (no AMASS needed):
    mjpython scripts/demo_coworker.py --human g1 --spawn patrol --stage train \\
        --task saucepan --seed 0

    # SMPL-H capsules, procedural motion (no AMASS — walks like G1):
    mjpython scripts/demo_coworker.py --human smplh --smplh-motion procedural \\
        --spawn patrol --stage train --task saucepan --seed 0

The script prints reach-phase transitions to stdout so the user can
correlate what they see in the viewer with the underlying state machine.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

logging.basicConfig(level=logging.INFO)

from bigym.action_modes import JointPositionActionMode
from safety_bigym import HumanConfig, SafetyConfig, make_safety_env
from safety_bigym.scenarios import (
    DisruptionType,
    ParameterSpace,
    ScenarioSampler,
)


TASK_MAP = {
    "default": "bigym.bigym_env:BiGymEnv",
    "reach": "bigym.envs.reach_target:ReachTargetSingle",
    "reach_dual": "bigym.envs.reach_target:ReachTargetDual",
    "pick_box": "bigym.envs.pick_and_place:PickBox",
    "saucepan": "bigym.envs.pick_and_place:SaucepanToHob",
    "take_cups": "bigym.envs.pick_and_place:TakeCups",
    "put_cups": "bigym.envs.pick_and_place:PutCups",
    "flip_cup": "bigym.envs.manipulation:FlipCup",
    "stack_blocks": "bigym.envs.manipulation:StackBlocks",
    "dishwasher_open": "bigym.envs.dishwasher:DishwasherOpen",
    "dishwasher_close": "bigym.envs.dishwasher:DishwasherClose",
    "dishwasher_load_plates": "bigym.envs.dishwasher:DishwasherLoadPlates",
    "drawer_open": "bigym.envs.cupboards:DrawerTopOpen",
    "drawers_open_all": "bigym.envs.cupboards:DrawersAllOpen",
    "drawers_close_all": "bigym.envs.cupboards:DrawersAllClose",
    "cupboard_open": "bigym.envs.cupboards:CupboardsOpenAll",
    "move_plate": "bigym.envs.move_plates:MovePlate",
    "groceries": "bigym.envs.groceries:GroceriesStoreLower",
}


def _load_task_cls(task_key: str) -> type:
    """Resolve a task key to its class. Accepts a key from ``TASK_MAP``
    or an arbitrary ``module.path:ClassName`` string for tasks not
    pre-registered here."""
    spec = TASK_MAP.get(task_key, task_key)
    if ":" not in spec:
        raise ValueError(
            f"Unknown task {task_key!r}. Either pick one of "
            f"{sorted(TASK_MAP)} or pass a fully-qualified "
            "'module.path:ClassName' string."
        )
    module_path, cls_name = spec.rsplit(":", 1)
    import importlib

    return getattr(importlib.import_module(module_path), cls_name)


# Mirror cfgs/disruption/coworker_*.yaml — same bands as g1_coworker_smoke.py.
_STAGE_BANDS = {
    "default": {},
    "idle": {
        "coworker_closest_approach_range": (3.0, 3.6),
        "coworker_reach_period_range": (30.0, 40.0),
        "coworker_target_mix_p_ee_range": (0.0, 0.0),
        "coworker_near_loiter_range": (1.0, 2.0),
        "coworker_walk_speed_range": (0.5, 0.8),
    },
    "easy": {
        "coworker_closest_approach_range": (1.8, 2.5),
        "coworker_reach_period_range": (8.0, 11.0),
        "coworker_target_mix_p_ee_range": (0.1, 0.25),
        "coworker_near_loiter_range": (3.0, 5.0),
        "coworker_walk_speed_range": (0.7, 1.1),
    },
    "train": {
        "coworker_closest_approach_range": (0.55, 0.85),
        "coworker_reach_period_range": (0.9, 1.6),
        "coworker_target_mix_p_ee_range": (0.55, 0.85),
        "coworker_near_loiter_range": (12.0, 18.0),
        "coworker_walk_speed_range": (1.0, 1.5),
        "coworker_trajectory_weights": {
            "COWORKER_PATROL": 8.0,
            "APPROACH_LOITER_DEPART": 1.0,
            "STATIONARY": 1.0,
        },
    },
}


def _make_sampler(
    motion_dir: Optional[str],
    motion_clip_paths,
    spawn_mode: str,
    arm: Optional[str],
    reach_target: str,
    stage: str,
) -> ScenarioSampler:
    """Build a sampler that forces COWORKER and biases the chosen knobs.

    We force COWORKER via ``disruption_weights={COWORKER: 1.0}`` and then
    post-process each sampled scenario in :func:`_override_scenario` to
    pin spawn mode / arm / target mix. The sampler exposes its result by
    value so this is safe to do.
    """
    return ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=motion_clip_paths,
            disruption_weights={DisruptionType.COWORKER: 1.0},
            **_STAGE_BANDS.get(stage, {}),
        ),
        motion_dir=motion_dir,
    )


def _override_scenario(scenario, spawn_mode: str, arm: Optional[str], reach_target: str):
    """Pin the scenario knobs after sampling."""
    if spawn_mode == "walk_in":
        scenario.trajectory_type = "APPROACH_LOITER_DEPART"
    elif spawn_mode == "in_place":
        scenario.trajectory_type = "STATIONARY"
    elif spawn_mode == "patrol":
        scenario.trajectory_type = "COWORKER_PATROL"
        # Ensure at least two away-and-back cycles fit in one viewing session.
        scenario.patrol_excursions = max(int(getattr(scenario, "patrol_excursions", 1)), 2)
        scenario.patrol_near_loiter = min(
            float(getattr(scenario, "patrol_near_loiter", 8.0)), 7.0
        )
    # else "alternate": let the sampler's choice stand

    if arm in ("left", "right"):
        scenario.disruption_config.coworker_active_arm = f"{arm}_arm"

    if reach_target == "ee":
        scenario.disruption_config.coworker_target_mix = (1.0, 0.0)
    elif reach_target == "task":
        scenario.disruption_config.coworker_target_mix = (0.0, 1.0)
    else:
        scenario.disruption_config.coworker_target_mix = (0.5, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="COWORKER disruption visual demo")
    parser.add_argument(
        "--task",
        default="reach",
        help=(
            "Which BiGym task to scene-load. Either a known key "
            f"({', '.join(sorted(TASK_MAP))}) or a fully-qualified "
            "'module.path:ClassName' string."
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the registered task keys and exit.",
    )
    parser.add_argument(
        "--spawn",
        default="alternate",
        choices=["walk_in", "in_place", "patrol", "alternate"],
        help="Force walk-in (APPROACH_LOITER_DEPART), already-in-place "
        "(STATIONARY), patrol (COWORKER_PATROL — walks in, moves away "
        "and returns from a different angle a couple of times), or let "
        "the sampler pick.",
    )
    parser.add_argument(
        "--arm",
        default="auto",
        choices=["left", "right", "auto"],
        help="Force one arm to be the active reach arm.",
    )
    parser.add_argument(
        "--reach-target",
        default="alternate",
        choices=["ee", "task", "alternate"],
        help="Always reach for EE, always for task object, or alternate "
        "between them across cycles.",
    )
    parser.add_argument(
        "--human",
        default="g1",
        choices=["smplh", "g1"],
        help="Coworker humanoid model (g1 skips AMASS).",
    )
    parser.add_argument(
        "--smplh-motion",
        default="amass",
        choices=["amass", "procedural"],
        help="SMPL-H body motion: amass clip playback or procedural (G1-style).",
    )
    parser.add_argument(
        "--stage",
        default="default",
        choices=list(_STAGE_BANDS),
        help="Curriculum disruption band (train = stage-2 coworker_train knobs).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.list_tasks:
        print("Registered task keys (pass via --task):")
        for key in sorted(TASK_MAP):
            print(f"  {key:<22s}  {TASK_MAP[key]}")
        print(
            "\nOr pass any other BiGym task as 'module.path:ClassName' to --task."
        )
        return

    amass_dir = os.environ.get("AMASS_DATA_DIR")
    if args.human == "smplh" and args.smplh_motion == "amass" and not amass_dir:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it to the CMU AMASS root, e.g.\n"
            "  export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU\n"
            "Or pass --smplh-motion procedural to skip AMASS."
        )

    task_cls = _load_task_cls(args.task)
    print("=" * 60)
    print(f"COWORKER demo  |  task={task_cls.__name__}  human={args.human}  "
          f"smplh_motion={args.smplh_motion}  "
          f"stage={args.stage}  spawn={args.spawn}  "
          f"arm={args.arm}  target={args.reach_target}")
    print("=" * 60)

    clip_paths = (
        ["74/74_01_poses.npz"]
        if args.human == "smplh" and args.smplh_motion == "amass"
        else []
    )
    human_config = HumanConfig(
        human_model=args.human,
        smplh_motion=args.smplh_motion,
        motion_clip_dir=amass_dir if clip_paths else None,
        motion_clip_paths=clip_paths,
    )
    safety_config = SafetyConfig(
        log_violations=False, terminate_on_violation=False,
    )

    sampler = _make_sampler(
        amass_dir if clip_paths else None,
        human_config.motion_clip_paths,
        args.spawn,
        args.arm if args.arm != "auto" else None,
        args.reach_target,
        args.stage,
    )

    env = make_safety_env(
        task_cls=task_cls,
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        safety_config=safety_config,
        human_config=human_config,
        scenario_sampler=sampler,
        inject_human=True,
    )

    # Patch the sampler's sample_scenario to apply our overrides each
    # episode. This is the least intrusive way to push CLI knobs through
    # without touching production code.
    _sample = sampler.sample_scenario

    def _sample_with_overrides(seed):
        scenario = _sample(seed)
        _override_scenario(
            scenario,
            args.spawn,
            args.arm if args.arm != "auto" else None,
            args.reach_target,
        )
        return scenario

    sampler.sample_scenario = _sample_with_overrides  # type: ignore[assignment]

    obs, info = env.reset(seed=args.seed)
    scenario = env._current_scenario
    print(f"scenario.trajectory_type = {scenario.trajectory_type}")
    print(f"scenario.closest_approach = {scenario.closest_approach:.2f} m")
    if scenario.trajectory_type == "COWORKER_PATROL":
        print(f"scenario.patrol_excursions = {scenario.patrol_excursions}")
        print(f"scenario.patrol_away_distance = {scenario.patrol_away_distance:.2f} m")
    print(f"scenario.coworker_active_arm = {scenario.disruption_config.coworker_active_arm}")
    print(f"scenario.coworker_target_mix = {scenario.disruption_config.coworker_target_mix}")

    model = env._mojo.model
    data = env._mojo.data

    print("\nOpening viewer. Press ESC to close.")
    print("-" * 60)

    last_phase: Optional[str] = None
    last_target_kind: Optional[str] = None
    last_traj_phase: Optional[str] = None
    last_reach_gate: Optional[bool] = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            coworker = getattr(env, "_coworker_controller", None)
            traj_phase = env.human_controller.current_phase if env.human_controller else "?"
            if coworker is not None:
                phase = coworker.last_phase
                target_kind = coworker._cycle.target_kind
                reach_gate = coworker.last_out_of_reach
                if (phase != last_phase or target_kind != last_target_kind
                        or traj_phase != last_traj_phase or reach_gate != last_reach_gate):
                    tgt = coworker.last_reach_target
                    tgt_str = (
                        f"({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f})"
                        if tgt is not None
                        else "<none>"
                    )
                    gate_str = "OUT-OF-REACH" if reach_gate else "in_reach"
                    print(
                        f"[t={data.time:5.2f}s] traj={traj_phase:<10s} "
                        f"reach_phase={phase:<8s} "
                        f"target_kind={target_kind:<11s} {gate_str:<12s} "
                        f"pos={tgt_str}"
                    )
                    last_phase = phase
                    last_target_kind = target_kind
                    last_traj_phase = traj_phase
                    last_reach_gate = reach_gate

            if terminated or truncated:
                obs, info = env.reset()
                scenario = env._current_scenario
                print("\n--- new episode ---")
                print(f"  trajectory_type={scenario.trajectory_type} "
                      f"arm={scenario.disruption_config.coworker_active_arm} "
                      f"target_mix={scenario.disruption_config.coworker_target_mix}")
                step = 0

            viewer.sync()
            step += 1

    env.close()


if __name__ == "__main__":
    main()
