#!/usr/bin/env python
"""
Diagnose why DP eval episodes are ~1 second long.

Runs N resets x up to max_steps no-op rollouts on DishwasherClose, captures:
- step at which the episode terminates/truncates
- whether UnstableSimulationWarning was raised (physics error)
- the scenario parameters used (approach angle, spawn distance, trajectory type)

Writes one row per episode to the CSV given by --out.

Usage:
    export AMASS_DATA_DIR=/path/to/CMU/CMU
    python scripts/diagnose_truncation.py --episodes 30 --max-steps 150
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

import numpy as np

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.envs.dishwasher import DishwasherClose
from bigym.utils.env_health import UnstableSimulationWarning
from safety_bigym import HumanConfig, SafetyConfig, make_safety_env


CSV_FIELDS = [
    "episode",
    "end_step",
    "terminated",
    "truncated",
    "physics_error_seen",
    "physics_warning_count",
    "approach_angle_deg",
    "spawn_distance_m",
    "trajectory_type",
    "disruption_type",
    "clip_path",
]


def build_env(inject_human: bool = True, demo_down_sample_rate: int = 20):
    """Build env matching SafetyBiGymEnvFactory training settings.

    demo_down_sample_rate=20 gives control_frequency=25 Hz and sub_steps=20,
    mirroring what the DP training pipeline uses. Lower values increase the
    control loop rate and reduce physics sub-stepping per env.step().
    """
    cmu = os.environ.get("AMASS_DATA_DIR")
    if not cmu:
        raise RuntimeError("AMASS_DATA_DIR is not set. Export the CMU AMASS root.")
    human_config = HumanConfig(
        motion_clip_dir=cmu,
        motion_clip_paths=["74/74_01_poses.npz"],
    )
    # Match enable_all_floating_dof=true in dishwasher_close.yaml
    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    return make_safety_env(
        task_cls=DishwasherClose,
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False, terminate_on_violation=False),
        human_config=human_config,
        inject_human=inject_human,
        control_frequency=CONTROL_FREQUENCY_MAX // demo_down_sample_rate,
    )


def run_episode(env, max_steps: int, action_mode: str = "zero", rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UnstableSimulationWarning)

        _, info = env.reset()
        scenario = info.get("scenario", {}) or {}
        current_scenario = getattr(env, "_current_scenario", None)

        low = env.action_space.low
        high = env.action_space.high
        zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        end_step = max_steps
        terminated = False
        truncated = False

        for step in range(max_steps):
            if action_mode == "zero":
                action = zero_action
            elif action_mode == "random":
                action = rng.uniform(low, high).astype(env.action_space.dtype)
            elif action_mode == "random_small":
                # ±10% of the action range around zero — more representative of what a DP may output
                mid = 0.5 * (high + low)
                span = 0.1 * (high - low)
                action = (mid + rng.uniform(-1.0, 1.0, size=low.shape) * span).astype(env.action_space.dtype)
            else:
                raise ValueError(f"unknown action_mode: {action_mode}")
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                end_step = step + 1
                break

        phys_warnings = [w for w in caught if issubclass(w.category, UnstableSimulationWarning)]

    approach_angle = getattr(current_scenario, "approach_angle", None) if current_scenario else None
    spawn_distance = getattr(current_scenario, "spawn_distance", None) if current_scenario else None
    return {
        "end_step": end_step,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "physics_error_seen": len(phys_warnings) > 0,
        "physics_warning_count": len(phys_warnings),
        "approach_angle_deg": approach_angle,
        "spawn_distance_m": spawn_distance,
        "trajectory_type": scenario.get("trajectory_type"),
        "disruption_type": scenario.get("disruption_type"),
        "clip_path": scenario.get("clip_path"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/diagnose/truncation.csv"),
        help="Output CSV path (parent dirs created)",
    )
    parser.add_argument(
        "--no-human", action="store_true",
        help="Run without injecting the human (baseline)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode", choices=("zero", "random", "random_small"),
        default="zero", help="Action profile per step (default: zero)",
    )
    parser.add_argument(
        "--demo-down-sample-rate", type=int, default=20,
        help="Must match cfg.env.demo_down_sample_rate used at training time",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(inject_human=not args.no_human, demo_down_sample_rate=args.demo_down_sample_rate)

    rows = []
    try:
        for ep in range(args.episodes):
            row = {"episode": ep, **run_episode(env, args.max_steps, args.action_mode, rng)}
            rows.append(row)
            print(
                f"[ep {ep:02d}] end_step={row['end_step']:3d} "
                f"trunc={row['truncated']} term={row['terminated']} "
                f"phys_err={row['physics_error_seen']} "
                f"traj={row['trajectory_type']} "
                f"angle={row['approach_angle_deg']} dist={row['spawn_distance_m']}"
            )
    finally:
        env.close()

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    end_steps = np.array([r["end_step"] for r in rows])
    phys = np.array([r["physics_error_seen"] for r in rows])
    trunc = np.array([r["truncated"] for r in rows])
    print("\n=== Summary ===")
    print(f"episodes: {len(rows)}")
    print(f"median end_step: {int(np.median(end_steps))} / {args.max_steps}")
    print(f"mean   end_step: {end_steps.mean():.1f}")
    print(f"truncated rate:  {trunc.mean():.2%}")
    print(f"physics errors:  {phys.mean():.2%}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
