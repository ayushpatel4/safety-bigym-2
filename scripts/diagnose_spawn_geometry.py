#!/usr/bin/env python
"""
Diagnose whether the human spawn pose overlaps task/scene geometry at t=0.

For each reset: run mj_forward only (no physics step), then check whether any
human collision geom's AABB overlaps any non-human, non-robot geom's AABB.

Writes a CSV row per reset; prints a short summary.

Usage:
    export AMASS_DATA_DIR=/path/to/CMU/CMU
    python scripts/diagnose_spawn_geometry.py --episodes 50
"""

import argparse
import csv
import os
from pathlib import Path

import mujoco
import numpy as np

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.envs.dishwasher import DishwasherClose
from safety_bigym import HumanConfig, SafetyConfig, make_safety_env


CSV_FIELDS = [
    "episode",
    "approach_angle_deg",
    "spawn_distance_m",
    "trajectory_type",
    "spawn_penetrating",
    "overlap_count",
    "worst_overlap_pair",
    "worst_overlap_depth_m",
]


def geom_aabb_world(model, data, gid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame AABB (lo, hi) for a geom, expanded by its own size."""
    pos = data.geom_xpos[gid]
    # model.geom_aabb is (ngeom, 6): (center_xyz, halfsize_xyz) in body frame
    half = model.geom_aabb[gid, 3:6]
    # Rotate the half-sizes conservatively by the absolute rotation matrix
    R = data.geom_xmat[gid].reshape(3, 3)
    half_world = np.abs(R) @ half
    return pos - half_world, pos + half_world


def aabb_overlap_depth(lo_a, hi_a, lo_b, hi_b) -> float:
    """Return positive overlap depth along the most-overlapping axis, else 0."""
    overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
    if np.all(overlap > 0):
        return float(np.min(overlap))
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/diagnose/spawn_overlap.csv"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cmu = os.environ.get("AMASS_DATA_DIR")
    if not cmu:
        raise RuntimeError("AMASS_DATA_DIR is not set. Export the CMU AMASS root.")

    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    env = make_safety_env(
        task_cls=DishwasherClose,
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False, terminate_on_violation=False),
        human_config=HumanConfig(
            motion_clip_dir=cmu, motion_clip_paths=["74/74_01_poses.npz"]
        ),
        inject_human=True,
        control_frequency=CONTROL_FREQUENCY_MAX // 20,
    )

    model = env._mojo.model
    data = env._mojo.data

    human_geoms = []
    scene_geoms = []
    name_of = {}
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        name_of[gid] = nm
        if nm.endswith("_col"):
            human_geoms.append(gid)
        elif not env._is_robot_geom(nm):
            # Skip floor if desired; floor overlap is not the problem we care about.
            if "floor" in nm.lower():
                continue
            scene_geoms.append(gid)

    rows = []
    try:
        for ep in range(args.episodes):
            _, info = env.reset()
            mujoco.mj_forward(model, data)
            scenario = info.get("scenario", {}) or {}

            overlap_count = 0
            worst_pair = ""
            worst_depth = 0.0
            for hg in human_geoms:
                lo_h, hi_h = geom_aabb_world(model, data, hg)
                for sg in scene_geoms:
                    lo_s, hi_s = geom_aabb_world(model, data, sg)
                    d = aabb_overlap_depth(lo_h, hi_h, lo_s, hi_s)
                    if d > 0:
                        overlap_count += 1
                        if d > worst_depth:
                            worst_depth = d
                            worst_pair = f"{name_of[hg]}|{name_of[sg]}"

            row = {
                "episode": ep,
                "approach_angle_deg": scenario.get("approach_angle"),
                "spawn_distance_m": scenario.get("spawn_distance"),
                "trajectory_type": scenario.get("trajectory_type"),
                "spawn_penetrating": overlap_count > 0,
                "overlap_count": overlap_count,
                "worst_overlap_pair": worst_pair,
                "worst_overlap_depth_m": round(worst_depth, 4),
            }
            rows.append(row)
            print(
                f"[ep {ep:02d}] overlaps={overlap_count:3d} "
                f"worst={worst_pair} depth={worst_depth:.3f}m"
            )
    finally:
        env.close()

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    pen_rate = np.mean([r["spawn_penetrating"] for r in rows])
    print(f"\nspawn-penetration rate: {pen_rate:.2%}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
