#!/usr/bin/env python
"""
Diagnose which geom pairs are colliding during DP rollouts.

Classifies every MuJoCo contact each step into {human<->robot, human<->scene,
robot<->scene, other}. Writes a CSV of per-episode aggregates and prints the
top offending geom pairs to stdout.

Usage:
    export AMASS_DATA_DIR=/path/to/CMU/CMU
    python scripts/diagnose_contacts.py --episodes 10 --max-steps 60
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.envs.dishwasher import DishwasherClose
from safety_bigym import HumanConfig, SafetyConfig, make_safety_env


CSV_FIELDS = [
    "episode",
    "end_step",
    "truncated",
    "human_robot_contacts",
    "human_scene_contacts",
    "robot_scene_contacts",
    "other_contacts",
    "max_penetration_m",
    "max_contact_force_n",
    "worst_human_scene_pair",
]


def classify(name: str, human_geoms: set, robot_geoms: set) -> str:
    if name in human_geoms:
        return "human"
    if name in robot_geoms:
        return "robot"
    return "scene"


def run_episode(env, max_steps: int) -> dict:
    model = env._mojo.model
    data = env._mojo.data

    human_geom_ids = set()
    robot_geom_ids = set()
    name_of = {}
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm is None:
            nm = f"geom_{gid}"
        name_of[gid] = nm
        if nm.endswith("_col"):
            human_geom_ids.add(gid)
        elif env._is_robot_geom(nm):
            robot_geom_ids.add(gid)

    _, _ = env.reset()
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    counters = {
        "human_robot": 0,
        "human_scene": 0,
        "robot_scene": 0,
        "other": 0,
    }
    scene_pair_counter: Counter = Counter()
    max_pen = 0.0
    max_force = 0.0
    force_buf = np.zeros(6, dtype=np.float64)

    end_step = max_steps
    terminated = False
    truncated = False

    for step in range(max_steps):
        _, _, terminated, truncated, _ = env.step(zero_action)

        ncon = data.ncon
        for i in range(ncon):
            c = data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            n1 = name_of.get(g1, f"geom_{g1}")
            n2 = name_of.get(g2, f"geom_{g2}")
            k1 = "human" if g1 in human_geom_ids else "robot" if g1 in robot_geom_ids else "scene"
            k2 = "human" if g2 in human_geom_ids else "robot" if g2 in robot_geom_ids else "scene"
            pair = tuple(sorted([k1, k2]))
            if pair == ("human", "robot"):
                counters["human_robot"] += 1
            elif pair == ("human", "scene"):
                counters["human_scene"] += 1
                key = tuple(sorted([n1, n2]))
                scene_pair_counter[key] += 1
            elif pair == ("robot", "scene"):
                counters["robot_scene"] += 1
            else:
                counters["other"] += 1

            # contact.dist is penetration depth when negative
            if c.dist < -max_pen:
                max_pen = -float(c.dist)

            mujoco.mj_contactForce(model, data, i, force_buf)
            fmag = float(np.linalg.norm(force_buf[:3]))
            if fmag > max_force:
                max_force = fmag

        if terminated or truncated:
            end_step = step + 1
            break

    worst_pair = scene_pair_counter.most_common(1)
    worst = f"{worst_pair[0][0][0]}|{worst_pair[0][0][1]}:{worst_pair[0][1]}" if worst_pair else ""

    return {
        "end_step": end_step,
        "truncated": bool(truncated),
        "human_robot_contacts": counters["human_robot"],
        "human_scene_contacts": counters["human_scene"],
        "robot_scene_contacts": counters["robot_scene"],
        "other_contacts": counters["other"],
        "max_penetration_m": round(max_pen, 5),
        "max_contact_force_n": round(max_force, 2),
        "worst_human_scene_pair": worst,
        "_scene_pairs": scene_pair_counter,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/diagnose/contacts.csv"),
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

    rows = []
    total_scene_pairs: Counter = Counter()
    try:
        for ep in range(args.episodes):
            res = run_episode(env, args.max_steps)
            total_scene_pairs.update(res.pop("_scene_pairs"))
            row = {"episode": ep, **res}
            rows.append(row)
            print(
                f"[ep {ep:02d}] end={row['end_step']:3d} "
                f"h-r={row['human_robot_contacts']:4d} "
                f"h-s={row['human_scene_contacts']:4d} "
                f"max_pen={row['max_penetration_m']:.3f}m "
                f"max_F={row['max_contact_force_n']:.0f}N "
                f"worst={row['worst_human_scene_pair']}"
            )
    finally:
        env.close()

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Top 10 human<->scene pairs across all episodes ===")
    for (a, b), n in total_scene_pairs.most_common(10):
        print(f"  {n:5d}  {a}  <->  {b}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
