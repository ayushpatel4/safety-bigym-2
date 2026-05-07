#!/usr/bin/env python
"""Diagnose: are contact forces being captured during CONTACT scenarios?

Phase 1 ablation produced ep_pfl_violation_rate=0 / max_contact_force=0 even
with ssm_margin = -13 m. Two hypotheses:

    H1 (filter):  data.ncon == 0 during overlap. Bit-1 promotion or
                  human contype/conaffinity wrong; MuJoCo never generates
                  the human<->robot contact pairs.
    H2 (timing):  data.ncon > 0 but mj_contactForce returns 0. Most likely
                  caused by qpos teleport in human_controller.step
                  resolving the penetration outside the constraint solve
                  that mj_rnePostConstraint reads from.

This script forces a CONTACT episode, monkey-patches the safety wrapper to
log every substep, and prints whatever it finds. ~30 steps is enough.

Usage:
    AMASS_DATA_DIR=/path/to/CMU/CMU \\
    mjpython scripts/diagnose_contact_forces.py [--steps 30]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTargetSingle
from bigym.bigym_env import CONTROL_FREQUENCY_MAX

from safety_bigym import HumanConfig, SafetyConfig, make_safety_env
from safety_bigym.scenarios import (
    DisruptionType,
    ParameterSpace,
    ScenarioSampler,
)


def _human_robot_label(g1_name: str, g2_name: str, wrapper) -> str:
    g1id = mujoco.mj_name2id(wrapper.model, mujoco.mjtObj.mjOBJ_GEOM, g1_name) if g1_name else -1
    g2id = mujoco.mj_name2id(wrapper.model, mujoco.mjtObj.mjOBJ_GEOM, g2_name) if g2_name else -1
    g1_h = g1id in wrapper.human_geoms
    g2_h = g2id in wrapper.human_geoms
    g1_r = g1id in wrapper.robot_geoms
    g2_r = g2id in wrapper.robot_geoms
    g1_f = g1id in wrapper.fixture_geoms
    g2_f = g2id in wrapper.fixture_geoms
    if (g1_h and g2_r) or (g2_h and g1_r):
        return "HUMAN<->ROBOT"
    if (g1_h and g2_f) or (g2_h and g1_f):
        return "HUMAN<->FIXTURE"
    if g1_h or g2_h:
        return "HUMAN<->OTHER"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="reach", choices=["reach", "dishwasher_close"])
    args = parser.parse_args()

    if not os.environ.get("AMASS_DATA_DIR"):
        print("AMASS_DATA_DIR not set. Export it before running.", file=sys.stderr)
        sys.exit(1)

    if args.task == "reach":
        from bigym.envs.reach_target import ReachTargetSingle as task_cls
    else:
        from bigym.envs.dishwasher import DishwasherClose as task_cls

    action_mode = JointPositionActionMode(floating_base=True, absolute=True)
    human_config = HumanConfig(
        motion_clip_dir=os.environ["AMASS_DATA_DIR"],
        motion_clip_paths=[
            "74/74_01_poses.npz",
            "74/74_02_poses.npz",
            "09/09_01_poses.npz",
        ],
    )

    # Force CONTACT only; this is the most aggressive scenario and is
    # guaranteed to drive the human into the robot.
    forced_sampler = ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=human_config.motion_clip_paths,
            disruption_weights={DisruptionType.CONTACT: 1.0},
        ),
        motion_dir=os.environ["AMASS_DATA_DIR"],
    )

    env = make_safety_env(
        task_cls=task_cls,
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False, terminate_on_violation=False),
        human_config=human_config,
        scenario_sampler=forced_sampler,
        inject_human=True,
        control_frequency=CONTROL_FREQUENCY_MAX // 20,
    )

    obs, info = env.reset(seed=args.seed)
    wrapper = env.safety_wrapper
    model = wrapper.model
    data = wrapper.data

    print(f"task={task_cls.__name__}  steps={args.steps}  seed={args.seed}")
    print(f"human_geoms={len(wrapper.human_geoms)}  robot_geoms={len(wrapper.robot_geoms)}  "
          f"fixture_geoms={len(wrapper.fixture_geoms)}")
    if not wrapper.human_geoms:
        print("!! No human geoms detected in wrapper. Filter is misconfigured at registration time.")
    sample_human = sorted(wrapper.human_geoms)[:5]
    print("  sample human_geoms:",
          [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sample_human])
    sample_robot = sorted(wrapper.robot_geoms)[:5]
    print("  sample robot_geoms:",
          [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sample_robot])

    # Inspect collision-bit setup on a representative human geom + robot geom
    if sample_human and sample_robot:
        hg, rg = sample_human[0], sample_robot[0]
        h_t, h_a = int(model.geom_contype[hg]), int(model.geom_conaffinity[hg])
        r_t, r_a = int(model.geom_contype[rg]), int(model.geom_conaffinity[rg])
        cross = (h_t & r_a) | (r_t & h_a)
        print(f"  human geom {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hg)}: "
              f"contype={h_t:b} conaffinity={h_a:b}")
        print(f"  robot geom {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, rg)}: "
              f"contype={r_t:b} conaffinity={r_a:b}")
        print(f"  collision-rule cross = {cross:b}  ({'pair eligible' if cross else 'NO COLLISION POSSIBLE'})")

    # --- monkey-patch check_safety_substep so we see every substep ---
    substep_log: list[dict] = []
    orig = wrapper.check_safety_substep

    def _logged_check_safety_substep():
        contacts = orig()
        ncon = int(data.ncon)
        seen = []
        force_buf = np.zeros(6)
        for i in range(ncon):
            c = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or f"geom{c.geom1}"
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or f"geom{c.geom2}"
            mujoco.mj_contactForce(model, data, i, force_buf)
            fmag = float(np.linalg.norm(force_buf[:3]))
            kind = _human_robot_label(g1, g2, wrapper)
            seen.append((kind, g1, g2, fmag, float(c.dist)))
        substep_log.append({"ncon": ncon, "contacts": seen,
                             "wrapper_contacts": len(contacts)})
        return contacts

    wrapper.check_safety_substep = _logged_check_safety_substep

    print(f"\n--- scenario: {info.get('scenario', {})} ---\n")

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for step in range(args.steps):
        substep_log.clear()
        _, _, term, trunc, info = env.step(zero_action)
        safety = info.get("safety", {})

        # Per-step summary
        max_ncon = max((s["ncon"] for s in substep_log), default=0)
        any_hr_contact = any(
            kind.startswith("HUMAN<->ROBOT")
            for s in substep_log for kind, *_ in s["contacts"]
        )
        max_force = max(
            (fmag for s in substep_log for kind, _, _, fmag, _ in s["contacts"]
             if kind.startswith("HUMAN<->ROBOT")),
            default=0.0,
        )
        sep = safety.get("min_separation", float("nan"))
        ratio = safety.get("pfl_force_ratio", 0.0)
        margin = safety.get("ssm_margin", float("nan"))
        phase = info.get("human_phase", "?")

        print(f"step {step:>2d}  phase={phase:<8s}  sep={sep:>5.2f}m  ssm_margin={margin:>+6.2f}m  "
              f"pfl_ratio={ratio:>5.3f}  max_ncon={max_ncon:>2d}  "
              f"hr_contact={int(any_hr_contact)}  max_F={max_force:>6.1f}N")

        # If anything interesting happened, drill in
        if any_hr_contact or max_ncon > 0:
            for si, s in enumerate(substep_log):
                if not s["contacts"]:
                    continue
                interesting = [(k, g1, g2, f, d) for (k, g1, g2, f, d) in s["contacts"]
                               if k.startswith("HUMAN")]
                if not interesting:
                    continue
                print(f"     substep {si}: ncon={s['ncon']}  wrapper_contacts={s['wrapper_contacts']}")
                for kind, g1, g2, fmag, dist in interesting[:6]:
                    print(f"       {kind:<16s}  {g1:<22s} <-> {g2:<22s}  "
                          f"F={fmag:>7.2f}N  dist={dist:+.4f}m")

        if term or trunc:
            print("episode ended early; resetting")
            obs, info = env.reset(seed=args.seed + step)

    env.close()
    print("\n--- diagnostic complete ---")
    print("Interpretation:")
    print("  * If max_ncon never includes HUMAN<->ROBOT, the collision filter is wrong.")
    print("  * If HUMAN<->ROBOT contacts appear with F=0, mj_contactForce isn't reading them")
    print("    (likely qpos-teleport bypassing the constraint solve, or wrong API).")
    print("  * If contacts appear with F>0 but pfl_ratio=0, the wrapper-side filter is wrong.")


if __name__ == "__main__":
    main()
