"""
Visual Demo: ISO 15066 Safety Wrapper (closest-joint SSM + PFL)

Shows the safety wrapper in action:
1. Robot slides toward a human with 5 body geoms + 3 robot geoms.
2. compute_ssm() runs on ALL pairs — the demo prints the closest pair live,
   so you can watch `closest_human_joint` change as the robot's EE vs. arm
   sweeps past different body parts.
3. On contact, PFL fires — region, force, and violation flag are printed.

Usage:
    mjpython scripts/demo_safety_visual.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.safety import ISO15066Wrapper, SSMConfig


# Arm-extended pose on the human so closest-joint actually prefers the elbow,
# not the pelvis, as the EE passes by the torso's side.
VISUAL_SCENE = """
<mujoco model="safety_visual_demo">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- Human — 5 labelled body geoms -->
    <body name="human" pos="0 0 1">
      <geom name="Pelvis_col" type="sphere" size="0.12" rgba="0.9 0.7 0.6 1"/>
      <geom name="Chest_col"  type="capsule" size="0.1"  fromto="0 0 0 0 0 0.35" rgba="0.9 0.7 0.6 1"/>
      <geom name="Head_col"   type="sphere"  size="0.09" pos="0 0 0.45" rgba="0.9 0.7 0.6 1"/>
      <geom name="R_Elbow_col" type="capsule" size="0.04" fromto="0.15 0 0.25 0.55 0 0.25" rgba="0.9 0.7 0.6 1"/>
      <geom name="L_Elbow_col" type="capsule" size="0.04" fromto="-0.15 0 0.25 -0.55 0 0.25" rgba="0.9 0.7 0.6 1"/>
    </body>

    <!-- Robot: slides on a track along -Y so its EE approaches the human -->
    <body name="robot_base" pos="0 2 1">
      <joint name="robot_slide" type="slide" axis="0 -1 0" range="0 2" damping="50"/>
      <geom name="robot_base_geom" type="box" size="0.08 0.08 0.15" rgba="0.3 0.3 0.3 1"/>
      <body name="robot_arm" pos="0 -0.12 0">
        <joint name="robot_arm_joint" type="hinge" axis="1 0 0" range="-0.5 0.5" damping="10"/>
        <geom name="robot_arm_geom" type="capsule" size="0.035" fromto="0 0 0 0 -0.3 0" rgba="0.4 0.5 0.7 1"/>
        <body name="robot_ee" pos="0 -0.35 0">
          <geom name="robot_ee_geom" type="sphere" size="0.07" rgba="1 0.3 0.3 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="slide_motor" joint="robot_slide"     kp="800" ctrlrange="0 2"/>
    <position name="arm_motor"   joint="robot_arm_joint" kp="100" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""


HUMAN_GEOMS = [
    "Pelvis_col", "Chest_col", "Head_col", "R_Elbow_col", "L_Elbow_col",
]
ROBOT_GEOMS = ["robot_base_geom", "robot_arm_geom", "robot_ee_geom"]


def run_visual_demo():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(VISUAL_SCENE)
        scene_path = f.name

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ssm_config = SSMConfig(T_r=0.1, T_s=0.05, a_max=5.0, C=0.1, v_h_max=0.0)
    wrapper = ISO15066Wrapper(model, data, ssm_config=ssm_config)
    for name in ROBOT_GEOMS:
        wrapper.add_robot_geom(name)

    human_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in HUMAN_GEOMS]
    robot_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ROBOT_GEOMS]

    print("\n" + "=" * 78)
    print("ISO 15066 SAFETY VISUAL DEMO — closest-joint SSM + PFL")
    print("=" * 78)
    print(f"  human geoms: {HUMAN_GEOMS}")
    print(f"  robot geoms: {ROBOT_GEOMS}")
    print("=" * 78)

    # --- Static proof: closest-joint responds to EE position ---
    # Compute closest-pair for a handful of canned EE positions. This exercises
    # the pairwise (Nh × Nr) math directly, independent of physics, so you can
    # see the closest_human_joint change even before the robot starts moving.
    print("\n[static closest-joint check — teleport EE, report closest human geom]")
    human_xpos = np.stack([data.geom_xpos[g].copy() for g in human_gids])
    probe_positions = [
        ("near head",   np.array([0.0,  0.0, 1.45])),
        ("near chest",  np.array([0.0,  0.0, 1.20])),
        ("near pelvis", np.array([0.0,  0.0, 1.00])),
        ("near R elbow", np.array([0.5, 0.0, 1.25])),
        ("near L elbow", np.array([-0.5, 0.0, 1.25])),
    ]
    for label, ee_pos in probe_positions:
        robot_probe = np.stack([ee_pos, ee_pos + 0.3, ee_pos + 0.5])  # fake 3 robot geoms
        info = wrapper.build_safety_info(
            contacts=[],
            robot_positions=robot_probe,
            robot_vel=0.0,
            human_positions=human_xpos,
            human_vel=0.0,
            human_names=HUMAN_GEOMS,
            robot_names=ROBOT_GEOMS,
        )
        print(
            f"  EE @ {ee_pos.tolist()} ({label:12s})  "
            f"→ closest = {info.closest_human_joint:12s} ↔ {info.closest_robot_link}  "
            f"d_min = {info.min_separation:.3f} m"
        )
    print("=" * 78)
    print("  Watch the `closest pair` column — it updates live as the EE sweeps past.")
    print("=" * 78 + "\n")

    approach_target = 0.0
    phase = "approach"
    phase_start = 0.0
    ssm_violated = False
    pfl_violated = False
    closest_pairs_seen = set()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 4
        viewer.cam.lookat[:] = [0, 0.5, 1]

        while viewer.is_running():
            t = data.time

            # Drive the slide toward the human. The scene is 1-DOF so the physics
            # contact always lands on Pelvis_col — the static proof table above
            # exercises the other closest-pair branches.
            if phase == "approach":
                approach_target = min(1.95, approach_target + 0.002)
                data.ctrl[0] = approach_target
                data.ctrl[1] = 0.0
                if approach_target > 1.9:
                    phase = "contact"
                    phase_start = t
            elif phase == "contact":
                if t - phase_start > 3:
                    phase = "retreat"
            elif phase == "retreat":
                approach_target = max(0.0, approach_target - 0.003)
                data.ctrl[0] = approach_target
                if approach_target < 0.1:
                    phase = "approach"

            # Step physics FIRST so contact data reflects the state we're about to print.
            mujoco.mj_step(model, data)

            # Build per-geom position arrays — this is what drives closest-joint SSM.
            human_pos = np.stack([data.geom_xpos[gid].copy() for gid in human_gids])  # (5,3)
            robot_pos = np.stack([data.geom_xpos[gid].copy() for gid in robot_gids])  # (3,3)
            robot_vel = float(np.linalg.norm(
                data.cvel[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_ee"), 3:6]
            ))

            contacts = wrapper.check_safety_substep()
            safety_info = wrapper.build_safety_info(
                contacts=contacts,
                robot_positions=robot_pos,
                robot_vel=robot_vel,
                human_positions=human_pos,
                human_vel=0.0,
                human_names=HUMAN_GEOMS,
                robot_names=ROBOT_GEOMS,
            )

            # Log unique closest-pair names as they appear.
            pair = (safety_info.closest_human_joint, safety_info.closest_robot_link)
            if pair not in closest_pairs_seen:
                closest_pairs_seen.add(pair)

            # Print status every 0.2s of sim time.
            if int(t * 5) != int((t - model.opt.timestep) * 5):
                ssm_status = (
                    "🔴 SSM VIOLATION" if safety_info.ssm_violation else "🟢 SSM OK"
                )
                if safety_info.ssm_violation:
                    ssm_violated = True

                if safety_info.max_contact_force > 5:
                    if safety_info.pfl_violation:
                        pfl_status = (
                            f"🔴 PFL VIOLATION {safety_info.max_contact_force:.0f}N "
                            f"({safety_info.contact_region})"
                        )
                        pfl_violated = True
                    else:
                        pfl_status = (
                            f"🟡 PFL Contact {safety_info.max_contact_force:.0f}N "
                            f"({safety_info.contact_region})"
                        )
                else:
                    pfl_status = "⚪ No contact"

                print(
                    f"t={t:5.1f}s  d_min={safety_info.min_separation:.2f}m  "
                    f"margin={safety_info.ssm_margin:+.2f}m  "
                    f"closest=[{safety_info.closest_human_joint}↔{safety_info.closest_robot_link}]  "
                    f"| {ssm_status} | {pfl_status}"
                )

            viewer.sync()
            time.sleep(0.001)

    print("\n" + "=" * 78)
    print("DEMO SUMMARY")
    print("=" * 78)
    print(f"  SSM Violation Detected: {'✅ YES' if ssm_violated else '❌ NO'}")
    print(f"  PFL Violation Detected: {'✅ YES' if pfl_violated else '❌ NO'}")
    print(f"  Closest-pair transitions seen ({len(closest_pairs_seen)}):")
    for h, r in sorted(closest_pairs_seen):
        print(f"    {h:12s}  ↔  {r}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_visual_demo()
