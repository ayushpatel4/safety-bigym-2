#!/usr/bin/env python
"""
Demo: Trajectory Planner in 3D MuJoCo Viewer

Shows a simple humanoid following each trajectory type in sequence:
1. PASS_BY — walks past the robot with a lateral offset
2. APPROACH_LOITER_DEPART — walks to robot, pauses, walks away
3. ARC — curves around the robot workspace

The "robot" is a static orange box at the origin.
The "human" is a capsule body that follows the planner's output.

Usage:
    mjpython scripts/demo_trajectory_3d.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import tempfile
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.human.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryConfig,
    TrajectoryType,
)


# Minimal MuJoCo scene with a movable "human" capsule and a static "robot" box
SCENE_XML = """
<mujoco model="trajectory_demo">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" castshadow="true"/>
    
    <!-- Ground with grid -->
    <geom name="ground" type="plane" size="6 6 0.01" rgba="0.95 0.95 0.95 1"
          material="grid"/>
    
    <!-- Robot (static orange box at origin) -->
    <body name="robot" pos="0 0 0.5">
      <geom name="robot_body" type="box" size="0.15 0.15 0.5" rgba="1.0 0.6 0.1 0.9"/>
      <geom name="robot_head" type="sphere" size="0.12" pos="0 0 0.55" rgba="1.0 0.6 0.1 0.9"/>
    </body>
    
    <!-- Safety radius marker (1m circle on ground) -->
    <body name="safety_ring" pos="0 0 0.001">
      <geom type="cylinder" size="1.0 0.002" rgba="1.0 0.3 0.3 0.2"/>
    </body>
    
    <!-- Human (moveable capsule body with freejoint) -->
    <body name="human" pos="3 0 0.9">
      <freejoint name="human_root"/>
      <!-- Torso -->
      <geom name="human_torso" type="capsule" size="0.12" fromto="0 0 -0.3 0 0 0.25" 
            rgba="0.4 0.7 1.0 0.9" mass="40"/>
      <!-- Head -->
      <geom name="human_head" type="sphere" size="0.1" pos="0 0 0.35" 
            rgba="0.9 0.75 0.65 1" mass="5"/>
      <!-- Left arm -->
      <geom name="human_larm" type="capsule" size="0.04" fromto="-0.15 0 0.15 -0.4 0 0.0" 
            rgba="0.4 0.7 1.0 0.9" mass="3"/>
      <!-- Right arm -->
      <geom name="human_rarm" type="capsule" size="0.04" fromto="0.15 0 0.15 0.4 0 0.0" 
            rgba="0.4 0.7 1.0 0.9" mass="3"/>
      <!-- Left leg -->
      <geom name="human_lleg" type="capsule" size="0.06" fromto="-0.08 0 -0.3 -0.08 0 -0.85" 
            rgba="0.3 0.5 0.8 0.9" mass="8"/>
      <!-- Right leg -->
      <geom name="human_rleg" type="capsule" size="0.06" fromto="0.08 0 -0.3 0.08 0 -0.85" 
            rgba="0.3 0.5 0.8 0.9" mass="8"/>
      <!-- Direction indicator (nose cone) -->
      <geom name="human_nose" type="sphere" size="0.04" pos="0 -0.13 0.35" 
            rgba="1 0.2 0.2 1"/>
    </body>
    
    <!-- Trail markers (will be positioned dynamically) -->
  </worldbody>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.9 0.9 0.9" rgb2="0.85 0.85 0.85"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
</mujoco>
"""

# Phase colors for console output
PHASE_EMOJI = {
    "walk": "🚶",
    "approach": "➡️ ",
    "loiter": "🔴",
    "depart": "🔙",
}


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Quaternion [w, x, y, z] for Z-axis rotation."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def run_trajectory(viewer, model, data, planner, name, human_body_id):
    """Run a single trajectory, moving the human body."""
    
    print(f"\n{'━'*60}")
    print(f"  {PHASE_EMOJI.get('walk', '')} {name}")
    print(f"  Duration: {planner.duration:.1f}s | "
          f"Min dist: {planner.closest_distance_to_robot():.2f}m")
    print(f"{'━'*60}")
    
    dt = model.opt.timestep
    t = 0.0
    prev_phase = None
    step = 0
    
    # Find human root qpos start (freejoint: 3 pos + 4 quat)
    human_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "human_root")
    qpos_start = model.jnt_qposadr[human_joint_id]
    
    while t <= planner.duration + 0.5 and viewer.is_running():
        x, y, yaw, phase = planner.get_pose(t)
        
        # Set human position (XY from planner, Z fixed for standing height)
        data.qpos[qpos_start + 0] = x
        data.qpos[qpos_start + 1] = y
        data.qpos[qpos_start + 2] = 0.9  # Standing pelvis height
        
        # Set human orientation (quaternion from yaw)
        quat = quat_from_yaw(yaw)
        data.qpos[qpos_start + 3:qpos_start + 7] = quat
        
        # Zero velocity (we're kinematically controlling)
        data.qvel[:] = 0
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        # Print phase transitions
        if phase != prev_phase:
            emoji = PHASE_EMOJI.get(phase, "  ")
            dist = np.sqrt(x**2 + y**2)
            print(f"  {emoji} t={t:5.1f}s  Phase: {phase:10s}  "
                  f"Pos: ({x:+5.2f}, {y:+5.2f})  Dist: {dist:.2f}m")
            prev_phase = phase
        
        # Print position periodically
        elif step % 200 == 0:
            dist = np.sqrt(x**2 + y**2)
            emoji = PHASE_EMOJI.get(phase, "  ")
            print(f"  {emoji} t={t:5.1f}s  {phase:10s}  "
                  f"Pos: ({x:+5.2f}, {y:+5.2f})  Dist: {dist:.2f}m")
        
        viewer.sync()
        time.sleep(dt)  # Real-time
        t += dt
        step += 1
    
    # Brief pause between trajectories
    time.sleep(0.5)


def main():
    robot_pos = np.array([0.0, 0.0])
    spawn_pos = np.array([3.0, 0.0])
    
    # Define trajectory configs
    trajectories = [
        ("1. PASS_BY — Walk past robot with 1m offset", TrajectoryConfig(
            trajectory_type=TrajectoryType.PASS_BY,
            robot_pos=robot_pos,
            spawn_pos=spawn_pos,
            approach_yaw=np.pi,
            pass_by_offset=1.0,
            pass_by_side=1,
            walk_speed=1.2,
        )),
        ("2. APPROACH_LOITER_DEPART — Approach, pause 2s, leave", TrajectoryConfig(
            trajectory_type=TrajectoryType.APPROACH_LOITER_DEPART,
            robot_pos=robot_pos,
            spawn_pos=spawn_pos,
            approach_yaw=np.pi,
            closest_approach=1.0,
            loiter_duration=2.0,
            departure_angle=150.0,
            walk_speed=1.2,
        )),
        ("3. ARC — Curve around robot at 1.5m radius", TrajectoryConfig(
            trajectory_type=TrajectoryType.ARC,
            robot_pos=robot_pos,
            spawn_pos=spawn_pos,
            arc_radius=1.5,
            arc_extent=150.0,
            walk_speed=1.0,
        )),
    ]
    
    # Create scene
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(SCENE_XML)
        scene_path = f.name
    
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    human_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human")
    
    print("=" * 60)
    print("  TRAJECTORY PLANNER — 3D MuJoCo Demo")
    print("=" * 60)
    print("  Orange box = Robot (at origin)")
    print("  Blue capsule = Human (follows planner)")
    print("  Red ring = 1m safety radius")
    print("")
    print("  Will play 3 trajectories in sequence.")
    print("  Press ESC in viewer to quit.")
    print("=" * 60)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera for good overview
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -35
        viewer.cam.distance = 8
        viewer.cam.lookat[:] = [0, 0, 0.5]
        
        for name, config in trajectories:
            if not viewer.is_running():
                break
            
            planner = TrajectoryPlanner(config)
            run_trajectory(viewer, model, data, planner, name, human_body_id)
        
        if viewer.is_running():
            print(f"\n{'='*60}")
            print("  ✅ All trajectories complete! Viewer stays open.")
            print("  Press ESC to close.")
            print(f"{'='*60}")
            
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
