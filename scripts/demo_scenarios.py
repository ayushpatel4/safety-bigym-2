"""
Visual Demo: Scenario Sampler

Demonstrates the scenario sampler by sampling random scenarios
and showing:
1. Different disruption types
2. Motion playback with speed variations
3. IK reaching during disruption phase

Usage:
    mjpython scripts/demo_scenarios.py [--seed N]
"""

import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.scenarios import ScenarioSampler, DisruptionType
from safety_bigym.human import HumanController, HumanIK


# Scene with target marker for IK visualization
SCENE_XML = """
<mujoco model="scenario_demo">
  <include file="{smplh_path}"/>
  
  <worldbody>
    <!-- IK target marker (mocap for movement) -->
    <body name="ik_target" mocap="true" pos="0.5 0.5 1.0">
      <geom name="target_sphere" type="sphere" size="0.05" 
            rgba="1 0.2 0.2 0.9" contype="0" conaffinity="0"/>
    </body>
    
    <!-- Robot position marker (where robot would be) -->
    <body name="robot_marker" pos="0 1.5 0.8">
      <geom name="robot_base" type="box" size="0.2 0.2 0.4" 
            rgba="0.2 0.5 0.8 0.5" contype="0" conaffinity="0"/>
      <geom name="robot_ee" type="sphere" size="0.08" pos="0 -0.3 0.6"
            rgba="0.2 0.8 0.2 0.9" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def demo_scenario(seed: int = 0, motion_dir: str = None):
    """
    Demo a sampled scenario.
    
    Args:
        seed: Random seed for scenario sampling
        motion_dir: Path to AMASS motion clips
    """
    # Setup motion directory
    if motion_dir is None:
        motion_dir = Path(os.environ.get("AMASS_DATA_DIR", "/home/ap2322/Documents/CMU/CMU"))
    else:
        motion_dir = Path(motion_dir)
    
    # Create sampler
    sampler = ScenarioSampler(motion_dir=motion_dir)
    
    if not sampler.params.clip_paths:
        print("ERROR: No motion clips found")
        return
    
    # Sample scenario
    scenario = sampler.sample_scenario(seed)
    
    print(f"\n{'='*60}")
    print(f"SCENARIO (seed={seed})")
    print(f"{'='*60}")
    print(f"  Disruption:  {scenario.disruption_type.name}")
    print(f"  Trigger:     {scenario.trigger_time:.1f}s")
    print(f"  Blend:       {scenario.blend_duration:.2f}s")
    print(f"  Speed:       {scenario.speed_multiplier:.2f}x")
    print(f"  Height:      {scenario.human_height_percentile:.0%} percentile")
    print(f"  Approach:    {scenario.approach_angle:.0f}°")
    print(f"  Arm:         {scenario.reaching_arm}")
    print(f"  Clip:        {Path(scenario.clip_path).name}")
    print(f"{'='*60}\n")
    
    # Create scene
    smplh_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
    scene_xml = SCENE_XML.format(smplh_path=str(smplh_path))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(scene_xml)
        scene_path = f.name
    
    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Create controller and IK
    controller = HumanController(model, data)
    ik = HumanIK(model, data)
    
    # Load clip and set scenario (ScenarioParams is now unified)
    controller.set_scenario(scenario)
    controller.reset()
    
    print("Launching viewer...")
    print("  Watch the human motion")
    print(f"  At t={scenario.trigger_time:.1f}s: Disruption starts ({scenario.disruption_type.name})")
    print("  Close window to exit")
    
    # State
    paused = False
    current_seed = seed
    
    def key_callback(key):
        nonlocal paused, current_seed
        if key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            # Next scenario
            current_seed += 1
            print(f"\nLoading scenario seed={current_seed}...")
            new_scenario = sampler.sample_scenario(current_seed)
            print(f"  Type: {new_scenario.disruption_type.name}")
    
    dt = model.opt.timestep
    
    # Simulated robot state
    robot_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_ee")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused:
                # Get robot state (simulated)
                robot_state = {
                    'ee_pos': data.geom_xpos[robot_ee_id].copy(),
                    'task_object_pos': np.array([0.5, 0.5, 1.0]),
                    'robot_base_pos': np.array([0, 1.5, 0.8]),
                }
                
                # Check if in IK phase
                if controller.t >= scenario.trigger_time and scenario.disruption_config.requires_ik():
                    # Compute IK target
                    ik_target = scenario.disruption_config.get_ik_target(
                        robot_state, 
                        np.random.default_rng(int(controller.t * 1000))
                    )
                    
                    if ik_target is not None:
                        # Update visual marker
                        data.mocap_pos[0] = ik_target
                        
                        # Solve IK
                        arm = scenario.reaching_arm
                        arm_angles = ik.solve(arm, ik_target, max_iterations=20)
                        
                        # Apply arm angles (blend with AMASS for body)
                        qpos_indices = ik._chain_cache[arm]["qpos_indices"]
                        
                        # Get AMASS targets for body
                        amass_targets = controller._get_amass_targets(controller.t)
                        data.qpos[:] = amass_targets
                        
                        # Override arm with IK
                        blend = min(1.0, (controller.t - scenario.trigger_time) / scenario.blend_duration)
                        for i, idx in enumerate(qpos_indices):
                            data.qpos[idx] = (1 - blend) * amass_targets[idx] + blend * arm_angles[i]
                else:
                    # Pure AMASS
                    controller.step(dt, robot_state)
                
                # Kinematics update
                mujoco.mj_kinematics(model, data)
                
                # Print phase changes
                phase = controller.current_phase
                if int(controller.t * 10) % 20 == 0:
                    print(f"  t={controller.t:.1f}s phase={phase}", end='\r')
                
                controller.t += dt
            
            viewer.sync()
            time.sleep(0.001)
    
    print("\nDemo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo scenario sampler")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--motion-dir", type=str, help="Path to motion clips")
    
    args = parser.parse_args()
    demo_scenario(args.seed, args.motion_dir)
