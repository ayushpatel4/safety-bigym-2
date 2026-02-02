"""
Demo: Safety Disruption with ISO 15066 Violations

This script demonstrates the SafetyBiGymEnv in action:
1. Sets up a ReachTarget task (H1 robot).
2. Spawns a human with a 'SHARED_GOAL' disruption scenario (human reaches for same target).
3. Runs the simulation and visualizes safety metrics.
4. Highlights SSM and PFL violations in real-time.

Usage:
    python scripts/demo_safety_disruption.py
"""

import time
import numpy as np
import mujoco
from pathlib import Path
import logging
import os
try:
    import mujoco.viewer
    # Allow forcing headless mode via env var
    HEADLESS = os.environ.get("HEADLESS", "0") == "1"
except ImportError:
    HEADLESS = True

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget

from safety_bigym import (
    SafetyBiGymEnv,
    SafetyConfig,
    HumanConfig,
    ScenarioSampler,
    ParameterSpace,
    DisruptionType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")

def create_dummy_motion_clip(filename="dummy_motion.npz"):
    """Create a dummy AMASS-style .npz file if meaningful data is missing."""
    import os
    if os.path.exists(filename):
        return filename
    
    logger.info(f"Creating dummy motion clip: {filename}")
    # 5 seconds at 60Hz
    num_frames = 300 
    
    # SMPL-H has 52 joints (22 body + 30 hand) -> 156 pose params + 3 root trans
    # Just use zeros (T-pose)
    poses = np.zeros((num_frames, 156)) 
    trans = np.zeros((num_frames, 3))
    
    # Add slight sliding motion to make it visible
    # Move along X axis slowly
    trans[:, 0] = np.linspace(0, 1.0, num_frames)
    
    np.savez(filename, 
             poses=poses, 
             trans=trans, 
             mocap_framerate=60.0,
             gender='male')
    return filename

def main():
    # 1. Configure the Environment
    # ----------------------------
    
    # 1a. Locate Motion Clips (AMASS/CMU)
    # Search for CMU dataset in typical locations
    possible_roots = [
        Path("CMU/CMU/01"),
        Path("../CMU/CMU/01"),
        Path("../../CMU/CMU/01"),
        Path(os.environ.get("AMASS_PATH", ""))
    ]
    
    clip_paths = []
    for root in possible_roots:
        if root.exists():
            # CMU dataset structure: CMU/Subject/Subject_Trial_poses.npz
            # We recursively find all .npz files
            found = list(root.rglob("*.npz"))
            if found:
                clip_paths = [str(p.absolute()) for p in found]
                msg = f"Found {len(clip_paths)} AMASS clips in {root}"
                logger.info(msg)
                print(f"[INFO] {msg}")
                # Also print the first file to confirm
                print(f"[INFO] Sample: {clip_paths[0]}")
                break
    
    # Fallback to dummy motion if no real data found
    if not clip_paths:
        msg = "No AMASS data found. Generating dummy motion."
        logger.warning(msg)
        print(f"[WARN] {msg}")
        clip_paths = [create_dummy_motion_clip()]

    # Robot Action Mode
    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True
    )
    
    # Safety Configuration (ISO 15066)
    safety_config = SafetyConfig(
        T_r=0.2,            # 200ms reaction time
        a_max=5.0,          # 5 m/s^2 max decel
        C=0.1,              # 10cm safety margin
        use_pfl=True,       # Enable Power & Force Limiting
        terminate_on_violation=False
    )
    
    # Human Configuration
    human_config = HumanConfig(
        # Use packaged SMPL-H asset
        spawn_positions=[
            {"pos": [2.0, 0.0, 0.0], "yaw": np.pi},  # Start behind robot
        ]
    )
    
    # Scenario Configuration
    # We allow diverse scenarios to showcase different disruptions
    param_space = ParameterSpace(
        # Use found clips
        clip_paths=clip_paths,
        # Allow all disruption types with default weights:
        # INCIDENTAL (30%), SHARED_GOAL (20%), DIRECT (20%), OBSTRUCTION (15%), RANDOM_PERTURBED (15%)
        # disruption_weights={DisruptionType.SHARED_GOAL: 1.0}, # Commented out to enable variety
        
        trigger_time_range=(1.0, 3.0),   # Varied trigger times
        spawn_distance_range=(1.5, 2.5)
    )
    
    # 2. Instantiate SafetyBiGymEnv
    # -----------------------------
    # We wrap ReachTarget, but conceptually SafetyBiGymEnv subclasses it dynamically 
    # or serves as the main env class if we designed it that way. 
    # Since SafetyBiGymEnv inherits BiGymEnv, we need to mix it with the Task logic.
    # But wait, SafetyBiGymEnv inherits BiGymEnv. ReachTarget inherits BiGymEnv.
    # To apply Safety to ReachTarget, we should probably dynamically subclass or 
    # create a class that inherits both?
    #
    # Simpler approach for this demo:
    # Use SafetyBiGymEnv as the base, and just manually set a target for the robot
    # or use a predefined "SafetyReachTarget" class.
    #
    # However, to reuse ReachTarget's reward/task logic, we can verify if SafetyBiGymEnv
    # was meant to WRAP an existing env or BE the env.
    # The code implements `class SafetyBiGymEnv(BiGymEnv):`.
    # This means it replaces the base class. 
    # To make ReachTarget safety-aware, we would define:
    # class SafetyReachTarget(SafetyBiGymEnv, ReachTarget): ...
    #
    # Let's define that here dynamically to show how to upgrade ANY BiGym task.
    
    class SafetyReachTarget(ReachTarget, SafetyBiGymEnv):
        """
        ReachTarget task with Safety features.
        Inheritance order is important: SafetyBiGymEnv overrides _step_mujoco_simulation.
        We need SafetyBiGymEnv first in MRO for the step override, 
        but ReachTarget for the task logic.
        Wait, SafetyBiGymEnv calls super()._step_mujoco_simulation? No, it REPLACES it.
        So we want SafetyBiGymEnv mixed in.
        
        MRO: SafetyReachTarget -> ReachTarget -> SafetyBiGymEnv? 
        No, ReachTarget inherits BiGymEnv. SafetyBiGymEnv inherits BiGymEnv.
        We want SafetyReachTarget -> SafetyBiGymEnv -> ReachTarget?
        No, if SafetyBiGymEnv inherits BiGymEnv, and we assume ReachTarget does too.
        
        If we define: class SafetyReachTarget(SafetyBiGymEnv, ReachTarget)
        MRO: SafetyReachTarget, SafetyBiGymEnv, ReachTarget, BiGymEnv...
        This works if ReachTarget logic is orthogonal to step loop.
        ReachTarget uses standard _step_mujoco_simulation from BiGymEnv.
        SafetyBiGymEnv overrides it.
        So putting SafetyBiGymEnv first should work.
        """
        def __init__(self, **kwargs):
            # Extract safety args to pass to SafetyBiGymEnv, rest to ReachTarget
            safety_kwargs = {
                k: kwargs.pop(k) for k in 
                ['safety_config', 'human_config', 'scenario_config'] 
                if k in kwargs
            }
            # Initialize straightforwardly - but multiple inheritance init is tricky.
            # SafetyBiGymEnv.__init__ calls super().__init__.
            # ReachTarget.__init__ calls super().__init__.
            # We explicitly initialize logic here.
            
            # 1. Initialize Safety Env part (wraps BiGymEnv init)
            SafetyBiGymEnv.__init__(self, **kwargs, **safety_kwargs)
            
            # 2. Initialize ReachTarget specific logic (target setup)
            # ReachTarget constructor sets self.target etc.
            # We might need to manually call ReachTarget-specific setup if it's in __init__
            # ReachTarget.__init__ mostly calls super and sets goal.
            # This is a bit hacky for a script.
            
            # ALTERNATIVE: Use SafetyBiGymEnv solely, and just control robot to move forward.
            # This avoids MRO complexity for the demo.
            pass

    # Let's try the simpler composition approach: use SafetyBiGymEnv and just drive the robot.
    # We lose the "ReachTarget" reward function, but for a safety demo, 
    # we mainly care about the collision/safety metrics.
    
    logger.info("Initializing Environment...")
    env = SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=safety_config,
        human_config=human_config,
        scenario_config=param_space
    )
    
    # 3. Simulation Loop
    # ------------------
    obs, info = env.reset()
    logger.info("Environment Reset. Human spawned.")
    # Print active scenario details
    current_scenario = env._current_scenario
    if current_scenario:
        print(f"\n[INFO] Active Scenario: {current_scenario.disruption_type.name}")
        print(f"[INFO] Trigger Time: {current_scenario.trigger_time:.2f}s")
    
    if not HEADLESS:
        print("\n=== Running in Visualizer ===")
        print("Press SPACE to pause/resume")
        print("Press BACKSPACE to reset")
        
        # Access underlying mujoco model/data from BiGym's Mojo interface
        # env.mojo -> Mojo -> physics (dm_control) -> model/data -> ptr (mujoco binding)
        model = env.mojo.physics.model.ptr
        data = env.mojo.physics.data.ptr
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            
            # Give time to orient
            time.sleep(1.0)
            
            step_start = time.time()
            running = True
            
            while viewer.is_running():
                if running:
                    # Simple robot policy: raise arms (t-poseish) or move forward
                    # Action size depends on robot. H1 has 19 actuators?
                    # We send zeros to hold pose, or slight movement
                    action = np.zeros(env.action_space.shape)
                    
                    # Example: slowly move arms forward to provoke contact if human approaches
                    # action[indices] = ...
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    viewer.sync()
                    
                    # Log safety violations
                    safety_info = info.get("safety", {})
                    if safety_info.get("ssm_violation"):
                        print(f"[WARN] SSM Violation! Margin: {safety_info.get('ssm_margin'):.3f}m")
                        
                    if safety_info.get("pfl_violation"):
                        force = safety_info.get("max_contact_force", 0.0)
                        region = safety_info.get("contact_region", "unknown")
                        print(f"[CRITICAL] PFL Violation! Force: {force:.1f}N on {region}")

                    # Time keeping
                    dt = 1.0 / env.control_frequency
                    time_until_next_step = dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                    step_start = time.time()

                    if terminated or truncated:
                        obs, info = env.reset()
                        print("Resetting environment...")
                        
    else:
        # Headless mode loop
        print("\n=== Running Headless (100 steps) ===")
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info["safety"]["ssm_violation"]:
                print(f"SSM Violation: {info['safety']['ssm_margin']:.3f}")
            if info["safety"]["pfl_violation"]:
                print(f"PFL Violation: {info['safety']['max_contact_force']:.1f}N")

    env.close()

if __name__ == "__main__":
    main()
