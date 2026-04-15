from bigym.envs.reach_target import ReachTargetSingle
from bigym.envs.pick_and_place import PickBox
from bigym.envs.dishwasher import DishwasherOpen
from bigym.action_modes import JointPositionActionMode
import os
from safety_bigym import make_safety_env, SafetyConfig, HumanConfig, SSMConfig


def setup():
    action_mode = JointPositionActionMode(floating_base=True, absolute=True)
    # Create human config with AMASS motion clip
    cmu_clips_dir = os.environ.get("AMASS_DATA_DIR")
    if not cmu_clips_dir:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it to the CMU AMASS root, e.g.\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
        )
    human_config = HumanConfig(
        motion_clip_dir=cmu_clips_dir,
        motion_clip_paths=["74/74_01_poses.npz"],  # Walking motion
    )

    safety_config = SafetyConfig(
        # SSM parameters (ISO 15066)
        ssm=SSMConfig(
            T_r=0.2,       # Robot reaction time (s)
            T_s=0.1,       # System response time (s)
            a_max=5.0,     # Max braking deceleration (m/s²)
            C=0.1,         # Intrusion distance / uncertainty (m)
            v_h_max=1.6,   # Max assumed human velocity (m/s)
        ),
        # PFL settings
        use_pfl=True,                    # Enable force monitoring
        # Violation behaviour
        terminate_on_violation=False,    # End episode on violation
        add_violation_penalty=True,      # Add penalty to reward
        violation_penalty=-1.0,          # Penalty magnitude
        # Logging
        log_violations=True,             # Log violations to console
        log_all_contacts=False,          # Log all contacts (verbose)
    )


    # Create environment using the factory
    print(f"Creating safety env with task: {DishwasherOpen.__name__}...")
    env = make_safety_env(
        task_cls=DishwasherOpen,
        action_mode=action_mode,
        safety_config=safety_config,
        human_config=human_config,
        inject_human=True,
    )
    
    print(f"✅ Environment created: {env.task_name}")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Human pelvis ID: {env._human_pelvis_id}")
    
    # Reset
    obs, info = env.reset()
    print(f"✅ Environment reset")
    print(f"   Scenario: {info.get('scenario', {})}")
    
    # Get model and data for viewer
    model = env._mojo.model
    data = env._mojo.data
    
    print("\nOpening viewer...")
    print("Press ESC to close")
    print("-" * 60)

    return model, data, env





def main():
    model, data, env = setup()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            safety = info["safety"]
            # Print status every 100 steps
            if step % 100 == 0:
                human_pos = data.xpos[env._human_pelvis_id] if env._human_pelvis_id else [0,0,0]
                sep = safety.get('min_separation', float('inf'))
                sep_str = f"{sep:.2f}m" if sep != float('inf') else "inf"
                print(f"Step {step:4d} | "
                      f"Sep: {sep_str} | "
                      f"SSM: {'⚠️' if safety.get('ssm_violation') else '✓'} | "
                      f"PFL: {'⚠️' if safety.get('pfl_violation') else '✓'} | "
                      f"Force: {safety.get('max_contact_force', 0):.1f}N | "
                      f"Human: ({human_pos[0]:.2f}, {human_pos[1]:.2f}, {human_pos[2]:.2f})")
            
            # Check termination
            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, info = env.reset()
                step = 0
            
            # Sync viewer
            viewer.sync()
            step += 1

    print("\nViewer closed. Cleaning up...")
    env.close()
    print("Done!")



if __name__ == "__main__":
    main()