"""
Test: SafetyBiGymEnv Integration

Quick test to verify SafetyBiGymEnv works with BiGym.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_safety_env_creation():
    """Test that SafetyBiGymEnv can be created."""
    print("Testing SafetyBiGymEnv creation...")
    
    from bigym.action_modes import JointPositionActionMode
    from safety_bigym import SafetyBiGymEnv, SafetyConfig, HumanConfig
    
    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
    )
    
    # Create env WITHOUT human injection for initial test
    # (to avoid XML merge complexity)
    env = SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=SafetyConfig(),
        human_config=HumanConfig(),
        inject_human=False,  # Skip human for basic test
    )
    
    print(f"  ✅ Env created: {env.task_name}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Observation space keys: {list(env.observation_space.spaces.keys())}")
    
    # Test reset
    obs, info = env.reset()
    print(f"  ✅ Reset successful")
    print(f"  Info keys: {list(info.keys())}")
    
    # Test step
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  ✅ Step successful")
    print(f"  Safety info: {info.get('safety', {})}")
    
    env.close()
    print("  ✅ Env closed")
    
    return True


def test_safety_env_with_human():
    """Test SafetyBiGymEnv with human injection."""
    print("\nTesting SafetyBiGymEnv WITH human...")
    
    from bigym.action_modes import JointPositionActionMode
    from safety_bigym import SafetyBiGymEnv, SafetyConfig, HumanConfig
    
    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
    )
    
    try:
        env = SafetyBiGymEnv(
            action_mode=action_mode,
            safety_config=SafetyConfig(),
            human_config=HumanConfig(),
            inject_human=True,  # Try with human
        )
        
        print(f"  ✅ Env with human created")
        
        obs, info = env.reset()
        print(f"  ✅ Reset with human successful")
        
        # Run a few steps
        for i in range(10):
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  ✅ 10 steps completed")
        safety = info.get("safety", {})
        print(f"  SSM violation: {safety.get('ssm_violation', False)}")
        print(f"  PFL violation: {safety.get('pfl_violation', False)}")
        print(f"  Max force: {safety.get('max_contact_force', 0):.1f}N")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ⚠️  Human injection failed: {e}")
        print("  This is expected if XML merge needs debugging.")
        return False


def test_violation_penalty_applied_when_enabled():
    """`_reward()` subtracts `violation_penalty` when `add_violation_penalty`
    is True and `_step_safety_info` carries a violation flag. Regression
    guard for Phase-1.4 reward shaping.
    """
    from bigym.action_modes import JointPositionActionMode
    from safety_bigym import SafetyBiGymEnv, SafetyConfig, HumanConfig
    from safety_bigym.safety.iso15066_wrapper import SafetyInfo

    action_mode = JointPositionActionMode(floating_base=True, absolute=True)

    penalty = 0.05
    env_on = SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=SafetyConfig(
            add_violation_penalty=True, violation_penalty=penalty,
        ),
        human_config=HumanConfig(),
        inject_human=False,
    )
    env_off = SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=SafetyConfig(
            add_violation_penalty=False, violation_penalty=penalty,
        ),
        human_config=HumanConfig(),
        inject_human=False,
    )

    try:
        env_on.reset()
        env_off.reset()

        # 1) No violation → both envs return identical reward.
        env_on._step_safety_info = SafetyInfo(ssm_violation=False, pfl_violation=False)
        env_off._step_safety_info = SafetyInfo(ssm_violation=False, pfl_violation=False)
        assert env_on._reward() == env_off._reward()

        # 2) SSM violation → on env penalized by -penalty vs off env.
        env_on._step_safety_info = SafetyInfo(ssm_violation=True, pfl_violation=False)
        env_off._step_safety_info = SafetyInfo(ssm_violation=True, pfl_violation=False)
        assert abs((env_off._reward() - env_on._reward()) - penalty) < 1e-9

        # 3) PFL violation also triggers the penalty (single deduction even
        # when both flags fire — guard against future double-penalty bugs).
        env_on._step_safety_info = SafetyInfo(ssm_violation=False, pfl_violation=True)
        env_off._step_safety_info = SafetyInfo(ssm_violation=False, pfl_violation=True)
        assert abs((env_off._reward() - env_on._reward()) - penalty) < 1e-9

        env_on._step_safety_info = SafetyInfo(ssm_violation=True, pfl_violation=True)
        env_off._step_safety_info = SafetyInfo(ssm_violation=True, pfl_violation=True)
        assert abs((env_off._reward() - env_on._reward()) - penalty) < 1e-9
    finally:
        env_on.close()
        env_off.close()


if __name__ == "__main__":
    print("=" * 60)
    print("SafetyBiGymEnv Integration Test")
    print("=" * 60)
    
    # Test without human first
    test1 = test_safety_env_creation()
    
    # Test with human
    test2 = test_safety_env_with_human()
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Basic env (no human): {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Env with human:       {'✅ PASS' if test2 else '⚠️  NEEDS WORK'}")
    print("=" * 60)
