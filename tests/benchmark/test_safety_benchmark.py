import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from safety_bigym.benchmark.policy import RandomPolicy
from safety_bigym.benchmark.safety_benchmark import SafetyBenchmark

@pytest.fixture
def mock_env():
    """Mock SafetyBiGymEnv for testing without MuJoCo."""
    env = MagicMock()
    env.action_space.sample.return_value = np.zeros(7)
    env.action_space.shape = (7,)
    
    # Mock Mojo
    env._mojo = MagicMock()
    env._mojo.model = MagicMock()
    env._mojo.data = MagicMock()

    # Mock reset return
    env.reset.return_value = ({'proprioception': np.zeros(10)}, {'scenario': {'disruption_type': 'TEST'}})
    
    # Mock step return
    # obs, reward, terminated, truncated, info
    safety_info = {
        "ssm_violation": False,
        "pfl_violation": False,
        "min_separation": 2.0,
        "max_contact_force": 0.0
    }
    env.step.return_value = (
        {'proprioception': np.zeros(10)}, 0.0, False, False, 
        {'safety': safety_info, 'success': True}
    )
    return env

@patch('safety_bigym.benchmark.safety_benchmark.make_safety_env')
@patch('safety_bigym.benchmark.safety_benchmark.mujoco.viewer.launch_passive')
def test_evaluate(mock_launch, mock_make_env, mock_env):
    """Test evaluate method runs loop and computes metrics."""
    # Setup mocks
    mock_make_env.return_value = mock_env
    mock_viewer = MagicMock()
    mock_launch.return_value = mock_viewer
    # Context manager support for viewer
    mock_viewer.__enter__.return_value = mock_viewer
    mock_viewer.__exit__.return_value = None
    mock_viewer.is_running.return_value = True

    # Initialize benchmark
    benchmark = SafetyBenchmark(task_cls=MagicMock())
    policy = RandomPolicy(action_space=mock_env.action_space)

    # Run evaluation
    results = benchmark.evaluate(policy, num_episodes=2, seed=42)

    # Verify calls
    assert mock_make_env.call_count == 1
    assert mock_env.reset.call_count == 2
    assert mock_env.step.call_count > 0
    
    # Verify results structure
    assert "metrics" in results
    metrics = results["metrics"]
    assert "ssm_violation_rate" in metrics
    assert "pfl_violation_rate" in metrics
    assert "success_rate" in metrics
    assert metrics["success_rate"] == 1.0  # We mocked success=True
