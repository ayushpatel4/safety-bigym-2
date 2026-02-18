"""
Unit tests for TrajectoryPlanner module.

Tests all trajectory types, interpolation, and boundary conditions.
"""

import numpy as np
import sys
sys.path.insert(0, "/Users/ayushpatel/Documents/FYP3/safety_bigym")

from safety_bigym.human.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryConfig,
    TrajectoryType,
)
from safety_bigym.scenarios.scenario_sampler import (
    ScenarioSampler,
    ScenarioParams,
    ParameterSpace,
)


def test_pass_by_trajectory():
    """Test PASS_BY trajectory doesn't approach robot too closely."""
    config = TrajectoryConfig(
        trajectory_type=TrajectoryType.PASS_BY,
        robot_pos=np.array([0.0, 0.0]),
        spawn_pos=np.array([3.0, 0.0]),
        approach_yaw=np.pi,  # facing -X (toward robot)
        pass_by_offset=1.0,
        pass_by_side=1,
        walk_speed=1.2,
    )
    
    planner = TrajectoryPlanner(config)
    
    print(f"  Duration: {planner.duration:.2f}s")
    print(f"  Waypoints: {len(planner.waypoints)}")
    print(f"  Min distance to robot: {planner.closest_distance_to_robot():.2f}m")
    
    # Should never get closer than pass_by_offset
    assert planner.closest_distance_to_robot() >= 0.9, \
        f"Pass-by too close: {planner.closest_distance_to_robot():.2f}m"
    
    # First waypoint should be at spawn
    x0, y0, yaw0, phase0 = planner.get_pose(0.0)
    assert abs(x0 - 3.0) < 0.1, f"Start x wrong: {x0}"
    assert phase0 == "walk"
    
    # Should be walking past (positive duration)
    assert planner.duration > 0
    
    # Sample along trajectory
    print("  Path:")
    for t in np.linspace(0, planner.duration, 6):
        x, y, yaw, phase = planner.get_pose(t)
        dist = np.sqrt(x**2 + y**2)
        print(f"    t={t:.2f}s: ({x:.2f}, {y:.2f}) yaw={np.degrees(yaw):.0f}° d={dist:.2f}m [{phase}]")
    
    print("  ✅ PASS_BY test passed")


def test_approach_loiter_depart():
    """Test APPROACH_LOITER_DEPART lifecycle."""
    config = TrajectoryConfig(
        trajectory_type=TrajectoryType.APPROACH_LOITER_DEPART,
        robot_pos=np.array([0.0, 0.0]),
        spawn_pos=np.array([3.0, 0.0]),
        approach_yaw=np.pi,
        closest_approach=1.0,
        loiter_duration=2.0,
        departure_angle=150.0,
        walk_speed=1.2,
    )
    
    planner = TrajectoryPlanner(config)
    
    print(f"  Duration: {planner.duration:.2f}s")
    print(f"  Waypoints: {len(planner.waypoints)}")
    
    # Check phases appear in order
    phases_seen = []
    prev_phase = None
    for t in np.linspace(0, planner.duration, 50):
        _, _, _, phase = planner.get_pose(t)
        if phase != prev_phase:
            phases_seen.append(phase)
            prev_phase = phase
    
    print(f"  Phases in order: {phases_seen}")
    assert "approach" in phases_seen, "Missing approach phase"
    assert "loiter" in phases_seen, "Missing loiter phase"
    assert "depart" in phases_seen, "Missing depart phase"
    
    # Approach should come before loiter, loiter before depart
    approach_idx = phases_seen.index("approach")
    loiter_idx = phases_seen.index("loiter")
    depart_idx = phases_seen.index("depart")
    assert approach_idx < loiter_idx < depart_idx, \
        f"Phase order wrong: {phases_seen}"
    
    # During loiter, position should be ~constant
    loiter_positions = []
    for t in np.linspace(
        planner.waypoints[2].time,  # loiter start
        planner.waypoints[3].time,  # loiter end
        10
    ):
        x, y, _, phase = planner.get_pose(t)
        loiter_positions.append([x, y])
    
    loiter_arr = np.array(loiter_positions)
    loiter_spread = np.max(np.std(loiter_arr, axis=0))
    print(f"  Loiter position spread: {loiter_spread:.4f}m")
    assert loiter_spread < 0.5, f"Loiter position unstable: {loiter_spread}"
    
    # Sample along trajectory
    print("  Path:")
    for t in np.linspace(0, planner.duration, 8):
        x, y, yaw, phase = planner.get_pose(t)
        dist = np.sqrt(x**2 + y**2)
        print(f"    t={t:.2f}s: ({x:.2f}, {y:.2f}) yaw={np.degrees(yaw):.0f}° d={dist:.2f}m [{phase}]")
    
    print("  ✅ APPROACH_LOITER_DEPART test passed")


def test_arc_trajectory():
    """Test ARC trajectory curves around robot."""
    config = TrajectoryConfig(
        trajectory_type=TrajectoryType.ARC,
        robot_pos=np.array([0.0, 0.0]),
        spawn_pos=np.array([2.0, 0.0]),
        arc_radius=1.5,
        arc_extent=120.0,
        walk_speed=1.0,
    )
    
    planner = TrajectoryPlanner(config)
    
    print(f"  Duration: {planner.duration:.2f}s")
    print(f"  Waypoints: {len(planner.waypoints)}")
    print(f"  Min distance to robot: {planner.closest_distance_to_robot():.2f}m")
    
    # Should stay at approximately arc_radius from robot
    for wp in planner.waypoints:
        dist = np.linalg.norm(wp.position)
        assert dist > 0.8 * config.arc_radius, \
            f"Arc point too close to robot: {dist:.2f}m at t={wp.time:.2f}s"
    
    # Sample along trajectory
    print("  Path:")
    for t in np.linspace(0, planner.duration, 8):
        x, y, yaw, phase = planner.get_pose(t)
        dist = np.sqrt(x**2 + y**2)
        print(f"    t={t:.2f}s: ({x:.2f}, {y:.2f}) yaw={np.degrees(yaw):.0f}° d={dist:.2f}m [{phase}]")
    
    print("  ✅ ARC test passed")


def test_scenario_sampler_trajectory_params():
    """Test that ScenarioSampler produces trajectory parameters."""
    params = ParameterSpace()
    sampler = ScenarioSampler(parameter_space=params)
    
    # Sample multiple scenarios and check trajectory fields
    trajectory_types_seen = set()
    for seed in range(30):
        scenario = sampler.sample_scenario(seed)
        
        assert hasattr(scenario, 'trajectory_type'), "Missing trajectory_type"
        assert hasattr(scenario, 'pass_by_offset'), "Missing pass_by_offset"
        assert hasattr(scenario, 'loiter_duration'), "Missing loiter_duration"
        assert hasattr(scenario, 'departure_angle'), "Missing departure_angle"
        assert hasattr(scenario, 'walk_speed'), "Missing walk_speed"
        
        trajectory_types_seen.add(scenario.trajectory_type)
        
        # Validate ranges
        assert 0.3 <= scenario.pass_by_offset <= 2.0, f"pass_by_offset out of range: {scenario.pass_by_offset}"
        assert 0.5 <= scenario.closest_approach <= 1.5, f"closest_approach out of range: {scenario.closest_approach}"
        assert 1.0 <= scenario.loiter_duration <= 5.0, f"loiter_duration out of range: {scenario.loiter_duration}"
        assert 0.8 <= scenario.walk_speed <= 1.6, f"walk_speed out of range: {scenario.walk_speed}"
    
    print(f"  Trajectory types seen: {trajectory_types_seen}")
    assert len(trajectory_types_seen) >= 2, \
        f"Low trajectory diversity: {trajectory_types_seen}"
    
    print("  ✅ Scenario sampler trajectory params test passed")


def test_interpolation_smoothness():
    """Test that interpolation is smooth (no jumps)."""
    config = TrajectoryConfig(
        trajectory_type=TrajectoryType.APPROACH_LOITER_DEPART,
        robot_pos=np.array([0.0, 0.0]),
        spawn_pos=np.array([3.0, 0.0]),
        closest_approach=1.0,
        loiter_duration=2.0,
        departure_angle=150.0,
        walk_speed=1.2,
    )
    
    planner = TrajectoryPlanner(config)
    
    # Sample at high frequency and check for jumps
    dt = 0.02  # 50Hz
    prev_x, prev_y = None, None
    max_jump = 0
    
    for i in range(int(planner.duration / dt)):
        t = i * dt
        x, y, _, _ = planner.get_pose(t)
        
        if prev_x is not None:
            jump = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            max_jump = max(max_jump, jump)
            # At 1.6 m/s and 0.02s, max displacement is ~0.032m per step
            # Allow 2x for smoothing overshoots
            assert jump < 0.1, f"Position jump at t={t:.3f}s: {jump:.4f}m"
        
        prev_x, prev_y = x, y
    
    print(f"  Max position step: {max_jump:.4f}m")
    print("  ✅ Interpolation smoothness test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("TrajectoryPlanner Unit Tests")
    print("=" * 60)
    
    print("\n1. PASS_BY trajectory:")
    test_pass_by_trajectory()
    
    print("\n2. APPROACH_LOITER_DEPART trajectory:")
    test_approach_loiter_depart()
    
    print("\n3. ARC trajectory:")
    test_arc_trajectory()
    
    print("\n4. Scenario sampler trajectory params:")
    test_scenario_sampler_trajectory_params()
    
    print("\n5. Interpolation smoothness:")
    test_interpolation_smoothness()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
