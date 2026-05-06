"""Tests for the CONTACT disruption type and the APPROACH_AND_PRESS trajectory.

These tests do not step a MuJoCo env — they verify the sampling and
planner geometry that determine whether the human will actually press
into the robot at runtime.
"""

import numpy as np

from safety_bigym.human.trajectory_planner import (
    TrajectoryConfig,
    TrajectoryPlanner,
    TrajectoryType,
)
from safety_bigym.scenarios.disruption_types import (
    DEFAULT_CONFIGS,
    DisruptionConfig,
    DisruptionType,
)
from safety_bigym.scenarios.scenario_sampler import (
    ParameterSpace,
    ScenarioSampler,
)


def _force_disruption_sampler(disruption_type: DisruptionType) -> ScenarioSampler:
    """Build a sampler that always emits one disruption type."""
    return ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=["dummy.npz"],
            disruption_weights={disruption_type: 1.0},
        ),
    )


def test_contact_default_config_press_intent():
    """The DEFAULT_CONFIG for CONTACT must encode the press metadata."""
    cfg = DEFAULT_CONFIGS[DisruptionType.CONTACT]
    assert cfg.embed_distance >= 0.0
    assert cfg.contact_target_part in {"ee", "left_forearm", "right_forearm", "torso"}
    assert cfg.target_noise_std == 0.0


def test_contact_scenarios_have_press_metadata():
    """Sampling CONTACT must produce a press IK config and route to APPROACH_LOITER_DEPART."""
    sampler = _force_disruption_sampler(DisruptionType.CONTACT)
    for seed in range(10):
        s = sampler.sample_scenario(seed)
        assert s.disruption_type == DisruptionType.CONTACT
        assert s.disruption_config.embed_distance >= 0.0
        assert s.disruption_config.contact_target_part in {
            "ee", "left_forearm", "right_forearm", "torso"
        }
        # Routing: CONTACT shares the APPROACH_LOITER_DEPART trajectory so
        # the human always eventually walks away after pressing.
        assert s.trajectory_type == "APPROACH_LOITER_DEPART"
        # Pinned aggressive: small closest_approach during the loiter window.
        assert s.closest_approach <= 0.2
        # Loiter is bounded — long enough to provoke a violation, short
        # enough to leave the robot recovery time.
        assert 4.0 <= s.loiter_duration <= 10.0


def test_obstruction_passive_intrusion_metadata():
    """OBSTRUCTION must plant in workspace, loiter, then depart."""
    sampler = _force_disruption_sampler(DisruptionType.OBSTRUCTION)
    for seed in range(10):
        s = sampler.sample_scenario(seed)
        assert s.disruption_type == DisruptionType.OBSTRUCTION

        target = s.disruption_config.obstruction_target
        assert target is not None
        # Workspace box: in front of robot, narrow lateral, around shoulder/EE height.
        assert 0.30 <= target[0] <= 0.65
        assert -0.20 <= target[1] <= 0.20
        assert 0.60 <= target[2] <= 0.95
        # Loiter is bounded so the human eventually departs.
        assert 4.0 <= s.loiter_duration <= 10.0
        # Stop tight to the workspace point.
        assert 0.2 <= s.closest_approach <= 0.5


def test_contact_trajectory_has_depart_phase():
    """All disruption types must end with a depart phase so the robot can recover."""
    config = TrajectoryConfig(
        trajectory_type=TrajectoryType.APPROACH_LOITER_DEPART,
        robot_pos=np.array([0.0, 0.0]),
        spawn_pos=np.array([1.5, 0.0]),
        approach_yaw=np.pi,
        closest_approach=0.0,
        loiter_duration=5.0,
        departure_angle=150.0,
        walk_speed=1.2,
    )
    planner = TrajectoryPlanner(config)

    # All three phases must appear, in order
    phases = {wp.phase for wp in planner.waypoints}
    assert phases == {"approach", "loiter", "depart"}

    # During loiter, the human is at/near the robot
    assert planner.closest_distance_to_robot() <= 0.55

    # After full duration, the human is back somewhere far from the robot
    x, y, _, phase = planner.get_pose(planner.duration)
    assert phase == "depart"
    assert np.linalg.norm([x, y]) > 1.0


def test_get_ik_target_contact_embed_into_link():
    """CONTACT IK target must sit inside the link surface (embedded toward human)."""
    cfg = DisruptionConfig(
        disruption_type=DisruptionType.CONTACT,
        contact_target_part="ee",
        embed_distance=0.05,
    )
    ee_pos = np.array([0.5, 0.0, 1.0])
    pelvis_pos = np.array([1.0, 0.0, 1.0])  # human is +X of robot EE
    robot_state = {
        "ee_pos": ee_pos,
        "link_pos": {"ee": ee_pos},
        "human_pelvis_pos": pelvis_pos,
        "robot_base_pos": np.zeros(3),
    }
    rng = np.random.default_rng(0)
    target = cfg.get_ik_target(robot_state, rng)

    # Target should be displaced toward the human pelvis by embed_distance.
    np.testing.assert_allclose(target, ee_pos + np.array([0.05, 0.0, 0.0]), atol=1e-9)


def test_get_ik_target_contact_falls_back_to_ee():
    """If link_pos lacks the requested part, CONTACT falls back to ee_pos."""
    cfg = DisruptionConfig(
        disruption_type=DisruptionType.CONTACT,
        contact_target_part="left_forearm",
        embed_distance=0.0,
    )
    ee_pos = np.array([0.4, 0.1, 0.9])
    robot_state = {"ee_pos": ee_pos}
    rng = np.random.default_rng(0)
    target = cfg.get_ik_target(robot_state, rng)
    np.testing.assert_allclose(target, ee_pos)


def test_stratified_sample_aggression_distribution():
    """Across the default mix, ~30%+ of episodes must have closest_approach < 0.3."""
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(clip_paths=["dummy.npz"]),
    )
    n = 200
    close = 0
    contact = 0
    for seed in range(n):
        s = sampler.sample_scenario(seed)
        if s.disruption_type == DisruptionType.CONTACT:
            contact += 1
        if s.closest_approach < 0.3:
            close += 1
    assert contact >= n * 0.10, f"too few CONTACT episodes: {contact}/{n}"
    assert close >= n * 0.30, f"too few close-approach episodes: {close}/{n}"
