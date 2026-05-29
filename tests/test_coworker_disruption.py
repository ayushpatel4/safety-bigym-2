"""Tests for the COWORKER disruption type.

These tests do not need a MuJoCo env to be instantiated end-to-end; they
exercise the sampler and trajectory planner directly. The
``CoworkerArmController`` itself requires an ``mjModel`` so we only cover
its state-machine logic via a minimal synthetic model in
:func:`test_coworker_state_machine_cycle`.
"""

from __future__ import annotations

import numpy as np
import mujoco
import pytest

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
    make_coworker_train_space,
    make_coworker_eval_space,
    _COWORKER_TRAIN_RANGES,
    _COWORKER_EVAL_RANGES,
    _COWORKER_TRAIN_TRAJECTORY_WEIGHTS,
)


def _force_disruption_sampler(disruption_type: DisruptionType) -> ScenarioSampler:
    return ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=["dummy.npz"],
            disruption_weights={disruption_type: 1.0},
        ),
    )


# ----------------------------------------------------------------------
# Sampler + default-config wiring
# ----------------------------------------------------------------------


def test_coworker_default_config_has_reach_metadata():
    cfg = DEFAULT_CONFIGS[DisruptionType.COWORKER]
    assert cfg.coworker_reach_period > 0.0
    assert 0.0 < cfg.coworker_reach_fraction < 1.0
    assert cfg.coworker_active_arm in {"left_arm", "right_arm"}
    assert cfg.requires_ik()


def test_coworker_sampled_scenarios_route_to_loiter_trajectories():
    """COWORKER must alternate among STATIONARY, APPROACH_LOITER_DEPART,
    and COWORKER_PATROL — never PASS_BY / ARC."""
    sampler = _force_disruption_sampler(DisruptionType.COWORKER)
    trajectory_types = set()
    for seed in range(60):
        s = sampler.sample_scenario(seed)
        assert s.disruption_type == DisruptionType.COWORKER
        assert s.trajectory_type in {
            "APPROACH_LOITER_DEPART", "STATIONARY", "COWORKER_PATROL",
        }
        assert s.disruption_config.coworker_active_arm in {"left_arm", "right_arm"}
        # Loiter has to cover most of an episode for a sustained coworker.
        assert s.loiter_duration >= 10.0
        # Coworker stands at polite co-worker distance — far enough that
        # the *body* isn't inside the robot's bubble, close enough that
        # the *arm* can extend in. The arm reach is what closes the gap.
        assert 0.8 <= s.closest_approach <= 1.5
        # Patrol-specific knobs must be in sane ranges.
        assert 5.0 <= s.patrol_near_loiter <= 15.0
        assert 2.0 <= s.patrol_away_loiter <= 6.0
        assert 1.5 <= s.patrol_away_distance <= 4.0
        assert 1 <= s.patrol_excursions <= 4
        trajectory_types.add(s.trajectory_type)

    # In 60 samples we expect all three spawn modes to fire at least once.
    assert trajectory_types == {
        "APPROACH_LOITER_DEPART", "STATIONARY", "COWORKER_PATROL"
    }


def test_stratified_sampling_yields_coworker():
    """Stratified sampling must reach COWORKER without infinite-looping."""
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(clip_paths=["dummy.npz"]),
    )
    stratified = sampler.get_stratified_sample(n_per_type=2, base_seed=0)
    assert DisruptionType.COWORKER in stratified
    assert len(stratified[DisruptionType.COWORKER]) == 2


# ----------------------------------------------------------------------
# STATIONARY trajectory geometry
# ----------------------------------------------------------------------


def test_stationary_trajectory_holds_pose():
    cfg = TrajectoryConfig(
        trajectory_type=TrajectoryType.STATIONARY,
        spawn_pos=np.array([0.6, 0.0]),
        robot_pos=np.array([0.0, 0.0]),
        loiter_duration=5.0,
    )
    planner = TrajectoryPlanner(cfg)

    for t in (0.0, 1.0, 4.99):
        x, y, yaw, phase = planner.get_pose(t)
        assert phase == "loiter"
        assert np.isclose(x, 0.6)
        assert np.isclose(y, 0.0)
        # Faces the robot — robot is at (0,0), spawn at (+0.6, 0), so yaw = pi.
        assert np.isclose(yaw, np.pi, atol=1e-6)


# ----------------------------------------------------------------------
# State machine cycle (uses a hand-rolled MJCF arm)
# ----------------------------------------------------------------------


# A minimal MJCF with the Unitree G1 arm joint/body names that HumanIK
# expects (shoulder pitch/roll/yaw + elbow per side; wrist yaw link as the EE
# body). Single-DoF hinges, mirroring the production G1 asset.
_FAKE_G1_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <geom type="sphere" size="0.05"/>
      <body name="right_shoulder_pitch_link" pos="0.2 0 0">
        <joint name="right_shoulder_pitch_joint" type="hinge" axis="0 1 0" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.04"/>
        <body name="right_shoulder_roll_link" pos="0.1 0 0">
          <joint name="right_shoulder_roll_joint" type="hinge" axis="1 0 0" range="-3 3"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.04"/>
          <body name="right_shoulder_yaw_link" pos="0.1 0 0">
            <joint name="right_shoulder_yaw_joint" type="hinge" axis="0 0 1" range="-3 3"/>
            <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.04"/>
            <body name="right_elbow_link" pos="0.1 0 0">
              <joint name="right_elbow_joint" type="hinge" axis="0 1 0" range="-3 3"/>
              <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.035"/>
              <body name="right_wrist_yaw_link" pos="0.2 0 0">
                <geom type="sphere" size="0.03"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="left_shoulder_pitch_link" pos="-0.2 0 0">
        <joint name="left_shoulder_pitch_joint" type="hinge" axis="0 1 0" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 -0.1 0 0" size="0.04"/>
        <body name="left_shoulder_roll_link" pos="-0.1 0 0">
          <joint name="left_shoulder_roll_joint" type="hinge" axis="1 0 0" range="-3 3"/>
          <geom type="capsule" fromto="0 0 0 -0.1 0 0" size="0.04"/>
          <body name="left_shoulder_yaw_link" pos="-0.1 0 0">
            <joint name="left_shoulder_yaw_joint" type="hinge" axis="0 0 1" range="-3 3"/>
            <geom type="capsule" fromto="0 0 0 -0.1 0 0" size="0.04"/>
            <body name="left_elbow_link" pos="-0.1 0 0">
              <joint name="left_elbow_joint" type="hinge" axis="0 1 0" range="-3 3"/>
              <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.035"/>
              <body name="left_wrist_yaw_link" pos="-0.2 0 0">
                <geom type="sphere" size="0.03"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _build_fake_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(_FAKE_G1_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_coworker_state_machine_cycle():
    """End-to-end: stepping through one period must hit all 4 phases and
    the IDLE qpos must match the rest pose exactly."""
    # Import here so the test file is still loadable even if the
    # coworker module fails import (lets pytest collect other tests).
    from safety_bigym.scenarios.coworker_behavior import (
        CoworkerArmController,
        PHASE_EXTEND,
        PHASE_HOLD,
        PHASE_RETRACT,
        PHASE_IDLE,
    )

    model, data = _build_fake_model()

    sampler = _force_disruption_sampler(DisruptionType.COWORKER)
    scenario = sampler.sample_scenario(seed=7)
    # Pin the period so the math below is deterministic.
    scenario.disruption_config.coworker_reach_period = 4.0
    scenario.disruption_config.coworker_reach_fraction = 0.25
    scenario.disruption_config.coworker_hold_fraction = 0.25
    scenario.disruption_config.coworker_retract_fraction = 0.25
    scenario.disruption_config.coworker_active_arm = "right_arm"
    scenario.disruption_config.coworker_target_mix = (1.0, 0.0)  # always EE

    rng = np.random.default_rng(0)
    controller = CoworkerArmController(model, data, scenario, rng)

    # Stub robot_state: EE at a point reachable by the arm.
    robot_state = {"ee_pos": np.array([0.4, 0.15, 1.0]), "t": 0.0}

    phases_seen: list[str] = []
    extended_norm: float = 0.0

    for t_sample in (0.5, 1.5, 2.5, 3.5):
        robot_state["t"] = t_sample
        qpos = controller.compute_qpos(t_sample, robot_state)
        phases_seen.append(controller.last_phase)
        if controller.last_phase == PHASE_HOLD:
            # During HOLD the arm must be at full IK, so the qpos for arm
            # DoFs is non-zero.
            arm_qpos = np.array(
                [qpos[i] for i in controller._arm_qpos_indices]
            )
            extended_norm = float(np.linalg.norm(arm_qpos))

    assert PHASE_EXTEND in phases_seen
    assert PHASE_HOLD in phases_seen
    assert PHASE_RETRACT in phases_seen
    assert PHASE_IDLE in phases_seen

    # IDLE qpos must equal the cached rest pose exactly. The rest pose
    # is the arms-down configuration computed at controller init — it
    # may be non-zero (the IK solver tries to hang each arm at the
    # side), but it must be reproducible: every IDLE-phase call returns
    # the same buffer.
    robot_state["t"] = 3.9
    qpos_idle = controller.compute_qpos(3.9, robot_state)
    assert np.allclose(qpos_idle, controller._rest_qpos)

    # And extension must have moved the arm appreciably away from rest.
    assert extended_norm > 0.05


def test_coworker_target_alternates_across_cycles():
    """With coworker_target_mix=(0.5, 0.5) we expect both EE and task
    targets to be chosen across cycles."""
    from safety_bigym.scenarios.coworker_behavior import CoworkerArmController

    model, data = _build_fake_model()
    sampler = _force_disruption_sampler(DisruptionType.COWORKER)
    scenario = sampler.sample_scenario(seed=3)
    scenario.disruption_config.coworker_reach_period = 1.0
    scenario.disruption_config.coworker_target_mix = (0.5, 0.5)
    scenario.disruption_config.coworker_active_arm = "right_arm"

    rng = np.random.default_rng(11)
    controller = CoworkerArmController(model, data, scenario, rng)

    robot_state = {
        "ee_pos": np.array([0.4, 0.15, 1.0]),
        "task_object_pos": np.array([0.0, 0.5, 1.1]),
    }

    kinds_seen: set[str] = set()
    # Sample 20 distinct cycles. Each cycle resamples the target.
    for cycle in range(20):
        robot_state["t"] = cycle * 1.0 + 0.05  # land in EXTEND
        controller.compute_qpos(robot_state["t"], robot_state)
        kinds_seen.add(controller._cycle.target_kind)
        if kinds_seen == {"ee", "task_object"}:
            break

    assert kinds_seen == {"ee", "task_object"}, (
        f"target should alternate; saw only {kinds_seen}"
    )


def test_coworker_falls_back_to_ee_when_no_task_object():
    """If the task doesn't expose a task object, every cycle must use EE."""
    from safety_bigym.scenarios.coworker_behavior import CoworkerArmController

    model, data = _build_fake_model()
    sampler = _force_disruption_sampler(DisruptionType.COWORKER)
    scenario = sampler.sample_scenario(seed=5)
    scenario.disruption_config.coworker_reach_period = 1.0
    scenario.disruption_config.coworker_target_mix = (0.0, 1.0)  # all task
    scenario.disruption_config.coworker_active_arm = "right_arm"

    rng = np.random.default_rng(0)
    controller = CoworkerArmController(model, data, scenario, rng)

    robot_state = {"ee_pos": np.array([0.4, 0.15, 1.0])}  # no task_object_pos

    for cycle in range(5):
        controller.compute_qpos(cycle * 1.0 + 0.05, robot_state)
        assert controller._cycle.target_kind == "ee"


# ----------------------------------------------------------------------
# Continuous knobs / train-vs-eval distribution
# ----------------------------------------------------------------------


_COWORKER_RANGE_FIELDS = (
    "coworker_closest_approach_range",
    "coworker_reach_period_range",
    "coworker_target_mix_p_ee_range",
    "coworker_near_loiter_range",
    "coworker_walk_speed_range",
)


def test_coworker_eval_is_strict_superset_of_train():
    """Every train range must lie strictly inside the eval range so
    eval probes both in-distribution and out-of-distribution conditions."""
    for field in _COWORKER_RANGE_FIELDS:
        t_lo, t_hi = _COWORKER_TRAIN_RANGES[field]
        e_lo, e_hi = _COWORKER_EVAL_RANGES[field]
        assert e_lo <= t_lo, f"{field}: eval lo {e_lo} not <= train lo {t_lo}"
        assert e_hi >= t_hi, f"{field}: eval hi {e_hi} not >= train hi {t_hi}"
        # Strict superset on at least one side (else there's no
        # generalisation gap to measure).
        assert (e_lo < t_lo) or (e_hi > t_hi), (
            f"{field}: train and eval ranges are identical — eval would "
            "not exercise OOD behaviour"
        )


def test_coworker_sampler_honours_continuous_knobs():
    """Each sampled COWORKER scenario must respect the active
    ParameterSpace's COWORKER ranges for all five knobs."""
    space = make_coworker_train_space()
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=["dummy.npz"],
            disruption_weights={DisruptionType.COWORKER: 1.0},
            **{k: space.__getattribute__(k) for k in _COWORKER_RANGE_FIELDS},
        ),
    )

    ca_lo, ca_hi = space.coworker_closest_approach_range
    rp_lo, rp_hi = space.coworker_reach_period_range
    p_lo, p_hi = space.coworker_target_mix_p_ee_range
    nl_lo, nl_hi = space.coworker_near_loiter_range
    ws_lo, ws_hi = space.coworker_walk_speed_range

    for seed in range(60):
        s = sampler.sample_scenario(seed)
        assert s.disruption_type == DisruptionType.COWORKER
        assert ca_lo <= s.closest_approach <= ca_hi
        rp = s.disruption_config.coworker_reach_period
        assert rp_lo <= rp <= rp_hi
        p_ee = s.disruption_config.coworker_target_mix[0]
        assert p_lo <= p_ee <= p_hi
        # P(EE) + P(task) must sum to ~1.0.
        assert abs(s.disruption_config.coworker_target_mix[1] - (1.0 - p_ee)) < 1e-9
        assert nl_lo <= s.patrol_near_loiter <= nl_hi
        assert ws_lo <= s.walk_speed <= ws_hi


def test_eval_space_samples_outside_train_ranges():
    """Sanity: with the eval space active, a fraction of episodes must
    land outside every train range — otherwise eval isn't exercising
    OOD behaviour and the experiment is uninformative."""
    eval_sampler = ScenarioSampler(
        parameter_space=make_coworker_eval_space(clip_paths=["dummy.npz"]),
    )

    counts = {f: 0 for f in _COWORKER_RANGE_FIELDS}
    total = 200
    for seed in range(total):
        s = eval_sampler.sample_scenario(seed)
        rp = s.disruption_config.coworker_reach_period
        p_ee = s.disruption_config.coworker_target_mix[0]
        samples = {
            "coworker_closest_approach_range": s.closest_approach,
            "coworker_reach_period_range": rp,
            "coworker_target_mix_p_ee_range": p_ee,
            "coworker_near_loiter_range": s.patrol_near_loiter,
            "coworker_walk_speed_range": s.walk_speed,
        }
        for f in _COWORKER_RANGE_FIELDS:
            t_lo, t_hi = _COWORKER_TRAIN_RANGES[f]
            if samples[f] < t_lo or samples[f] > t_hi:
                counts[f] += 1

    # Every knob's OOD region must be hit at least some of the time.
    # With eval ranges ~2x wider than train, expect >20% OOD per axis
    # in 200 samples. Floor at 10% to avoid flakiness.
    for f, c in counts.items():
        assert c >= total * 0.10, (
            f"{f}: only {c}/{total} samples landed OOD — eval distribution "
            "doesn't meaningfully widen this axis"
        )


def test_make_coworker_train_space_forces_only_coworker():
    """The train/eval factories pin disruption weights to COWORKER only
    so other disruption types don't pollute the experiment."""
    for space in (make_coworker_train_space(), make_coworker_eval_space()):
        assert space.disruption_weights == {DisruptionType.COWORKER: 1.0}


def test_coworker_train_space_favors_patrol():
    """Train distribution should heavily weight COWORKER_PATROL."""
    space = make_coworker_train_space()
    assert space.coworker_trajectory_weights == _COWORKER_TRAIN_TRAJECTORY_WEIGHTS
    sampler = ScenarioSampler(
        parameter_space=make_coworker_train_space(clip_paths=["dummy.npz"]),
    )
    counts = {"COWORKER_PATROL": 0, "APPROACH_LOITER_DEPART": 0, "STATIONARY": 0}
    total = 300
    for seed in range(total):
        counts[sampler.sample_scenario(seed).trajectory_type] += 1
    patrol_frac = counts["COWORKER_PATROL"] / total
    # 8:1:1 weights → expected 0.8; allow sampling variance.
    assert patrol_frac >= 0.65, (
        f"expected patrol-heavy train mix, got {counts} ({patrol_frac:.2%} patrol)"
    )
    assert counts["COWORKER_PATROL"] > 0
    assert counts["APPROACH_LOITER_DEPART"] > 0 or counts["STATIONARY"] > 0


def test_coworker_trajectory_weighted_choice():
    """Explicit weights must be honoured (patrol-only smoke)."""
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=["dummy.npz"],
            disruption_weights={DisruptionType.COWORKER: 1.0},
            coworker_trajectory_weights={"COWORKER_PATROL": 1.0},
        ),
    )
    for seed in range(20):
        assert sampler.sample_scenario(seed).trajectory_type == "COWORKER_PATROL"
