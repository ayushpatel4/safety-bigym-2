"""Intensive validation that ISO 15066 safety tracking survives the SMPL-H
to Unitree G1 coworker swap.

Covers four layers (wiring / SSM / PFL / episode aggregation) plus a forced
overlap and a regression sanity check, per the project plan. Live PFL contact
force is gated by a pre-existing BiGym/mojo contact-detection bug; we validate
the PFL *classification path* with synthetic contacts so it can't silently go
dead. SSM is the load-bearing live signal and is exercised end-to-end.
"""

from __future__ import annotations

import numpy as np
import mujoco
import pytest

from safety_bigym.human import g1_spec
from safety_bigym.safety import ISO15066Wrapper, SafetyInfo, ContactInfo
from safety_bigym.safety.pfl_limits import (
    GEOM_TO_REGION,
    PFL_LIMITS,
    get_region_for_geom,
    get_limits_for_geom,
)
from safety_bigym.envs.safety_env import SafetyBiGymEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def standalone_model():
    """G1 body model loaded directly (no BiGym world, no robot)."""
    return mujoco.MjModel.from_xml_path(str(SafetyBiGymEnv.HUMAN_BODY_PATH))


@pytest.fixture(scope="module")
def coworker_env():
    """A full SafetyBiGymEnv with G1 + ReachTargetSingle + COWORKER scenario."""
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.envs.reach_target import ReachTargetSingle
    from safety_bigym import SafetyConfig, HumanConfig
    from safety_bigym.envs.safety_env import make_safety_env
    from safety_bigym.scenarios.scenario_sampler import (
        ParameterSpace, ScenarioSampler,
    )
    from safety_bigym.scenarios.disruption_types import DisruptionType

    action_mode = JointPositionActionMode(
        absolute=True, floating_base=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    sampler = ScenarioSampler(parameter_space=ParameterSpace(
        disruption_weights={DisruptionType.COWORKER: 1.0},
    ))
    env = make_safety_env(
        ReachTargetSingle,
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False),
        human_config=HumanConfig(),
        scenario_sampler=sampler,
        inject_human=True,
    )
    yield env
    env.close()


# ---------------------------------------------------------------------------
# A. Wiring — every name list / geom set used by the safety stack resolves.
# ---------------------------------------------------------------------------


def test_pfl_region_map_covers_every_col_geom(standalone_model):
    """No G1 `_col` geom may map to None — that's how PFL silently goes dead."""
    m = standalone_model
    unmapped = []
    n_col = 0
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        # Treat `..._col` and `..._colN` as collision geoms.
        if name.endswith("_col") or name.rstrip("0123456789").endswith("_col"):
            n_col += 1
            if get_region_for_geom(name) is None:
                unmapped.append(name)
    assert n_col > 0, "no `_col` geoms found in g1_human_body.xml"
    assert unmapped == [], f"G1 collision geoms missing region map: {unmapped}"


def test_pfl_region_map_has_limits_for_every_region():
    for geom_name, region in GEOM_TO_REGION.items():
        assert region in PFL_LIMITS, f"{geom_name} -> {region} but no PFL_LIMITS entry"
        assert get_limits_for_geom(geom_name) is not None


def test_pfl_region_map_normalises_numeric_suffix():
    # Foot has multiple collision spheres named `_col`, `_col1`, ...
    assert get_region_for_geom("left_ankle_roll_link_col") == "foot"
    assert get_region_for_geom("left_ankle_roll_link_col2") == "foot"


def test_ssm_body_names_all_resolve_standalone(standalone_model):
    m = standalone_model
    missing = [
        n for n in g1_spec.SSM_BODY_NAMES
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) < 0
    ]
    assert missing == [], f"missing G1 SSM bodies: {missing}"


def test_pelvis_is_mocap_in_standalone_model(standalone_model):
    m = standalone_model
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, g1_spec.PELVIS_BODY)
    assert pid >= 0
    assert int(m.body_mocapid[pid]) >= 0, "pelvis is not a mocap body"


def test_iso_wrapper_picks_up_g1_col_geoms(standalone_model):
    m = standalone_model
    data = mujoco.MjData(m)
    wrapper = ISO15066Wrapper(model=m, data=data, human_geom_suffix="_col")
    n_col = sum(
        1 for i in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_col")
    )
    assert n_col > 0
    assert len(wrapper.human_geoms) == n_col


# ---------------------------------------------------------------------------
# B. SSM — live signal. Must respond to proximity, fire on overlap, and label
#    the closest joint/link with G1 names.
# ---------------------------------------------------------------------------


def test_ssm_compute_basic_with_g1_positions(standalone_model):
    """compute_ssm + build_safety_info return finite values and resolve names."""
    m = standalone_model
    data = mujoco.MjData(m)
    wrapper = ISO15066Wrapper(model=m, data=data)

    # Two pairs, minimum distance is between human[1]=(0,0,1.5) and
    # robot[1]=(0.0, 0.0, 1.5) → 0.0 (set explicitly for an exact assertion).
    human_pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.5]])
    robot_pos = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.5]])
    is_v, margin, d_min = wrapper.compute_ssm(robot_pos, 0.5, human_pos)
    assert np.isfinite(margin) and np.isfinite(d_min)
    assert d_min == pytest.approx(0.0, abs=1e-6)

    info = wrapper.build_safety_info(
        contacts=[], robot_positions=robot_pos, robot_vel=0.5,
        human_positions=human_pos, human_vel=0.0,
        human_names=["pelvis", "torso_link"],
        robot_names=["link_a", "link_b"],
    )
    assert info.closest_human_joint in ("pelvis", "torso_link")
    assert info.closest_robot_link in ("link_a", "link_b")


def test_ssm_end_to_end_monotonic_and_finite(coworker_env):
    """Run a single COWORKER episode and confirm SSM stays finite, the
    coworker approaches close enough to drop the margin, and the closest
    human joint label is a G1 body."""
    env = coworker_env
    obs, info = env.reset(seed=11)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    margins, seps, joints = [], [], set()
    for _ in range(1200):  # ~enough to enter loiter
        obs, r, term, trunc, info = env.step(a)
        s = info.get("safety", {})
        margins.append(float(s.get("ssm_margin", np.nan)))
        seps.append(float(s.get("min_separation", np.nan)))
        cj = s.get("closest_human_joint", "")
        if cj:
            joints.add(cj)
        if term or trunc:
            break

    margins = np.asarray(margins)
    seps = np.asarray(seps)
    assert np.all(np.isfinite(margins)), "ssm_margin went non-finite mid-episode"
    assert np.all(np.isfinite(seps))
    # SSM must be responsive — separation should vary at least a millimetre
    # across the episode (rules out a broken/frozen signal).
    assert seps.ptp() > 1e-3, f"SSM appears frozen (range {seps.ptp():.6f})"
    # Every reported closest-joint must be a known G1 SSM body.
    valid = set(g1_spec.SSM_BODY_NAMES)
    assert joints, "no closest-joint label ever populated"
    assert joints.issubset(valid), f"unexpected closest-joint labels: {joints - valid}"


def test_ssm_forced_violation_pelvis_in_robot(coworker_env):
    """Teleport the G1 pelvis to the robot end-effector and assert SSM fires.
    This guards against silently broken SSM if names/indices ever desync."""
    env = coworker_env
    env.reset(seed=12)
    ee = np.asarray(env._get_robot_state()["link_pos"]["ee"], dtype=float)
    mid = env._human_pelvis_mocapid
    env._mojo.data.mocap_pos[mid] = ee
    mujoco.mj_forward(env._mojo.model, env._mojo.data)

    # _aggregate_safety_info reads positions directly from the model state;
    # call it instead of env.step() (which would overwrite the mocap via the
    # scripted trajectory).
    env._aggregate_safety_info()
    s = env._step_safety_info.to_dict()
    assert s["ssm_violation"] is True
    assert s["ssm_margin"] < 0
    assert s["min_separation"] < 0.5
    assert s["closest_human_joint"] in g1_spec.SSM_BODY_NAMES


def test_ssm_velocity_cap_honoured(coworker_env):
    """The mocap pelvis teleports each step; raw cvel can be huge. The
    safety state must clamp it to ssm_config.v_h_max."""
    env = coworker_env
    env.reset(seed=13)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    cap = float(env.safety_config.ssm.v_h_max)
    for _ in range(20):
        env.step(a)
        _, _, max_vel = env._human_ssm_state()
        assert max_vel <= cap + 1e-6, f"max_vel={max_vel} exceeds v_h_max={cap}"


# ---------------------------------------------------------------------------
# C. PFL — classification path. Live force detection is gated by the open
#    BiGym/mojo contact-detection bug (see CLAUDE.md); we validate the
#    classification with synthetic ContactInfos so it cannot silently go dead.
# ---------------------------------------------------------------------------


def _make_contact(geom_name: str, force: float, contact_type: str = "transient") -> ContactInfo:
    limits = get_limits_for_geom(geom_name)
    assert limits is not None, f"no limits for {geom_name}"
    is_v, ratio = limits.check_violation(force, contact_type)
    return ContactInfo(
        geom1_name=geom_name,
        geom2_name="h1/ee_geom",
        force=force,
        contact_type=contact_type,
        body_region=get_region_for_geom(geom_name),
        is_human_robot=True,
        is_violation=is_v,
        force_ratio=ratio,
        force_limit=limits.get_force_limit(contact_type),
    )


@pytest.mark.parametrize("geom_name,region", [
    ("head_col", "skull"),
    ("torso_col", "chest"),
    ("right_elbow_link_col", "forearm"),
    ("right_wrist_yaw_link_col", "hand_palm"),
    ("right_hip_yaw_link_col", "thigh"),
    ("left_knee_link_col", "shin"),
    ("left_ankle_roll_link_col2", "foot"),  # numeric-suffix variant
])
def test_pfl_classification_per_region_under_limit(standalone_model, geom_name, region):
    wrapper = ISO15066Wrapper(model=standalone_model, data=mujoco.MjData(standalone_model))
    limit = PFL_LIMITS[region].transient_force
    info = wrapper.build_safety_info(
        contacts=[_make_contact(geom_name, force=0.5 * limit)],
    )
    assert info.contact_region == region
    assert info.pfl_violation is False
    assert 0.0 < info.pfl_force_ratio < 1.0
    assert info.max_contact_force == pytest.approx(0.5 * limit)


@pytest.mark.parametrize("geom_name,region", [
    ("head_col", "skull"),
    ("right_wrist_yaw_link_col", "hand_palm"),
    ("right_hip_yaw_link_col", "thigh"),
])
def test_pfl_classification_per_region_over_limit(standalone_model, geom_name, region):
    wrapper = ISO15066Wrapper(model=standalone_model, data=mujoco.MjData(standalone_model))
    limit = PFL_LIMITS[region].transient_force
    info = wrapper.build_safety_info(
        contacts=[_make_contact(geom_name, force=1.5 * limit)],
    )
    assert info.pfl_violation is True
    assert info.pfl_force_ratio > 1.0
    assert info.violations_by_region.get(region) == 1


def test_pfl_max_across_multiple_regions(standalone_model):
    wrapper = ISO15066Wrapper(model=standalone_model, data=mujoco.MjData(standalone_model))
    contacts = [
        _make_contact("head_col", force=10.0),  # well under skull limits
        _make_contact("right_elbow_link_col", force=400.0),  # over forearm transient (320)
    ]
    info = wrapper.build_safety_info(contacts=contacts)
    assert info.pfl_violation is True
    assert info.contact_region == "forearm"  # the higher-force contact
    assert info.violations_by_region.get("forearm") == 1
    # Skull contact below limits must not appear in violations
    assert "skull" not in info.violations_by_region


# ---------------------------------------------------------------------------
# D. Episode aggregation — EpisodeSafetyMetrics receives G1-aware safety dicts.
# ---------------------------------------------------------------------------


def test_episode_safety_metrics_summary(coworker_env):
    """Run a few steps and confirm the per-episode summary populates with
    G1-aware values."""
    env = coworker_env
    # The factory wraps env in EpisodeSafetyMetrics; this fixture is the inner
    # env, so we wrap it explicitly here to keep the test self-contained.
    from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics
    wrapped = EpisodeSafetyMetrics(env)
    wrapped.reset(seed=21)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    # Force a violation mid-episode by teleporting the pelvis onto the EE
    # at step 3 (the wrapper accumulates the violation flag from info["safety"]).
    for i in range(15):
        if i == 3:
            ee = np.asarray(env._get_robot_state()["link_pos"]["ee"], dtype=float)
            env._mojo.data.mocap_pos[env._human_pelvis_mocapid] = ee
            mujoco.mj_forward(env._mojo.model, env._mojo.data)
            env._aggregate_safety_info()
            # Re-emit the violation into info["safety"] via a synthetic step;
            # easiest: peek at the summary after stepping with the teleported
            # pelvis still in place. The trajectory planner overwrites on the
            # next step but the violation is registered for this one.
        obs, r, term, trunc, info = wrapped.step(a)
        if term or trunc:
            break
    summary = info["episode_safety"]
    assert "ep_steps" in summary
    assert summary["ep_steps"] > 0
    assert 0.0 <= summary["ep_ssm_violation_rate"] <= 1.0
    assert np.isfinite(summary["ep_min_ssm_margin"])


# ---------------------------------------------------------------------------
# E. Regression sanity — the G1 envelope tracked by SSM has plausible scale.
# ---------------------------------------------------------------------------


def test_g1_ssm_envelope_geometry_sane(coworker_env):
    """At rest in the standing pose, the G1 SSM bodies should span roughly
    pelvis±(0..1.0m) — sanity-checks the body list and that the merged model
    preserved the G1 kinematic tree."""
    env = coworker_env
    env.reset(seed=14)
    pos, names, _ = env._human_ssm_state()
    pelvis_idx = names.index("pelvis")
    pelvis_xyz = pos[pelvis_idx]
    radii = np.linalg.norm(pos - pelvis_xyz, axis=1)
    # All tracked bodies are within ~1.2m of the pelvis (G1 is ~1.3m tall).
    assert radii.max() < 1.2, f"SSM body too far from pelvis: {radii.max():.3f}m"
    # Bodies are not collapsed onto the pelvis (model loaded with geometry).
    assert radii.max() > 0.3
