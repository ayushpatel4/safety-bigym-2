"""Asset-level tests for ``assets/g1_human_body.xml`` (built from upstream
``assets/g1/g1.xml`` via ``scripts/build_g1_human_body.py``).

These run without ``AMASS_DATA_DIR`` set — G1 is AMASS-free by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.human import g1_human_spec  # noqa: E402


ASSET = Path(__file__).parent.parent / "safety_bigym" / "assets" / "g1_human_body.xml"


@pytest.fixture(scope="module")
def model():
    if not ASSET.exists():
        pytest.skip(
            f"g1_human_body.xml not generated yet; run "
            "scripts/build_g1_human_body.py first"
        )
    return mujoco.MjModel.from_xml_path(str(ASSET))


def test_xml_loads(model):
    assert model is not None


def test_pelvis_is_mocap(model):
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
    assert pid >= 0, "Body named 'Pelvis' not found in g1_human_body.xml"
    mocapid = int(model.body_mocapid[pid])
    assert mocapid >= 0, (
        "Pelvis must be a mocap body (mocap_pos / mocap_quat-driven) so the "
        "trajectory planner can teleport it."
    )


def test_29_body_joints(model):
    # All 29 hinge joints from BODY_JOINT_NAMES must exist; no extra joints
    # (Pelvis is mocap, no freejoint).
    assert model.njnt == len(g1_human_spec.BODY_JOINT_NAMES) == 29
    for joint_name in g1_human_spec.BODY_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert jid >= 0, f"Joint '{joint_name}' missing"


def test_one_actuator_per_joint(model):
    assert model.nu == 29
    for joint_name in g1_human_spec.BODY_JOINT_NAMES:
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{joint_name}"
        )
        assert aid >= 0, f"Actuator 'act_{joint_name}' missing"


def test_all_collision_geoms_end_with_col(model):
    cols = []
    other_named = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        # Visual-only geoms (both masks 0) are ignored — we have no such
        # geoms after the build script strips all mesh visuals.
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        if ct == 0 and ca == 0:
            continue
        if name and name.endswith("_col"):
            cols.append(name)
        elif name:
            other_named.append(name)
    assert cols, "No _col geoms found"
    assert not other_named, (
        f"Found {len(other_named)} collision geoms not ending in '_col': "
        f"{other_named[:5]}"
    )


def test_collision_bits_match_smplh_channel(model):
    """Human geoms must use the same cross-paired channel as SMPL-H:
    contype=0b010, conaffinity=0b100 (so `_configure_collision_bits`
    treats G1 identically to SMPL-H).
    """
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not (name and name.endswith("_col")):
            continue
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        assert ct == 0b010, f"{name}: contype={ct:b}, expected 010"
        assert ca == 0b100, f"{name}: conaffinity={ca:b}, expected 100"


def test_ssm_bodies_exist(model):
    for body_name in g1_human_spec.SSM_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert bid >= 0, f"SSM body '{body_name}' missing in g1 asset"


def test_arm_chains_resolve(model):
    for arm_name, chain in g1_human_spec.ARM_CHAINS.items():
        for j in chain["joints"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            assert jid >= 0, f"Arm {arm_name}: joint '{j}' missing"
        ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, chain["end_effector"]
        )
        assert ee_id >= 0, (
            f"Arm {arm_name}: end_effector body '{chain['end_effector']}' missing"
        )
        sh_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, chain["shoulder_body"]
        )
        assert sh_id >= 0, (
            f"Arm {arm_name}: shoulder_body '{chain['shoulder_body']}' missing"
        )


def test_no_mesh_references(model):
    """Curriculum-validated G1 preset (commit 2683b67): skin-toned ``_col``
    capsules are the render; no STL meshes. Empty ``<asset>`` is fine —
    ``_create_merged_world`` merges assets when present (no-op here).
    """
    assert model.nmesh == 0, (
        f"g1_human_body.xml still references {model.nmesh} meshes; the build "
        "script must strip all <mesh> and visual mesh geoms."
    )


def test_standing_pose_is_stable(model):
    """PD-hold at standing pose for 100 mj_step — pelvis stays put, no NaN."""
    data = mujoco.MjData(model)
    # Mocap pelvis at the configured standing height
    data.mocap_pos[0] = np.array([0.0, 0.0, g1_human_spec.STANDING_PELVIS_Z])
    data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
    # Body joints to standing pose
    for joint_name, angle in g1_human_spec.STANDING_POSE.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[jid]] = angle
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{joint_name}"
        )
        data.ctrl[aid] = angle
    mujoco.mj_forward(model, data)
    pelvis_z_initial = float(data.xpos[1, 2])
    for _ in range(100):
        mujoco.mj_step(model, data)
    pelvis_z_final = float(data.xpos[1, 2])
    assert np.isfinite(data.qpos).all(), "qpos went non-finite during PD hold"
    # Mocap body shouldn't drift (it's kinematically positioned).
    assert abs(pelvis_z_final - pelvis_z_initial) < 1e-5, (
        f"Mocap Pelvis drifted: initial={pelvis_z_initial}, final={pelvis_z_final}"
    )
