"""Smoke tests for :class:`G1HumanController`.

Verifies the controller can be constructed against the generated G1 model,
holds the standing pose under PD, and accepts the public set_*  API the
env wrapper calls (without requiring AMASS data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.human import g1_human_spec  # noqa: E402
from safety_bigym.human.g1_human_controller import G1HumanController  # noqa: E402


ASSET = Path(__file__).parent.parent / "safety_bigym" / "assets" / "g1_human_body.xml"


@pytest.fixture(scope="module")
def model_and_data():
    if not ASSET.exists():
        pytest.skip("g1_human_body.xml not generated yet")
    m = mujoco.MjModel.from_xml_path(str(ASSET))
    d = mujoco.MjData(m)
    return m, d


def test_controller_instantiates(model_and_data):
    m, d = model_and_data
    ctrl = G1HumanController(m, d)
    assert ctrl._mocap_id >= 0
    assert ctrl.clip is None
    # standing_qpos covers nq entries
    assert ctrl._standing_pose.shape == (m.nq,)


def test_pd_holds_standing_pose(model_and_data):
    """After ``step``, control values match standing pose; pelvis is at z=0.95."""
    m, d = model_and_data
    d.qpos[:] = 0.0  # start from defaults
    ctrl = G1HumanController(m, d)
    ctrl.step(0.005)  # one substep

    # Mocap pelvis written at the configured height.
    mid = ctrl._mocap_id
    assert abs(float(d.mocap_pos[mid, 2]) - g1_human_spec.STANDING_PELVIS_Z) < 1e-6

    # Body joints commanded to the standing pose.
    for joint_name, angle in g1_human_spec.STANDING_POSE.items():
        aid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{joint_name}"
        )
        assert aid >= 0
        assert abs(float(d.ctrl[aid]) - angle) < 1e-6, (
            f"{joint_name}: ctrl={d.ctrl[aid]}, expected {angle}"
        )


def test_load_clip_is_noop(model_and_data):
    """G1 has no AMASS; ``load_clip`` must accept the call without error."""
    m, d = model_and_data
    ctrl = G1HumanController(m, d)
    ctrl.load_clip("nonexistent/path.npz")  # must not raise
    assert ctrl.clip is None


def test_root_offset_writes_pelvis_xy(model_and_data):
    """``set_root_offset`` shifts the mocap pelvis in XY on next step."""
    m, d = model_and_data
    ctrl = G1HumanController(m, d)
    ctrl.set_root_offset(np.array([1.5, -2.0, 0.0]))
    ctrl.step(0.005)
    mid = ctrl._mocap_id
    assert abs(float(d.mocap_pos[mid, 0]) - 1.5) < 1e-6
    assert abs(float(d.mocap_pos[mid, 1]) - (-2.0)) < 1e-6


def test_ik_callback_blends_during_loiter(model_and_data):
    """During a (fake) loiter phase, controller calls the IK callback."""
    m, d = model_and_data
    ctrl = G1HumanController(m, d)

    # Fake trajectory planner: always returns ("at pos", phase="loiter").
    class FakePlanner:
        waypoints = [type("W", (), {"phase": "loiter", "time": 0.0})()]

        def get_pose(self, t):
            return 1.0, 0.0, 0.0, "loiter"

    ctrl.set_trajectory_planner(FakePlanner())

    # Fake scenario for blend_duration.
    ctrl.scenario = type("S", (), {"blend_duration": 0.0})()

    called = {"n": 0}

    def cb(state):
        called["n"] += 1
        # Return a qpos with all joints flexed (distinct from standing pose)
        q = np.full(m.nq, 0.5)
        return q

    ctrl.set_ik_callback(cb)
    ctrl.step(0.005)
    assert called["n"] >= 1, "IK callback never invoked during loiter phase"
