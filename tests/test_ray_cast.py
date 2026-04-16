"""Unit tests for filters.ray_cast.check_joint_visibility.

The BodySLAM++ perception stack uses the robot's head camera; joints blocked
by fixtures or the robot's own links are either missed entirely or estimated
with inflated error. check_joint_visibility models the geometric half of that
(line-of-sight from the camera to each target joint) so the wrapper can apply
an occlusion-conditional noise multiplier without rendering a depth image.
"""

import numpy as np
import mujoco
import pytest

from safety_bigym.filters.ray_cast import check_joint_visibility


def _make_model(with_occluder: bool, with_robot_link: bool = False):
    """Scene: camera 'head' at origin looking down +x; target bodies along +x;
    optional occluding wall at x=1, optional robot link along the same line."""
    occluder = (
        '<geom name="occluder" type="box" size="0.1 0.5 0.5" pos="1.0 0 0"/>'
        if with_occluder else ""
    )
    robot_link = (
        '<geom name="h1_link" type="box" size="0.1 0.1 0.1" pos="0.5 0 0"/>'
        if with_robot_link else ""
    )
    xml = f"""
    <mujoco model="ray_test">
      <worldbody>
        <camera name="head" mode="fixed" pos="0 0 0" xyaxes="0 -1 0 0 0 1"/>
        <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 -1"/>
        {occluder}
        {robot_link}
        <body name="target_0" pos="2.0 0 0">
          <geom name="target_0_col" type="sphere" size="0.05"/>
        </body>
        <body name="target_1" pos="2.0 0 0.5">
          <geom name="target_1_col" type="sphere" size="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class TestCheckJointVisibility:
    def test_unobstructed_single_target_is_visible(self):
        model, data = _make_model(with_occluder=False)
        targets = np.array([[2.0, 0.0, 0.0]])
        visible = check_joint_visibility(model, data, "head", targets)
        assert visible.shape == (1,)
        assert visible.dtype == bool
        assert visible[0] == True  # noqa: E712

    def test_occluded_target_is_hidden(self):
        model, data = _make_model(with_occluder=True)
        targets = np.array([[2.0, 0.0, 0.0]])
        visible = check_joint_visibility(model, data, "head", targets)
        assert visible[0] == False  # noqa: E712

    def test_batch_query_partial_occlusion(self):
        """Wall at x=1 extends y in [-0.5, 0.5]. target_0 is behind it;
        target_1 at z=0.5 is on the edge and may or may not be occluded — so
        we point a high target (z=1.5) over the wall for the 'visible' case."""
        model, data = _make_model(with_occluder=True)
        targets = np.array([
            [2.0, 0.0, 0.0],   # behind wall → occluded
            [2.0, 0.0, 1.5],   # above wall  → visible
        ])
        visible = check_joint_visibility(model, data, "head", targets)
        assert visible[0] == False  # noqa: E712
        assert visible[1] == True   # noqa: E712

    def test_exclude_geoms_skips_own_links(self):
        """A robot link in the ray path should not flag the target as occluded
        when its geom id is listed in exclude_geoms — otherwise the robot
        occludes itself from its own head camera constantly."""
        model, data = _make_model(with_occluder=False, with_robot_link=True)
        targets = np.array([[2.0, 0.0, 0.0]])

        # Without exclusion the link blocks the ray.
        visible_blocked = check_joint_visibility(model, data, "head", targets)
        assert visible_blocked[0] == False  # noqa: E712

        robot_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "h1_link"
        )
        visible_ok = check_joint_visibility(
            model, data, "head", targets, exclude_geoms=[robot_geom_id]
        )
        assert visible_ok[0] == True  # noqa: E712

    def test_unknown_camera_raises(self):
        model, data = _make_model(with_occluder=False)
        targets = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="camera"):
            check_joint_visibility(model, data, "does_not_exist", targets)

    def test_empty_targets_returns_empty_array(self):
        model, data = _make_model(with_occluder=False)
        targets = np.zeros((0, 3))
        visible = check_joint_visibility(model, data, "head", targets)
        assert visible.shape == (0,)
        assert visible.dtype == bool
