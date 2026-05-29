"""AMASS-free SMPL-H controller — procedural motion like G1.

Parallel to :class:`G1HumanController` and
:class:`safety_bigym.human.human_controller.HumanController`. Selected when
``HumanConfig.human_model == "smplh"`` and
``HumanConfig.smplh_motion == "procedural"``.

The SMPL-H MJCF asset is unchanged; only motion synthesis differs:

- **No AMASS clip** — body joints PD-track a fixed standing pose
- **Pelvis** — ``TrajectoryPlanner`` writes mocap XY + yaw (walk / patrol)
- **Arms** — ``CoworkerArmController`` IK callback during loiter only
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import mujoco
import numpy as np

from safety_bigym.human import smplh_procedural_spec
from safety_bigym.human.pd_controller import PDController, PDGains
from safety_bigym.human.trajectory_planner import TrajectoryPlanner
from safety_bigym.scenarios.scenario_sampler import ScenarioParams


class SmplhProceduralController:
    """SMPL-H standing-pose controller with trajectory-driven pelvis mocap."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gains: Optional[PDGains] = None,
        standing_pelvis_z: float = smplh_procedural_spec.STANDING_PELVIS_Z,
    ):
        self.model = model
        self.data = data
        self._standing_pelvis_z = float(standing_pelvis_z)

        self.pd_controller = PDController(model, data, gains)

        self.clip = None
        self.scenario: Optional[ScenarioParams] = None
        self.t = 0.0

        self._trajectory_planner: Optional[TrajectoryPlanner] = None
        self._ik_target_callback: Optional[Callable[[dict], np.ndarray]] = None

        self._build_joint_mapping()

        pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, smplh_procedural_spec.PELVIS_BODY
        )
        if pelvis_id >= 0:
            mid = int(self.model.body_mocapid[pelvis_id])
            self._mocap_id = mid if mid >= 0 else -1
        else:
            self._mocap_id = -1

        self._root_pose = np.zeros(7)
        self._root_pose[2] = self._standing_pelvis_z
        self._root_pose[3] = 1.0

        self._standing_pose = smplh_procedural_spec.build_standing_qpos(
            model, data
        )

        self._root_offset = np.zeros(3)
        self._root_yaw = 0.0

    def load_clip(self, clip_path: str, include_hands: bool = False) -> None:
        """No-op — procedural mode does not use AMASS."""
        return None

    def set_scenario(self, scenario: ScenarioParams) -> None:
        self.scenario = scenario

    def set_ik_callback(
        self, callback: Callable[[dict], np.ndarray]
    ) -> None:
        self._ik_target_callback = callback

    def set_trajectory_planner(self, planner: TrajectoryPlanner) -> None:
        self._trajectory_planner = planner

    def set_root_offset(
        self,
        spawn_pos: np.ndarray,
        clip_origin: Optional[np.ndarray] = None,
    ) -> None:
        del clip_origin
        self._root_offset = np.array(
            [float(spawn_pos[0]), float(spawn_pos[1]), 0.0]
        )

    def set_root_yaw(self, yaw: float) -> None:
        self._root_yaw = float(yaw)

    def reset(self) -> None:
        self.t = 0.0
        self._trajectory_planner = None

    def step(self, dt: float, robot_state: Optional[dict] = None) -> None:
        robot_state = robot_state or {}

        rest_targets = self._standing_pose.copy()
        root_pose = self._compute_root_pose()

        targets = rest_targets
        if self._trajectory_planner is not None:
            _, _, _, phase = self._trajectory_planner.get_pose(self.t)
            if phase == "loiter" and self._ik_target_callback is not None:
                targets = self._get_ik_targets(robot_state)

        self.pd_controller.set_targets(targets)
        self.pd_controller.apply_control()
        self._write_root_mocap(root_pose)
        self.t += dt

    def _build_joint_mapping(self) -> None:
        self.joint_to_qpos: Dict[str, int] = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_to_qpos[name] = int(self.model.jnt_qposadr[i])

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    def _compute_root_pose(self) -> np.ndarray:
        root_pose = self._root_pose.copy()
        if self._trajectory_planner is not None:
            px, py, plan_yaw, _phase = self._trajectory_planner.get_pose(self.t)
            root_pose[0] = px
            root_pose[1] = py
            root_pose[2] = self._standing_pelvis_z
            root_pose[3:7] = self._quat_from_yaw(plan_yaw)
        else:
            root_pose[0] = self._root_offset[0]
            root_pose[1] = self._root_offset[1]
            root_pose[2] = self._standing_pelvis_z
            root_pose[3:7] = self._quat_from_yaw(self._root_yaw)
        return root_pose

    def _write_root_mocap(self, root_pose: np.ndarray) -> None:
        if self._mocap_id < 0:
            return
        self.data.mocap_pos[self._mocap_id] = root_pose[0:3]
        self.data.mocap_quat[self._mocap_id] = root_pose[3:7]

    def _get_ik_targets(self, robot_state: dict) -> np.ndarray:
        if self._ik_target_callback is None:
            return self._standing_pose.copy()
        state = dict(robot_state) if robot_state else {}
        state.setdefault("t", float(self.t))
        return self._ik_target_callback(state)

    @property
    def current_phase(self) -> str:
        if self._trajectory_planner is not None:
            _, _, _, phase = self._trajectory_planner.get_pose(self.t)
            return phase
        return "standing"
