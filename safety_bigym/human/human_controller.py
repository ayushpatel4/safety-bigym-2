"""
Coworker Motion Controller

Drives the Unitree G1 coworker (see assets/g1_human_body.xml). The G1 root
(``pelvis``) is a mocap body teleported along a scripted trajectory; the body
joints are PD-held to a fixed standing pose. AMASS playback has been dropped —
the only disruption type is COWORKER, whose root path is fully scripted by the
TrajectoryPlanner and whose arm reaches come from an IK callback during loiter.

Integrates with:
- PD controller for joint target tracking
- HumanIK for arm reaching during disruptions
- TrajectoryPlanner for scripted root motion
- ScenarioParams from scenario sampler for configuration
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Callable, Dict

from safety_bigym.human.pd_controller import PDController, PDGains
from safety_bigym.human.human_ik import HumanIK
from safety_bigym.human.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryConfig,
    TrajectoryType,
)
from safety_bigym.human import g1_spec
from safety_bigym.scenarios.scenario_sampler import ScenarioParams


class HumanController:
    """
    Controller for the G1 coworker.

    The root pose comes from the scripted TrajectoryPlanner; body joints are
    PD-held to a fixed standing pose, except the active arm during loiter,
    which blends toward an IK reach target supplied by the COWORKER callback.
    """

    # G1 actuated joint names (single-DoF hinges; no _x/_y/_z sub-joints).
    BODY_JOINT_NAMES = g1_spec.BODY_JOINT_NAMES

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gains: Optional[PDGains] = None,
    ):
        """
        Initialize human controller.
        
        Args:
            model: MuJoCo model containing SMPL-H human
            data: MuJoCo simulation data
            gains: PD controller gains
        """
        self.model = model
        self.data = data
        
        # Initialize PD controller
        self.pd_controller = PDController(model, data, gains)
        
        # Initialize IK solver
        self.ik_solver = HumanIK(model, data)
        
        # No motion clip any more (AMASS dropped). Kept as an always-None
        # attribute so callers that probe `.clip` degrade gracefully.
        self.clip = None
        self.scenario: Optional[ScenarioParams] = None
        self.t = 0.0
        
        # Trajectory planner (NEW: controls root XY/yaw)
        self._trajectory_planner: Optional[TrajectoryPlanner] = None
        
        # IK target callback (to be set by scenario)
        self._ik_target_callback: Optional[Callable[[dict], np.ndarray]] = None
        
        # Random generator for IK noise
        self._rng = np.random.default_rng()
        
        # Build joint name to qpos index mapping
        self._build_joint_mapping()

        # Look up the pelvis mocap index — root pose is written to
        # data.mocap_pos / data.mocap_quat rather than qpos[0:7].
        pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, g1_spec.PELVIS_BODY
        )
        if pelvis_id >= 0:
            mid = int(self.model.body_mocapid[pelvis_id])
            self._mocap_id = mid if mid >= 0 else -1
        else:
            self._mocap_id = -1

        # Per-step root pose (written to mocap each step). Tracked separately
        # from qpos because the pelvis is a mocap body and has no qpos entries.
        self._root_pose = np.zeros(7)
        self._root_pose[3] = 1.0  # identity quaternion (w,x,y,z)

        # Fixed G1 standing pose for body joints; matches qpos size.
        self._standing_pose = g1_spec.standing_qpos(self.model)

        # Root position offset (to shift AMASS motion to spawn position)
        self._root_offset = np.zeros(3)

        # Root yaw rotation (rotate AMASS motion direction toward robot)
        self._root_yaw = 0.0  # radians
        self._clip_origin = np.zeros(3)  # First frame root position
    
    def set_root_offset(self, spawn_pos: np.ndarray, clip_origin: Optional[np.ndarray] = None):
        """Store the coworker spawn XY (root anchor for the fallback path).

        The scripted TrajectoryPlanner supplies world-frame root XY directly,
        so this only matters when no planner is set.
        """
        self._clip_origin = np.zeros(3)
        self._root_offset = np.array([spawn_pos[0], spawn_pos[1], 0.0])

    def set_root_yaw(self, yaw: float):
        """Store the desired facing yaw (used only by the fallback path)."""
        self._root_yaw = yaw
    
    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        """Create quaternion [w, x, y, z] for rotation around Z axis."""
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    
    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    def _build_joint_mapping(self):
        """Build mapping from joint names to qpos indices."""
        self.joint_to_qpos: Dict[str, int] = {}
        
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_to_qpos[name] = self.model.jnt_qposadr[i]
    
    def load_clip(self, clip_path: str, include_hands: bool = False):
        """No-op: AMASS playback has been dropped for the G1 coworker.

        Retained so the env's reset path (gated on an empty ``clip_path``)
        and any legacy callers don't break.
        """
        self.t = 0.0

    def set_scenario(self, scenario: ScenarioParams):
        """
        Set the current scenario parameters.
        
        Args:
            scenario: Scenario configuration
        """
        self.scenario = scenario
    
    def set_ik_callback(self, callback: Callable[[dict], np.ndarray]):
        """
        Set callback for computing IK targets.
        
        Args:
            callback: Function that takes robot_state dict and returns
                     target qpos array for the human
        """
        self._ik_target_callback = callback
    
    def set_trajectory_planner(self, planner: TrajectoryPlanner):
        """
        Set trajectory planner for scripted root (pelvis) motion. Body joints
        stay at the standing pose; the planner only drives root XY/yaw.
        """
        self._trajectory_planner = planner

    def reset(self):
        """Reset controller state and apply the standing pose."""
        self.t = 0.0
        self._trajectory_planner = None
        self._apply_standing_pose()

    def _get_amass_targets(self, t: float):
        """
        Get body-joint targets and the root pose at time t.

        Body joints are the fixed standing pose. Root XY/yaw come from the
        TrajectoryPlanner when set (Z fixed at the G1 standing height); with no
        planner, the current mocap pose is held.

        (Name kept for compatibility — there is no AMASS clip any more.)

        Returns:
            Tuple (qpos_targets, root_pose) where qpos_targets is a full-sized
            qpos buffer with the standing body-joint angles set, and root_pose
            is a (7,) array [x, y, z, qw, qx, qy, qz] for data.mocap_pos/quat.
        """
        if self.clip is None:
            # Standing pose fallback. If a trajectory planner is set,
            # still drive the pelvis from the planner (XY/yaw) so the
            # human visits the loiter waypoint even without an AMASS
            # clip — matches G1HumanController behaviour. Z stays at
            # the current mocap Z (no clip to source it from).
            targets = self._standing_pose.copy()
            root_pose = self._root_pose.copy()
            if self._mocap_id >= 0:
                root_pose[0:3] = self.data.mocap_pos[self._mocap_id]
                root_pose[3:7] = self.data.mocap_quat[self._mocap_id]
            if self._trajectory_planner is not None:
                px, py, plan_yaw, _phase = self._trajectory_planner.get_pose(t)
                root_pose[0] = px
                root_pose[1] = py
                root_pose[3:7] = self._quat_from_yaw(plan_yaw)
            return targets, root_pose

        # Apply speed multiplier
        speed = self.scenario.speed_multiplier if self.scenario else 1.0
        frame_idx = self.clip.get_time_frame(t * speed)

        # Get motion data
        joint_angles, root_trans, root_quat = self.clip.get_frame(frame_idx)

        # Build target qpos for body joints (root not in qpos any more)
        targets = self.data.qpos.copy()
        root_pose = np.empty(7)

        # --- Root position and orientation ---
        if self._trajectory_planner is not None:
            px, py, plan_yaw, _phase = self._trajectory_planner.get_pose(t)
            root_pose[0] = px
            root_pose[1] = py
            root_pose[2] = g1_spec.STANDING_PELVIS_Z
            root_pose[3:7] = self._quat_from_yaw(plan_yaw)
        elif self._mocap_id >= 0:
            # No planner: hold the current mocap pose.
            root_pose[0:3] = self.data.mocap_pos[self._mocap_id]
            root_pose[3:7] = self.data.mocap_quat[self._mocap_id]

        return targets, root_pose

    def _get_ik_targets(self, robot_state: dict) -> np.ndarray:
        """Get qpos targets from IK solver (body joints only — IK does not
        modify the root pose).

        Returns the same qpos buffer as `_get_amass_targets`'s first return
        value when no IK callback is wired. External callbacks must
        likewise return a qpos-sized array.
        """
        if self._ik_target_callback is not None:
            # Inject the controller's reference time so callbacks can
            # drive their own state machines without re-reading data.time
            # (which is per-substep and not always synced after kinematic
            # mocap writes).
            state = dict(robot_state) if robot_state else {}
            state.setdefault("t", float(self.t))
            return self._ik_target_callback(state)
        # Default: hold current AMASS pose
        targets, _ = self._get_amass_targets(self.t)
        return targets

    def _write_root_mocap(self, root_pose: np.ndarray) -> None:
        """Write the 7-element root pose to the Pelvis mocap slot."""
        if self._mocap_id < 0:
            return
        self.data.mocap_pos[self._mocap_id] = root_pose[0:3]
        self.data.mocap_quat[self._mocap_id] = root_pose[3:7]

    def _apply_standing_pose(self):
        """Set the body-joint qpos to the fixed G1 standing pose."""
        for joint_name in self.BODY_JOINT_NAMES:
            qpos_idx = self.joint_to_qpos.get(joint_name)
            if qpos_idx is not None:
                self.data.qpos[qpos_idx] = self._standing_pose[qpos_idx]
        mujoco.mj_kinematics(self.model, self.data)

    def step(self, dt: float, robot_state: Optional[dict] = None):
        """
        Step the controller forward in time.

        When a TrajectoryPlanner is set, IK blending is driven by the
        planner's phase rather than a fixed trigger_time:
        - "approach" / "walk" / "depart" → pure AMASS body joints
        - "loiter" → blend AMASS body + IK arm reaching

        The root pose (Pelvis) is always sourced from AMASS / the
        trajectory planner — IK never modifies it. Root pose is written
        to data.mocap_pos / data.mocap_quat each step; body joints are
        PD-controlled.
        """
        robot_state = robot_state or {}

        # Always compute amass targets first; root pose comes from here.
        amass_targets, root_pose = self._get_amass_targets(self.t)

        # Determine current phase and blend body qpos accordingly.
        if self._trajectory_planner is not None:
            _, _, _, phase = self._trajectory_planner.get_pose(self.t)

            if phase == "loiter":
                blend = self.scenario.blend_duration if self.scenario else 0.4
                loiter_start = self._get_loiter_start_time()
                loiter_elapsed = self.t - loiter_start
                ik_targets = self._get_ik_targets(robot_state)
                if loiter_elapsed < blend:
                    alpha = loiter_elapsed / blend
                    targets = (1 - alpha) * amass_targets + alpha * ik_targets
                else:
                    targets = ik_targets
            elif phase == "depart":
                blend = self.scenario.blend_duration if self.scenario else 0.4
                loiter_end = self._get_loiter_end_time()
                depart_elapsed = self.t - loiter_end
                if depart_elapsed < blend:
                    alpha = 1.0 - (depart_elapsed / blend)
                    ik_targets = self._get_ik_targets(robot_state)
                    targets = (1 - alpha) * amass_targets + alpha * ik_targets
                else:
                    targets = amass_targets
            else:
                # "approach" or "walk" → pure AMASS
                targets = amass_targets
        else:
            # Legacy mode: fixed trigger_time-based phase switching
            trigger = self.scenario.trigger_time if self.scenario else float('inf')
            blend = self.scenario.blend_duration if self.scenario else 0.4

            if self.t < trigger:
                targets = amass_targets
            elif self.t < trigger + blend:
                alpha = (self.t - trigger) / blend
                ik_targets = self._get_ik_targets(robot_state)
                targets = (1 - alpha) * amass_targets + alpha * ik_targets
            else:
                targets = self._get_ik_targets(robot_state)

        # Body joints: PD control
        self.pd_controller.set_targets(targets)
        self.pd_controller.apply_control()

        # Root: mocap teleport (kinematic, refreshes collision broadphase
        # via the next mj_step)
        self._write_root_mocap(root_pose)

        # Advance time
        self.t += dt
    
    def _get_loiter_start_time(self) -> float:
        """Get time when loiter phase starts from trajectory planner."""
        if self._trajectory_planner is None:
            return float('inf')
        
        for wp in self._trajectory_planner.waypoints:
            if wp.phase == "loiter":
                return wp.time
        return float('inf')
    
    def _get_loiter_end_time(self) -> float:
        """Get time when loiter phase ends from trajectory planner."""
        if self._trajectory_planner is None:
            return float('inf')
        
        loiter_end = float('inf')
        in_loiter = False
        for wp in self._trajectory_planner.waypoints:
            if wp.phase == "loiter":
                in_loiter = True
            elif in_loiter:
                # First waypoint after loiter
                loiter_end = wp.time
                break
        return loiter_end
    
    @property
    def current_phase(self) -> str:
        """Get current motion phase name."""
        if self._trajectory_planner is not None:
            _, _, _, phase = self._trajectory_planner.get_pose(self.t)
            return phase
        
        if self.scenario is None:
            return "amass"
        
        trigger = self.scenario.trigger_time
        blend = self.scenario.blend_duration
        
        if self.t < trigger:
            return "amass"
        elif self.t < trigger + blend:
            return "blending"
        else:
            return "ik"

