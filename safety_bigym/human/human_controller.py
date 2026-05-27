"""
Human Motion Controller

Controls the SMPL-H humanoid using AMASS motion playback with optional
IK blending for disruption scenarios. Implements the motion blending
architecture from the implementation plan.

Integrates with:
- AMASS motion loader for clip playback
- PD controller for joint target tracking
- HumanIK for arm reaching during disruptions
- ScenarioParams from scenario sampler for configuration
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Callable, Dict

from safety_bigym.motion.amass_loader import load_amass_clip, MotionClip
from safety_bigym.human.pd_controller import PDController, PDGains
from safety_bigym.human.human_ik import HumanIK
from safety_bigym.human.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryConfig,
    TrajectoryType,
)
from safety_bigym.scenarios.scenario_sampler import ScenarioParams


class HumanController:
    """
    Controller for SMPL-H humanoid motion.
    
    Manages motion playback from AMASS data and blending to IK targets
    at scenario trigger times. The controller operates in three phases:
    
    1. t < trigger: Pure AMASS motion playback
    2. trigger <= t < trigger + blend: Interpolate AMASS -> IK targets
    3. t >= trigger + blend: Pure IK-driven disruption motion
    """
    
    # Joint names from SMPL-H (matching MJCF actuator naming)
    BODY_JOINT_NAMES = [
        "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
        "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck",
        "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    ]
    
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
        
        # Motion clip and playback state
        self.clip: Optional[MotionClip] = None
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

        # Look up the Pelvis mocap index — root pose is written to
        # data.mocap_pos / data.mocap_quat rather than qpos[0:7].
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
        if pelvis_id >= 0:
            mid = int(self.model.body_mocapid[pelvis_id])
            self._mocap_id = mid if mid >= 0 else -1
        else:
            self._mocap_id = -1

        # Per-step root pose (written to mocap each step). Tracked separately
        # from qpos because Pelvis is a mocap body and has no qpos entries.
        self._root_pose = np.zeros(7)
        self._root_pose[3] = 1.0  # identity quaternion (w,x,y,z)

        # Standing pose for body joints only; matches qpos size.
        self._standing_pose = np.zeros(self.model.nq)

        # Root position offset (to shift AMASS motion to spawn position)
        self._root_offset = np.zeros(3)

        # Root yaw rotation (rotate AMASS motion direction toward robot)
        self._root_yaw = 0.0  # radians
        self._clip_origin = np.zeros(3)  # First frame root position
    
    def set_root_offset(self, spawn_pos: np.ndarray, clip_origin: Optional[np.ndarray] = None):
        """
        Set root offset to shift AMASS motion to spawn position.
        
        Only offsets X and Y positions - Z is preserved from AMASS motion
        since it contains the correct pelvis height for standing.
        
        Args:
            spawn_pos: Desired spawn position [x, y, z] (z typically 0 for floor)
            clip_origin: AMASS clip's first frame root position (auto-detected if None)
        """
        if clip_origin is None and self.clip is not None:
            # Get first frame root position from clip
            _, root_trans, _ = self.clip.get_frame(0)
            clip_origin = root_trans
        
        if clip_origin is not None:
            self._clip_origin = clip_origin.copy()
            # Offset is computed AFTER rotation, so just store spawn XY
            # The actual offset application happens in _get_amass_targets
            self._root_offset = np.array([spawn_pos[0], spawn_pos[1], 0.0])
        else:
            self._clip_origin = np.zeros(3)
            self._root_offset = np.array([spawn_pos[0], spawn_pos[1], 0.0])
    
    def set_root_yaw(self, yaw: float):
        """
        Set yaw rotation to apply to AMASS motion direction.
        
        This rotates the clip's root trajectory around the clip's origin
        so the human's movement direction faces toward the robot.
        
        Args:
            yaw: Desired facing direction in radians (toward robot)
        """
        if self.clip is not None:
            # Determine AMASS clip's natural forward direction from first few frames
            _, start_pos, _ = self.clip.get_frame(0)
            # Use a frame ~1 second in (or last frame) to find direction
            end_idx = min(30, self.clip.num_frames - 1)
            _, end_pos, _ = self.clip.get_frame(end_idx)
            
            clip_dir = end_pos[:2] - start_pos[:2]  # XY direction
            if np.linalg.norm(clip_dir) > 0.01:
                clip_yaw = np.arctan2(clip_dir[1], clip_dir[0])
            else:
                clip_yaw = 0.0  # Clip doesn't move much, assume facing +X
            
            # Rotation needed = desired yaw - clip's natural yaw
            self._root_yaw = yaw - clip_yaw
        else:
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
        """
        Load an AMASS motion clip.
        
        Args:
            clip_path: Path to AMASS .npz file
            include_hands: Whether to include hand joint data
        """
        self.clip = load_amass_clip(clip_path, include_hands=include_hands)
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
        Set trajectory planner for root motion control.
        
        When a planner is set, it overrides the AMASS root trajectory.
        Body joint angles still come from AMASS clip playback.
        
        Args:
            planner: TrajectoryPlanner instance
        """
        self._trajectory_planner = planner
    
    def reset(self):
        """Reset controller state."""
        self.t = 0.0
        self._trajectory_planner = None
        if self.clip is not None:
            # Set initial pose from clip
            self._apply_amass_frame(0)
    
    def _get_amass_targets(self, t: float):
        """
        Get joint targets from AMASS motion at time t.

        If a TrajectoryPlanner is set, root XY and yaw come from the planner.
        Body joint angles always come from the AMASS clip.

        Returns:
            Tuple (qpos_targets, root_pose) where qpos_targets is the
            full-sized qpos buffer with body joint angles set, and
            root_pose is a (7,) array [x, y, z, qw, qx, qy, qz] meant
            to be written to data.mocap_pos / data.mocap_quat.
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
            # USE TRAJECTORY PLANNER for root XY and yaw
            px, py, plan_yaw, phase = self._trajectory_planner.get_pose(t)
            root_pose[0] = px
            root_pose[1] = py
            root_pose[2] = root_trans[2]  # Z from AMASS (pelvis height)
            root_pose[3:7] = self._quat_from_yaw(plan_yaw)
        else:
            # Fall back to original AMASS root motion with offset/yaw
            pos_centered = root_trans - self._clip_origin
            cos_y = np.cos(self._root_yaw)
            sin_y = np.sin(self._root_yaw)
            rotated_x = cos_y * pos_centered[0] - sin_y * pos_centered[1]
            rotated_y = sin_y * pos_centered[0] + cos_y * pos_centered[1]
            root_pose[0] = rotated_x + self._root_offset[0]
            root_pose[1] = rotated_y + self._root_offset[1]
            root_pose[2] = root_trans[2]
            if abs(self._root_yaw) > 1e-6:
                yaw_quat = self._quat_from_yaw(self._root_yaw)
                root_pose[3:7] = self._quat_multiply(yaw_quat, root_quat)
            else:
                root_pose[3:7] = root_quat

        # --- Body joint angles always from AMASS ---
        for joint_idx, joint_name in enumerate(self.BODY_JOINT_NAMES):
            for axis_idx, axis in enumerate(["x", "y", "z"]):
                full_name = f"{joint_name}_{axis}"
                if full_name in self.joint_to_qpos:
                    qpos_idx = self.joint_to_qpos[full_name]
                    # joint_angles[0] is Pelvis (root), skip it
                    targets[qpos_idx] = joint_angles[joint_idx + 1, axis_idx]

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

    def _apply_amass_frame(self, frame_idx: int):
        """Directly set qpos from AMASS frame (for initialization)."""
        if self.clip is None:
            return

        joint_angles, root_trans, root_quat = self.clip.get_frame(frame_idx)

        # Root pose -> mocap (Pelvis is a mocap body, not in qpos)
        if self._mocap_id >= 0:
            self.data.mocap_pos[self._mocap_id] = root_trans
            self.data.mocap_quat[self._mocap_id] = root_quat

        # Set body joint qpos
        for joint_idx, joint_name in enumerate(self.BODY_JOINT_NAMES):
            for axis_idx, axis in enumerate(["x", "y", "z"]):
                full_name = f"{joint_name}_{axis}"
                if full_name in self.joint_to_qpos:
                    qpos_idx = self.joint_to_qpos[full_name]
                    self.data.qpos[qpos_idx] = joint_angles[joint_idx + 1, axis_idx]

        # Forward kinematics
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

