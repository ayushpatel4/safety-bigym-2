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
from dataclasses import dataclass

from safety_bigym.motion.amass_loader import load_amass_clip, MotionClip
from safety_bigym.human.pd_controller import PDController, PDGains
from safety_bigym.human.human_ik import HumanIK


# Re-export ScenarioParams for backwards compatibility
# (New code should import from safety_bigym.scenarios)
@dataclass 
class ScenarioParams:
    """Parameters for a human behavior scenario.
    
    Note: For full scenario configuration including disruption types,
    use safety_bigym.scenarios.ScenarioParams instead.
    """
    clip_path: str              # Path to AMASS motion clip
    trigger_time: float = 2.0   # Time when disruption starts (seconds)
    blend_duration: float = 0.4 # Blend duration between AMASS and IK (seconds)
    speed_multiplier: float = 1.0  # Motion playback speed


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
        
        # IK target callback (to be set by scenario)
        self._ik_target_callback: Optional[Callable[[dict], np.ndarray]] = None
        
        # Random generator for IK noise
        self._rng = np.random.default_rng()
        
        # Build joint name to qpos index mapping
        self._build_joint_mapping()
        
        # Store initial standing pose (all zeros for human joints)
        self._standing_pose = np.zeros(self.model.nq)
        # Keep root at initial position
        self._standing_pose[3:7] = [0, 0, 0, 1]  # Identity quaternion
        
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
        if scenario.clip_path:
            self.load_clip(scenario.clip_path)
    
    def set_ik_callback(self, callback: Callable[[dict], np.ndarray]):
        """
        Set callback for computing IK targets.
        
        Args:
            callback: Function that takes robot_state dict and returns
                     target qpos array for the human
        """
        self._ik_target_callback = callback
    
    def reset(self):
        """Reset controller state."""
        self.t = 0.0
        if self.clip is not None:
            # Set initial pose from clip
            self._apply_amass_frame(0)
    
    def _get_amass_targets(self, t: float) -> np.ndarray:
        """
        Get joint targets from AMASS motion at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            Target qpos array
        """
        if self.clip is None:
            # Return standing pose as fallback (keeps human upright)
            targets = self._standing_pose.copy()
            # Preserve current root position
            targets[0:7] = self.data.qpos[0:7]
            return targets
        
        # Apply speed multiplier
        speed = self.scenario.speed_multiplier if self.scenario else 1.0
        frame_idx = self.clip.get_time_frame(t * speed)
        
        # Get motion data
        joint_angles, root_trans, root_quat = self.clip.get_frame(frame_idx)
        
        # Build target qpos
        targets = self.data.qpos.copy()
        
        # Rotate root position around clip origin, then translate to spawn
        # 1. Center on clip origin
        pos_centered = root_trans - self._clip_origin
        
        # 2. Rotate XY by root yaw
        cos_y = np.cos(self._root_yaw)
        sin_y = np.sin(self._root_yaw)
        rotated_x = cos_y * pos_centered[0] - sin_y * pos_centered[1]
        rotated_y = sin_y * pos_centered[0] + cos_y * pos_centered[1]
        
        # 3. Translate to spawn position (XY from offset, Z from AMASS)
        targets[0] = rotated_x + self._root_offset[0]
        targets[1] = rotated_y + self._root_offset[1]
        targets[2] = root_trans[2]  # Keep original Z (pelvis height)
        
        # 4. Rotate quaternion orientation
        if abs(self._root_yaw) > 1e-6:
            yaw_quat = self._quat_from_yaw(self._root_yaw)
            targets[3:7] = self._quat_multiply(yaw_quat, root_quat)
        else:
            targets[3:7] = root_quat
        
        # Set joint angles
        for joint_idx, joint_name in enumerate(self.BODY_JOINT_NAMES):
            for axis_idx, axis in enumerate(["x", "y", "z"]):
                full_name = f"{joint_name}_{axis}"
                if full_name in self.joint_to_qpos:
                    qpos_idx = self.joint_to_qpos[full_name]
                    # joint_angles[0] is Pelvis (root), skip it
                    targets[qpos_idx] = joint_angles[joint_idx + 1, axis_idx]
        
        return targets
    
    def _get_ik_targets(self, robot_state: dict) -> np.ndarray:
        """
        Get joint targets from IK solver.
        
        Args:
            robot_state: Current robot state dict
            
        Returns:
            Target qpos array
        """
        if self._ik_target_callback is not None:
            return self._ik_target_callback(robot_state)
        else:
            # Default: hold current AMASS pose
            return self._get_amass_targets(self.t)
    
    def _apply_amass_frame(self, frame_idx: int):
        """Directly set qpos from AMASS frame (for initialization)."""
        if self.clip is None:
            return
            
        joint_angles, root_trans, root_quat = self.clip.get_frame(frame_idx)
        
        # Set root
        self.data.qpos[0:3] = root_trans
        self.data.qpos[3:7] = root_quat
        
        # Set joints
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
        
        Args:
            dt: Time step in seconds
            robot_state: Current robot state (for IK computation)
        """
        robot_state = robot_state or {}
        
        # Get scenario parameters
        trigger = self.scenario.trigger_time if self.scenario else float('inf')
        blend = self.scenario.blend_duration if self.scenario else 0.4
        
        # Compute targets based on phase
        if self.t < trigger:
            # Phase 1: Pure AMASS playback
            targets = self._get_amass_targets(self.t)
        elif self.t < trigger + blend:
            # Phase 2: Blend AMASS -> IK
            alpha = (self.t - trigger) / blend
            amass_targets = self._get_amass_targets(self.t)
            ik_targets = self._get_ik_targets(robot_state)
            targets = (1 - alpha) * amass_targets + alpha * ik_targets
        else:
            # Phase 3: Pure IK
            targets = self._get_ik_targets(robot_state)
        
        # Set targets and apply control
        self.pd_controller.set_targets(targets)
        self.pd_controller.apply_control()
        
        # Directly set root position/orientation (freejoint can't be PD controlled)
        # This ensures the human follows the motion trajectory
        self.data.qpos[0:7] = targets[0:7]
        
        # Advance time
        self.t += dt
    
    @property
    def current_phase(self) -> str:
        """Get current motion phase name."""
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
