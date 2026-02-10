"""
PD Controller for SMPL-H Human Motion

A proportional-derivative controller that tracks target joint positions
for the SMPL-H humanoid. Works with MuJoCo's position actuators.
"""

import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PDGains:
    """PD controller gains for different joint groups."""
    # Default gains - can be tuned per body region
    kp: float = 100.0  # Proportional gain
    kd: float = 10.0   # Derivative gain
    

class PDController:
    """
    PD controller for tracking joint targets on the SMPL-H model.
    
    Computes torques based on position error and velocity:
        tau = kp * (q_target - q) - kd * dq
    
    For MuJoCo position actuators, we set the actuator ctrl directly
    to the target position, letting MuJoCo's actuator dynamics handle
    the PD control internally.
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gains: Optional[PDGains] = None,
    ):
        """
        Initialize PD controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            gains: PD gains (optional, uses defaults if not provided)
        """
        self.model = model
        self.data = data
        self.gains = gains or PDGains()
        
        # Build actuator name to index mapping
        self.actuator_name_to_idx: Dict[str, int] = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_name_to_idx[name] = i
        
        # Build joint name to qpos index mapping
        self.joint_name_to_qpos: Dict[str, int] = {}
        self.joint_name_to_qvel: Dict[str, int] = {}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_name_to_qpos[name] = model.jnt_qposadr[i]
                self.joint_name_to_qvel[name] = model.jnt_dofadr[i]
        
        # Initialize targets to current pose
        self.target_qpos = data.qpos.copy()
        
    def set_targets(self, targets: np.ndarray):
        """
        Set target joint positions.
        
        Args:
            targets: Target qpos values (full qpos array)
        """
        self.target_qpos = targets.copy()
    
    def set_joint_targets(self, joint_targets: Dict[str, float]):
        """
        Set target positions for specific joints by name.
        
        Args:
            joint_targets: Dict mapping joint names to target values
        """
        for name, value in joint_targets.items():
            if name in self.joint_name_to_qpos:
                idx = self.joint_name_to_qpos[name]
                self.target_qpos[idx] = value
    
    def compute_control(self) -> np.ndarray:
        """
        Compute control signals based on current state and targets.
        
        For position actuators, this returns the target positions.
        For torque actuators, this would return PD-computed torques.
        
        Returns:
            Control signal array (nu,) - only human actuators are set
        """
        # Start with current ctrl to preserve robot controls
        ctrl = self.data.ctrl.copy()
        
        for name, act_idx in self.actuator_name_to_idx.items():
            # Only control human actuators (act_* prefix)
            if not name.startswith("act_"):
                continue
                
            # Get the joint this actuator controls
            joint_id = self.model.actuator_trnid[act_idx, 0]
            if joint_id >= 0:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                ctrl[act_idx] = self.target_qpos[qpos_idx]
        
        return ctrl
    
    def compute_torque_control(self) -> np.ndarray:
        """
        Compute explicit PD torques (for general actuators).
        
        Returns:
            Torque array (nu,)
        """
        torques = np.zeros(self.model.nu)
        
        for name, act_idx in self.actuator_name_to_idx.items():
            joint_id = self.model.actuator_trnid[act_idx, 0]
            if joint_id >= 0:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                qvel_idx = self.model.jnt_dofadr[joint_id]
                
                # Position error
                pos_error = self.target_qpos[qpos_idx] - self.data.qpos[qpos_idx]
                
                # Velocity (we want to dampen it)
                vel = self.data.qvel[qvel_idx]
                
                # PD control law
                torques[act_idx] = self.gains.kp * pos_error - self.gains.kd * vel
        
        return torques
    
    def apply_control(self):
        """Apply computed control to the simulation."""
        self.data.ctrl[:] = self.compute_control()
    
    def step(self):
        """Compute and apply control for one timestep."""
        self.apply_control()
