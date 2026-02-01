"""
Inverse Kinematics for SMPL-H Humanoid

Implements Jacobian-based Inverse Kinematics (IK) to allow the human model
to reach for targets (e.g., robot end-effector or objects).

Uses Damped Least Squares (DLS) method for numerical stability near singularities.

Key features:
- Position-only IK (3D target, not 6D pose)
- Joint limit enforcement
- Works on data copy to avoid disturbing simulation state
- Supports both left and right arm chains
"""

import numpy as np
import mujoco
from typing import List, Optional, Tuple, Dict


class HumanIK:
    """
    Jacobian-based IK solver for SMPL-H model.
    
    Supports reaching with arm chains (Shoulder -> Elbow -> Wrist).
    Only the arm chain is controlled by IK; body stays on AMASS playback.
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize IK solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data (used as reference for copying)
        """
        self.model = model
        self.data = data
        
        # Create a working copy of data for IK iterations
        self._ik_data = mujoco.MjData(model)
        
        # Cache for joint/DoF indices
        self._chain_cache: Dict[str, dict] = {}
        
        # Define arm chains with explicit joint names
        # These match the SMPL-H MJCF joint naming convention
        self.chains = {
            "right_arm": {
                "joints": ["R_Shoulder", "R_Elbow", "R_Wrist"],
                "end_effector": "R_Wrist",
            },
            "left_arm": {
                "joints": ["L_Shoulder", "L_Elbow", "L_Wrist"],
                "end_effector": "L_Wrist",
            },
        }
        
        # Pre-compute chain data
        for chain_name in self.chains:
            self._build_chain_cache(chain_name)
    
    def _build_chain_cache(self, chain_name: str):
        """Build and cache joint indices for a chain."""
        chain = self.chains[chain_name]
        joint_names = chain["joints"]
        
        joint_ids = []
        dof_indices = []
        qpos_indices = []
        
        for name in joint_names:
            # Each SMPL-H joint has 3 hinge sub-joints (x, y, z)
            for axis in ["x", "y", "z"]:
                full_name = f"{name}_{axis}"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                if jid >= 0:
                    joint_ids.append(jid)
                    dof_indices.append(self.model.jnt_dofadr[jid])
                    qpos_indices.append(self.model.jnt_qposadr[jid])
        
        # Get end-effector body ID
        ee_name = chain["end_effector"]
        ee_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        
        self._chain_cache[chain_name] = {
            "joint_ids": joint_ids,
            "dof_indices": dof_indices,
            "qpos_indices": qpos_indices,
            "ee_body_id": ee_bid,
            "n_dof": len(dof_indices),
        }
    
    def get_end_effector_pos(self, chain_name: str) -> np.ndarray:
        """Get current end-effector position for a chain."""
        cache = self._chain_cache[chain_name]
        return self.data.xpos[cache["ee_body_id"]].copy()
    
    def select_arm(self, target_pos: np.ndarray) -> str:
        """
        Select which arm to use based on target position.
        
        Args:
            target_pos: Target position in world coordinates
            
        Returns:
            'right_arm' or 'left_arm'
        """
        # Get current pelvis position to determine human facing direction
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
        if pelvis_id < 0:
            return "right_arm"  # Fallback
        
        pelvis_pos = self.data.xpos[pelvis_id]
        
        # Get shoulder positions
        right_pos = self.get_end_effector_pos("right_arm")
        left_pos = self.get_end_effector_pos("left_arm")
        
        # Pick the closer arm
        right_dist = np.linalg.norm(target_pos - right_pos)
        left_dist = np.linalg.norm(target_pos - left_pos)
        
        return "right_arm" if right_dist < left_dist else "left_arm"
    
    def solve(
        self,
        chain_name: str,
        target_pos: np.ndarray,
        max_iterations: int = 50,
        tolerance: float = 0.01,
        damping: float = 0.01,
        step_size: float = 0.5,
    ) -> np.ndarray:
        """
        Solve IK to place end-effector at target position.
        
        Args:
            chain_name: 'right_arm' or 'left_arm'
            target_pos: Target position (x, y, z) in world coordinates
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance in meters
            damping: Damping factor for DLS (prevents singularity explosion)
            step_size: Update step size (0.0 to 1.0)
            
        Returns:
            Array of arm joint angles that achieve the target
        """
        cache = self._chain_cache[chain_name]
        dof_indices = cache["dof_indices"]
        qpos_indices = cache["qpos_indices"]
        joint_ids = cache["joint_ids"]
        ee_bid = cache["ee_body_id"]
        n_dof = cache["n_dof"]
        
        # Copy current qpos to working data
        # (mj_copyData not available in Python bindings, so we copy qpos/qvel manually)
        self._ik_data.qpos[:] = self.data.qpos[:]
        self._ik_data.qvel[:] = self.data.qvel[:]
        
        for iteration in range(max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self._ik_data)
            
            # Current end-effector position
            current_pos = self._ik_data.xpos[ee_bid]
            
            # Compute error
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                break
            
            # Compute Jacobian for this body
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self._ik_data, jacp, None, 
                         current_pos, ee_bid)
            
            # Extract columns for arm chain DoFs only
            J = jacp[:, dof_indices]  # (3, n_dof)
            
            # Damped pseudoinverse: J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹
            J_T = J.T
            JJ_T = J @ J_T
            damping_term = damping * np.eye(3)
            
            J_pinv = J_T @ np.linalg.inv(JJ_T + damping_term)
            
            # Compute joint angle update
            dq = J_pinv @ error
            
            # Apply update with step size
            for i, qpos_idx in enumerate(qpos_indices):
                self._ik_data.qpos[qpos_idx] += step_size * dq[i]
            
            # Clip to joint limits
            for i, jid in enumerate(joint_ids):
                qpos_idx = qpos_indices[i]
                lo = self.model.jnt_range[jid, 0]
                hi = self.model.jnt_range[jid, 1]
                # Only clip if joint has limits (lo < hi)
                if lo < hi:
                    self._ik_data.qpos[qpos_idx] = np.clip(
                        self._ik_data.qpos[qpos_idx], lo, hi
                    )
        
        # Return the arm joint angles
        return np.array([self._ik_data.qpos[idx] for idx in qpos_indices])
    
    def solve_with_selection(
        self,
        target_pos: np.ndarray,
        **kwargs
    ) -> Tuple[str, np.ndarray]:
        """
        Solve IK with automatic arm selection.
        
        Args:
            target_pos: Target position
            **kwargs: Additional arguments passed to solve()
            
        Returns:
            Tuple of (chain_name, joint_angles)
        """
        chain_name = self.select_arm(target_pos)
        angles = self.solve(chain_name, target_pos, **kwargs)
        return chain_name, angles
