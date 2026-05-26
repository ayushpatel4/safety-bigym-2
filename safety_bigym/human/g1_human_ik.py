"""Jacobian-based reach IK for the Unitree G1 humanoid.

Parallel to :mod:`safety_bigym.human.human_ik` (SMPL-H). Same public surface
(``select_arm``, ``solve``, ``get_end_effector_pos``,
``solve_with_selection``, ``chains`` dict, ``_ik_data``, ``_chain_cache``)
so :class:`safety_bigym.scenarios.coworker_behavior.CoworkerArmController`
can use either solver interchangeably.

Difference vs SMPL-H:

- G1 joints are single-DoF hinges (not axis-decomposed). The chain cache
  loops joint names directly instead of synthesising ``_x/_y/_z`` siblings.
- ``chains`` entries include a ``"shoulder_body"`` field so
  ``CoworkerArmController`` can read the active arm's shoulder body name
  through ``self.ik_solver.chains[arm]["shoulder_body"]`` without
  hard-coding SMPL-H names.
"""

from __future__ import annotations

from typing import Dict, Tuple

import mujoco
import numpy as np

from safety_bigym.human import g1_human_spec


class G1HumanIK:
    """Damped-least-squares IK solver for the G1 ``COWORKER`` reach arm."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        self._ik_data = mujoco.MjData(model)
        self._chain_cache: Dict[str, dict] = {}

        # Mirror HumanIK's ``chains`` schema so CoworkerArmController can
        # access ``end_effector`` and the new ``shoulder_body`` field
        # uniformly across both solvers. Pulled from the single spec module
        # so a model change only edits g1_human_spec.
        self.chains = {
            arm_name: {
                "joints": list(chain["joints"]),
                "end_effector": chain["end_effector"],
                "shoulder_body": chain["shoulder_body"],
            }
            for arm_name, chain in g1_human_spec.ARM_CHAINS.items()
        }

        for chain_name in self.chains:
            self._build_chain_cache(chain_name)

    def _build_chain_cache(self, chain_name: str) -> None:
        chain = self.chains[chain_name]
        joint_ids: list[int] = []
        dof_indices: list[int] = []
        qpos_indices: list[int] = []

        # G1: one hinge per name (no axis suffix loop).
        for joint_name in chain["joints"]:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if jid >= 0:
                joint_ids.append(jid)
                dof_indices.append(int(self.model.jnt_dofadr[jid]))
                qpos_indices.append(int(self.model.jnt_qposadr[jid]))

        ee_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, chain["end_effector"]
        )

        self._chain_cache[chain_name] = {
            "joint_ids": joint_ids,
            "dof_indices": dof_indices,
            "qpos_indices": qpos_indices,
            "ee_body_id": ee_bid,
            "n_dof": len(dof_indices),
        }

    def get_end_effector_pos(self, chain_name: str) -> np.ndarray:
        cache = self._chain_cache[chain_name]
        return self.data.xpos[cache["ee_body_id"]].copy()

    def select_arm(self, target_pos: np.ndarray) -> str:
        """Pick whichever arm's current EE is closer to the target."""
        right_pos = self.get_end_effector_pos("right_arm")
        left_pos = self.get_end_effector_pos("left_arm")
        right_dist = float(np.linalg.norm(target_pos - right_pos))
        left_dist = float(np.linalg.norm(target_pos - left_pos))
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
        """Solve for arm joint angles placing the EE at ``target_pos``."""
        cache = self._chain_cache[chain_name]
        dof_indices = cache["dof_indices"]
        qpos_indices = cache["qpos_indices"]
        joint_ids = cache["joint_ids"]
        ee_bid = cache["ee_body_id"]

        self._ik_data.qpos[:] = self.data.qpos[:]
        self._ik_data.qvel[:] = self.data.qvel[:]

        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, self._ik_data)
            current_pos = self._ik_data.xpos[ee_bid]
            error = target_pos - current_pos
            if float(np.linalg.norm(error)) < tolerance:
                break

            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(
                self.model, self._ik_data, jacp, None, current_pos, ee_bid
            )
            J = jacp[:, dof_indices]
            J_T = J.T
            J_pinv = J_T @ np.linalg.inv(J @ J_T + damping * np.eye(3))
            dq = J_pinv @ error

            for i, qpos_idx in enumerate(qpos_indices):
                self._ik_data.qpos[qpos_idx] += step_size * dq[i]

            for i, jid in enumerate(joint_ids):
                qpos_idx = qpos_indices[i]
                lo = self.model.jnt_range[jid, 0]
                hi = self.model.jnt_range[jid, 1]
                if lo < hi:
                    self._ik_data.qpos[qpos_idx] = float(np.clip(
                        self._ik_data.qpos[qpos_idx], lo, hi
                    ))

        return np.array([self._ik_data.qpos[idx] for idx in qpos_indices])

    def solve_with_selection(
        self, target_pos: np.ndarray, **kwargs
    ) -> Tuple[str, np.ndarray]:
        chain_name = self.select_arm(target_pos)
        return chain_name, self.solve(chain_name, target_pos, **kwargs)
