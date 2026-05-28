"""Constants for AMASS-free (procedural) SMPL-H coworker motion.

When ``HumanConfig.smplh_motion == "procedural"``, the human uses the same
control contract as :class:`G1HumanController`:

- Fixed standing body pose (PD-held every step)
- Pelvis XY/yaw from :class:`TrajectoryPlanner` via mocap
- Arm motion only from COWORKER IK during loiter

The asset remains ``smplh_human_body.xml`` — only the controller changes.
"""

from __future__ import annotations

import mujoco
import numpy as np

PELVIS_BODY = "Pelvis"

# Default pelvis height when standing (m). Matches ``SafetyBiGymEnv``'s
# SMPL-H spawn default when no AMASS clip is loaded.
STANDING_PELVIS_Z = 1.0


def build_standing_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return a natural arms-at-sides standing qpos (no T-pose).

    Legs and trunk stay at zero (upright SMPL-H rest). Each arm is IK-solved
    to a point below its shoulder so the coworker RETRACT pose looks human
    rather than a rigid mannequin.
    """
    from safety_bigym.human.human_ik import HumanIK

    qpos = np.zeros(model.nq)
    ik = HumanIK(model, data)
    ik_data = ik._ik_data
    ik_data.qpos[:] = data.qpos[:]
    ik_data.mocap_pos[:] = data.mocap_pos[:]
    ik_data.mocap_quat[:] = data.mocap_quat[:]
    mujoco.mj_forward(model, ik_data)

    for chain in ("right_arm", "left_arm"):
        shoulder_name = ik.chains[chain]["shoulder_body"]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name)
        if sid < 0:
            continue
        shoulder_pos = data.xpos[sid].copy()
        target = shoulder_pos + np.array([0.0, 0.0, -0.7])
        try:
            angles = ik.solve(
                chain, target, max_iterations=80, tolerance=0.05,
                damping=0.05, step_size=0.7,
            )
        except Exception:
            continue
        qpos_indices = ik._chain_cache[chain]["qpos_indices"]
        for idx, ang in zip(qpos_indices, angles):
            qpos[idx] = ang
        for idx, ang in zip(qpos_indices, angles):
            ik_data.qpos[idx] = ang
        mujoco.mj_forward(model, ik_data)

    return qpos
