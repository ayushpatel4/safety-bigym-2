"""Line-of-sight occlusion checks for Phase 1 BodySLAM++.

BodySLAM++ runs off the robot's head camera. A joint hidden behind a fixture
or the robot's own arm is either dropped or estimated with inflated error.
check_joint_visibility models the geometric half: ray from the camera to the
target joint, blocked by any visible geom that isn't the target itself or an
explicitly-excluded robot link.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import mujoco

_MAX_PASSES = 16  # max number of excluded hits to walk past before giving up


def check_joint_visibility(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    targets: np.ndarray,
    exclude_geoms: Optional[Iterable[int]] = None,
    tol: float = 0.05,
) -> np.ndarray:
    """Return a bool array saying whether each target is visible from the named
    camera.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (forward kinematics must be current).
        camera_name: name of an <camera> element.
        targets: (N, 3) world-frame positions.
        exclude_geoms: geom IDs whose hits should be ignored (e.g., robot's
            own links between the camera and the target).
        tol: hit-distance tolerance — a target is visible when the ray reaches
            within `tol` metres of its depth. Absorbs contact between the ray
            tip and the target's own collision geom without false "occluded".

    Returns:
        Bool ndarray of shape (N,). An empty-targets input returns shape (0,).
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.size == 0:
        return np.zeros((0,), dtype=bool)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"targets must have shape (N, 3), got {targets.shape}")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"camera {camera_name!r} not found in model")

    cam_pos = np.array(data.cam_xpos[cam_id], dtype=np.float64)
    exclude_set = set(int(g) for g in (exclude_geoms or []))

    visible = np.zeros(len(targets), dtype=bool)
    geomid_out = np.zeros(1, dtype=np.int32)

    for i, target in enumerate(targets):
        delta = target - cam_pos
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            visible[i] = True
            continue
        direction = delta / distance

        origin = cam_pos.copy()
        remaining = distance
        reached = False
        for _ in range(_MAX_PASSES):
            hit_dist = mujoco.mj_ray(
                model,
                data,
                origin,
                direction,
                None,           # geomgroup: default (all groups)
                1,              # flg_static: include static geoms
                -1,             # bodyexclude: handled via exclude_geoms instead
                geomid_out,
            )
            hit_geom = int(geomid_out[0])
            if hit_dist < 0 or hit_geom < 0 or hit_dist >= remaining - tol:
                reached = True
                break
            if hit_geom in exclude_set:
                # Step past the excluded hit and keep going.
                step = hit_dist + 1e-4
                origin = origin + direction * step
                remaining -= step
                continue
            # Genuine occluder.
            reached = False
            break
        visible[i] = reached

    return visible
