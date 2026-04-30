"""AMASS-driven position provider for BodySLAMWrapper demo replay.

When BC pretrains a policy on raw BiGym demos there is no live SMPL human in
the scene, so ``info["safety"]["human_pos"]`` is absent. The plan's
"demo-replay" mode resolves this by carrying a synthetic human trajectory
(an AMASS clip + a random root transform) for the wrapper to read against.
This keeps the ``human_pos_estimate`` channel non-degenerate during pretrain
so it's not a constant the policy learns to ignore.

Resolves the same parameter axes the live ``ScenarioSampler`` does (clip and
spawn pose) so the live and demo statistics overlap. We intentionally do not
reproduce trajectory/disruption logic — there's no robot to react to during
demo replay; an AMASS playback in the world frame is enough.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from safety_bigym.motion.amass_loader import AMASSLoader, MotionClip


class AMASSDemoPositionProvider:
    """Callable returning a per-step (3,) pelvis position for demo replay.

    Holds one AMASS clip at a time. ``reset()`` re-samples the clip and a
    random root transform; calling the instance with ``step_idx`` returns
    the world-frame pelvis position at that frame (modulo clip length).
    """

    def __init__(
        self,
        clip_paths: Sequence[str],
        motion_dir: str,
        seed: int = 0,
        spawn_distance_range: tuple[float, float] = (1.5, 3.0),
    ):
        if not clip_paths:
            raise ValueError("AMASSDemoPositionProvider needs at least one clip path")
        if not motion_dir:
            raise ValueError("AMASSDemoPositionProvider needs motion_dir to be set")
        self._clip_paths = list(clip_paths)
        self._motion_dir = motion_dir
        self._rng = np.random.default_rng(seed)
        self._loader = AMASSLoader()
        self._dist_lo, self._dist_hi = spawn_distance_range
        self._clip: Optional[MotionClip] = None
        self._root_pos = np.zeros(3, dtype=np.float32)
        self._root_yaw = 0.0
        # Lazy-load the first clip so construction is cheap.

    def reset(self) -> None:
        rel = self._clip_paths[int(self._rng.integers(len(self._clip_paths)))]
        full = os.path.join(self._motion_dir, rel)
        self._clip = self._loader.load(full)
        dist = float(self._rng.uniform(self._dist_lo, self._dist_hi))
        angle = float(self._rng.uniform(-np.pi, np.pi))
        self._root_pos = np.array(
            [dist * np.cos(angle), dist * np.sin(angle), 0.0], dtype=np.float32
        )
        self._root_yaw = float(self._rng.uniform(-np.pi, np.pi))

    def __call__(self, step_idx: int) -> np.ndarray:
        if self._clip is None:
            self.reset()
        assert self._clip is not None
        idx = int(step_idx) % self._clip.num_frames
        local = self._clip.root_translation[idx].astype(np.float32)
        cy, sy = np.cos(self._root_yaw), np.sin(self._root_yaw)
        rotated = np.array(
            [
                cy * local[0] - sy * local[1],
                sy * local[0] + cy * local[1],
                local[2],
            ],
            dtype=np.float32,
        )
        return self._root_pos + rotated
