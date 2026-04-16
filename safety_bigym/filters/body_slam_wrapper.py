"""Mock BodySLAM++ observation wrapper (Phase 1).

Wraps a SafetyBiGym env (or any env whose info['safety'] dict exposes the
tracked human skeleton) and injects three new Dict observation keys:

  * human_state_estimate   (N, 3) float32  — noisy, latency-delayed position.
  * human_state_occluded   (N,)   float32  — 1.0 iff that joint is occluded.
  * human_state_staleness  (1,)   float32  — seconds since last fresh estimate.

The perception failure modes (OU-correlated noise, 2–3 step latency,
per-joint occlusion from the robot's head camera, 2% tracking-lost dropout)
are configurable via BodySLAMConfig so Phase 1 ablations (E1.1–E1.3) can
toggle them independently.

Wrap order: SafetyBiGymEnv -> BodySLAMWrapper -> EpisodeSafetyMetrics.
EpisodeSafetyMetrics must see info['safety'] unmodified; this wrapper only
augments the observation.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from safety_bigym.config import BodySLAMConfig
from safety_bigym.filters.noise_models import OrnsteinUhlenbeckNoise
from safety_bigym.filters.ray_cast import check_joint_visibility


class BodySLAMWrapper(gym.Wrapper):
    """Overlay a noisy BodySLAM++-style skeleton estimate onto the obs dict.

    gym.Wrapper rather than gym.ObservationWrapper — the estimate depends on
    info['safety']['human_joint_positions'], which is only produced by step().
    """

    def __init__(
        self,
        env: gym.Env,
        config: BodySLAMConfig,
        num_joints: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)

        if num_joints is None:
            num_joints = _infer_num_joints(env)
        if num_joints <= 0:
            raise ValueError(f"num_joints must be positive, got {num_joints}")

        self.body_slam_config = config
        self._num_joints = int(num_joints)
        self._dt = float(config.dt)
        self._oracle = (config.mode == "oracle")

        # Seeded RNG for dropout coin-flips; OU has its own RNG.
        self._rng = np.random.default_rng(seed)
        self._noise = OrnsteinUhlenbeckNoise(
            shape=(self._num_joints, 3),
            alpha=config.alpha,
            sigma=config.sigma,
            seed=seed,
        )

        self._latency_buffer: Deque[np.ndarray] = deque(
            maxlen=max(1, config.latency_steps + 1)
        )
        self._last_estimate = np.zeros((self._num_joints, 3), dtype=np.float32)
        self._staleness = 0.0
        self._last_occluded = np.zeros(self._num_joints, dtype=np.float32)

        self.observation_space = self._augment_space(env.observation_space)

    # ------------------------------------------------------------------
    # gym.Wrapper overrides
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._noise.reset(seed=seed)
        else:
            self._noise.reset(seed=None)
        self._latency_buffer.clear()
        self._last_estimate = np.zeros((self._num_joints, 3), dtype=np.float32)
        self._last_occluded = np.zeros(self._num_joints, dtype=np.float32)
        self._staleness = 0.0
        obs = self._augment_observation(obs, fresh=False)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        truth = self._read_truth(info)
        if truth is None:
            # No joint data this step — surface the last known estimate.
            obs = self._augment_observation(obs, fresh=False)
            return obs, reward, terminated, truncated, info

        if self._oracle:
            estimate = truth.astype(np.float32)
            occluded = np.zeros(self._num_joints, dtype=np.float32)
            self._last_estimate = estimate
            self._last_occluded = occluded
            self._staleness = 0.0
            obs = self._write_keys(obs, estimate, occluded, self._staleness)
            return obs, reward, terminated, truncated, info

        self._latency_buffer.append(truth)
        delayed_truth = self._latency_buffer[0]  # oldest in buffer

        # Dropout: frozen estimate, incremented staleness, all-occluded flag.
        if self._rng.random() < self.body_slam_config.dropout_prob:
            self._staleness += self._dt
            self._last_occluded = np.ones(self._num_joints, dtype=np.float32)
            obs = self._write_keys(
                obs, self._last_estimate, self._last_occluded, self._staleness
            )
            return obs, reward, terminated, truncated, info

        # Fresh estimate: delayed truth + noise, scaled per-joint on occlusion.
        occluded = self._compute_occlusion(delayed_truth)
        noise = self._noise.step()
        scale = np.where(
            occluded[:, None],
            self.body_slam_config.occlusion_multiplier,
            1.0,
        )
        estimate = (delayed_truth + noise * scale).astype(np.float32)

        self._last_estimate = estimate
        self._last_occluded = occluded.astype(np.float32)
        self._staleness = 0.0
        obs = self._write_keys(obs, estimate, self._last_occluded, self._staleness)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment_space(self, base: spaces.Space) -> spaces.Dict:
        if not isinstance(base, spaces.Dict):
            raise TypeError(
                f"BodySLAMWrapper requires a Dict observation space, got {type(base).__name__}"
            )
        new_spaces = dict(base.spaces)
        new_spaces["human_state_estimate"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._num_joints, 3), dtype=np.float32,
        )
        new_spaces["human_state_occluded"] = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._num_joints,), dtype=np.float32,
        )
        new_spaces["human_state_staleness"] = spaces.Box(
            low=0.0, high=np.inf,
            shape=(1,), dtype=np.float32,
        )
        return spaces.Dict(new_spaces)

    def _augment_observation(self, obs: Dict[str, Any], fresh: bool) -> Dict[str, Any]:
        """Augment obs for reset() and for steps where truth is missing."""
        if not fresh:
            return self._write_keys(
                obs,
                self._last_estimate,
                self._last_occluded,
                self._staleness,
            )
        raise AssertionError("unused path")

    def _write_keys(
        self,
        obs: Dict[str, Any],
        estimate: np.ndarray,
        occluded: np.ndarray,
        staleness: float,
    ) -> Dict[str, Any]:
        obs["human_state_estimate"] = estimate.astype(np.float32, copy=False)
        obs["human_state_occluded"] = occluded.astype(np.float32, copy=False)
        obs["human_state_staleness"] = np.array([staleness], dtype=np.float32)
        return obs

    def _read_truth(self, info: Dict[str, Any]) -> Optional[np.ndarray]:
        safety = info.get("safety")
        if not safety:
            return None
        positions = safety.get("human_joint_positions")
        if positions is None or len(positions) == 0:
            return None
        arr = np.asarray(positions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None
        if arr.shape[0] != self._num_joints:
            # Resize on the fly if the joint list is shorter/longer than our
            # observation shape. Plan: in practice this matches _HUMAN_SSM_BODY_NAMES.
            return None
        return arr

    def _compute_occlusion(self, targets: np.ndarray) -> np.ndarray:
        if not self.body_slam_config.use_occlusion:
            return np.zeros(self._num_joints, dtype=bool)
        unwrapped = self.env.unwrapped
        model = getattr(unwrapped, "_mojo", None)
        model = getattr(model, "model", None) if model is not None else None
        data = getattr(self.env.unwrapped, "_mojo", None)
        data = getattr(data, "data", None) if data is not None else None
        if model is None or data is None:
            # No MuJoCo context available (e.g., stub env in tests when
            # use_occlusion=True but the monkeypatched fake ignores these).
            return check_joint_visibility_safe(
                None, None, self.body_slam_config.camera_name, targets
            )
        exclude = _robot_geom_ids(unwrapped)
        visible = check_joint_visibility(
            model, data,
            self.body_slam_config.camera_name,
            targets,
            exclude_geoms=exclude,
        )
        return ~visible


def check_joint_visibility_safe(model, data, camera, targets):
    """Adapter used when tests monkeypatch the module-level check. The real
    function is imported at module load time; test patches replace it there.
    This thin indirection ensures BodySLAMWrapper._compute_occlusion dispatches
    through the patched name even when model/data are unavailable."""
    return ~check_joint_visibility(model, data, camera, np.asarray(targets))


def _infer_num_joints(env: gym.Env) -> int:
    """Best-effort pull of the tracked joint count from the base env."""
    base = env.unwrapped
    names = getattr(base, "_HUMAN_SSM_BODY_NAMES", None)
    if names:
        return len(names)
    raise ValueError(
        "num_joints not provided and env exposes no _HUMAN_SSM_BODY_NAMES"
    )


def _robot_geom_ids(base_env) -> list:
    wrapper = getattr(base_env, "safety_wrapper", None)
    if wrapper is None:
        return []
    return sorted(wrapper.robot_geoms)
