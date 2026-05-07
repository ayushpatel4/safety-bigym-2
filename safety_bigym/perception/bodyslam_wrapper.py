"""BodySLAMWrapper — Phase 1 mock perception layer.

Adds a noisy human-pose estimate to the env's observation dict under the
key ``human_pos_estimate``. Shape (6,):

    [x, y, z, occluded, staleness, confidence]

The clean ``μ_t`` is read from ``info["safety"]["human_pos"]`` (populated
by ``ISO15066Wrapper.build_safety_info``). Three noise sources are
composed when ``mode="noisy"``:

1. Ornstein-Uhlenbeck process on position (temporal correlation).
2. Latency buffer (simulates perception → control delay).
3. Per-step dropout (simulates tracking-lost events).

Occlusion is injected via a pluggable callable (default: no occlusion).
The reference implementation for live MuJoCo envs is :class:`MujocoRayOcclusion`.

In ``mode="oracle"`` we emit the clean ``μ_t`` with all flags zero —
used as the upper-bound condition in the E1.1 ablation.

Demo replay (``demo_replay=True``): when ``info["safety"]`` is absent
(typical during BC pretraining on BiGym demos), the wrapper falls back
to a caller-supplied ``position_provider(step_idx) -> (3,)`` so the
observation channel still carries realistic human-state statistics.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


OBS_KEY = "human_pos_estimate"
OBS_DIM = 6
_STALENESS_NORM = 10.0


def _no_occlusion(env, info) -> bool:
    return False


NoOcclusion = _no_occlusion


class MujocoRayOcclusion:
    """Ray-cast occlusion checker for live MuJoCo envs.

    Casts a ray from a named camera (the H1 head camera, by default
    ``"head"``) toward a target geom (the human pelvis collision geom).
    If the ray hits any other geom first, the target is occluded.
    """

    def __init__(
        self,
        model,
        data,
        camera_id: int,
        target_geom_id: int,
    ):
        import mujoco  # local import — keep wrapper importable without mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self.target_geom_id = target_geom_id

    def __call__(self, env, info) -> bool:
        return self.is_occluded()

    def is_occluded(self) -> bool:
        mj = self._mujoco
        cam_pos = self.data.cam_xpos[self.camera_id].copy()
        target_pos = self.data.geom_xpos[self.target_geom_id].copy()
        direction = target_pos - cam_pos
        dist = float(np.linalg.norm(direction))
        if dist < 1e-9:
            return False
        direction /= dist

        geomid_out = np.zeros(1, dtype=np.int32)
        # mj_ray returns the distance to first hit (or -1 if none).
        # We want any geom hit before the target — so pass geomgroup=None
        # to test against all groups, and exclude nothing.
        hit_dist = mj.mj_ray(
            self.model,
            self.data,
            cam_pos,
            direction,
            None,         # geomgroup
            1,            # flg_static (include static geoms)
            -1,           # bodyexclude (none)
            geomid_out,
        )
        first_hit_id = int(geomid_out[0])
        if first_hit_id < 0 or hit_dist < 0:
            # No hit at all — line of sight unblocked (unlikely but safe)
            return False
        if first_hit_id == self.target_geom_id:
            return False
        # Hit something else first; check it's actually before the target.
        return hit_dist < dist - 1e-4


class BodySLAMWrapper(gym.Wrapper):
    """Add a noisy ``human_pos_estimate`` observation key to the env."""

    def __init__(
        self,
        env: gym.Env,
        mode: str = "noisy",
        ou_alpha: float = 0.9,
        noise_std: float = 0.05,
        latency_steps: int = 3,
        occlusion_noise_mult: float = 3.0,
        dropout_prob: float = 0.02,
        seed: int = 0,
        occlusion_fn: Optional[Callable] = None,
        position_provider: Optional[Callable[[int], np.ndarray]] = None,
        demo_replay: bool = False,
    ):
        super().__init__(env)
        if mode not in ("oracle", "noisy"):
            raise ValueError(
                f"BodySLAMWrapper.mode must be 'oracle' or 'noisy', got {mode!r}. "
                f"Use the factory's bodyslam.mode='off' to skip the wrapper entirely."
            )
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                f"BodySLAMWrapper requires a Dict observation_space, got "
                f"{type(env.observation_space).__name__}"
            )

        self._mode = mode
        self._ou_alpha = float(ou_alpha)
        self._noise_std = float(noise_std)
        self._lag = int(latency_steps)
        self._occlusion_mult = float(occlusion_noise_mult)
        self._dropout_prob = float(dropout_prob)
        self._seed = int(seed)
        self._occlusion_fn = occlusion_fn or _no_occlusion
        self._position_provider = position_provider
        self._demo_replay = bool(demo_replay)

        # Extend observation space with the new key.
        new_spaces = dict(env.observation_space.spaces)
        new_spaces[OBS_KEY] = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(new_spaces)

        self._rng: np.random.Generator
        self._ou_state: np.ndarray
        self._buf: deque
        self._last_emitted: np.ndarray
        self._staleness: int
        self._step_idx: int
        self.last_estimate: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)

        # Optional override hook for tests: deque[bool], True = drop.
        self._force_dropout_schedule: Optional[deque] = None

    # ---- helpers ----------------------------------------------------------

    def _derive_seed(self) -> int:
        scen = getattr(self.env.unwrapped, "_current_scenario", None)
        if scen is not None and hasattr(scen, "seed"):
            try:
                return int(scen.seed) ^ self._seed
            except Exception:
                pass
        return self._seed

    def _get_true_pos(self, info: dict, step_idx: int) -> np.ndarray:
        safety = info.get("safety") if isinstance(info, dict) else None
        if safety is not None and "human_pos" in safety:
            return np.asarray(safety["human_pos"], dtype=np.float32).reshape(3)
        if self._demo_replay and self._position_provider is not None:
            return np.asarray(
                self._position_provider(step_idx), dtype=np.float32
            ).reshape(3)
        raise RuntimeError(
            "BodySLAMWrapper: info['safety']['human_pos'] is missing and no "
            "position_provider was supplied (demo_replay=%s)." % self._demo_replay
        )

    def _draw_dropout(self) -> bool:
        """Return True if this step should drop (emit stale)."""
        if self._dropout_prob <= 0.0 and self._force_dropout_schedule is None:
            return False
        if self._force_dropout_schedule is not None and self._force_dropout_schedule:
            return bool(self._force_dropout_schedule.popleft())
        u = self._rng.uniform()
        return bool(u < self._dropout_prob)

    def _confidence(self, occluded: bool, staleness: int) -> float:
        occ = 1.0 if occluded else 0.0
        return float((1.0 - 0.5 * occ) * max(0.0, 1.0 - staleness / _STALENESS_NORM))

    def _build_estimate(
        self, pos: np.ndarray, occluded: bool, staleness: int
    ) -> np.ndarray:
        occ_f = 1.0 if occluded else 0.0
        conf = self._confidence(occluded, staleness)
        vec = np.array(
            [pos[0], pos[1], pos[2], occ_f, float(staleness), conf],
            dtype=np.float32,
        )
        return vec

    def _attach(self, obs: dict, vec: np.ndarray) -> dict:
        obs = dict(obs)  # don't mutate caller's dict
        obs[OBS_KEY] = vec
        self.last_estimate = vec.copy()
        return obs

    # ---- gym API ----------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._rng = np.random.default_rng(self._derive_seed())
        self._step_idx = 0

        # Position provider may want to re-sample its underlying clip on
        # episode boundary — duck-typed so simple lambdas still work.
        if self._demo_replay and self._position_provider is not None:
            reset_hook = getattr(self._position_provider, "reset", None)
            if callable(reset_hook):
                reset_hook()

        initial = self._get_true_pos(info, step_idx=0)
        self._ou_state = initial.copy()
        # Buffer of length lag+1 so emit = OU lagged by self._lag steps.
        self._buf = deque(
            [initial.copy() for _ in range(self._lag + 1)], maxlen=self._lag + 1
        )
        self._last_emitted = initial.copy()
        self._staleness = 0

        # Initial emit is always clean (fresh, not occluded).
        vec = self._build_estimate(initial, occluded=False, staleness=0)
        return self._attach(obs, vec), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_idx += 1
        mu = self._get_true_pos(info, step_idx=self._step_idx)

        if self._mode == "oracle":
            self._last_emitted = mu.copy()
            self._staleness = 0
            vec = self._build_estimate(mu, occluded=False, staleness=0)
            return self._attach(obs, vec), reward, terminated, truncated, info

        # Noisy mode: occlusion this step
        occluded = bool(self._occlusion_fn(self.env, info))
        sigma_eff = self._noise_std * (
            self._occlusion_mult if occluded else 1.0
        )

        # OU update — always advances, even during dropout, so recovery
        # doesn't produce a discontinuity.
        eps = self._rng.standard_normal(3)
        self._ou_state = (
            self._ou_alpha * self._ou_state
            + (1.0 - self._ou_alpha) * mu
            + sigma_eff * eps
        )
        self._buf.append(self._ou_state.copy())

        if self._draw_dropout():
            emit = self._last_emitted
            self._staleness += 1
        else:
            emit = np.asarray(self._buf[0], dtype=np.float32).copy()
            self._last_emitted = emit
            self._staleness = 0

        vec = self._build_estimate(emit, occluded=occluded, staleness=self._staleness)
        return self._attach(obs, vec), reward, terminated, truncated, info
