"""SafetyBiGym → CQN-AS env adapter.

CQN-AS's training loop and replay buffer expect a TimeStep-based env with
the (step_type, reward, discount, rgb_obs, low_dim_obs, demo) NamedTuple
API from CQN-AS/bigym_src/bigym_env.py. SafetyBiGymEnv (via
SafetyBiGymEnvFactory) is a gym.Env with a dict observation.

This adapter sits between them. It calls SafetyBiGymEnvFactory._create_env
so every existing safety wrapper composes — ISO15066Wrapper,
EpisodeSafetyMetrics, BodySLAMWrapper, COWORKER scenario sampler — and
translates each step into the CQN-AS TimeStep form, including:

- concatenating ``human_pos_estimate`` (6D) into low_dim_obs when
  ``cfg.env.bodyslam.mode != "off"``
- carrying per-step ``info["safety"]`` forward so the replay buffer can
  log cost signals at single-step resolution (Phase 3 prep, gate A6.3)
- emitting ``info["episode_safety"]`` aggregate at episode end

This file is NOT vendored from CQN-AS; it's a local glue module. Action
normalisation and frame-stacking logic mirror CQN-AS/bigym_src/bigym_env.py
:BiGym so the agent's action/obs assumptions hold unchanged.

Demos (CQN-AS is demo-driven RL): get_demos() is a stub for now. The
smoke gate (A6) needs env composition only, not demos. Wiring CQN-AS
demos through SafetyBiGymEnvFactory's demo path is follow-up work after
the smoke gate is green.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

try:
    from dm_env import StepType, specs
except ImportError as e:  # pragma: no cover - dm_env is a CQN-AS dependency
    raise ImportError(
        "dm_env is required for the CQN-AS adapter. Install it via the CQN-AS "
        "conda env or `pip install dm_env`."
    ) from e

from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory
from safety_bigym.perception.bodyslam_wrapper import OBS_KEY as BODYSLAM_OBS_KEY

logger = logging.getLogger(__name__)


_DEFAULT_STATE_KEYS: Tuple[str, ...] = (
    "proprioception",
    "proprioception_grippers",
    "proprioception_floating_base",
)


class TimeStep(NamedTuple):
    """Mirror of CQN-AS/bigym_src/bigym_env.py:TimeStep."""

    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    demo: Any
    info: Dict[str, Any] = {}  # extra field; ignored by agent, used by logger

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


class ExtendedTimeStep(NamedTuple):
    """Mirror of CQN-AS/bigym_src/bigym_env.py:ExtendedTimeStep."""

    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    action: Any
    demo: Any
    info: Dict[str, Any] = {}

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


class ExtendedTimeStepWrapper:
    """Mirror of CQN-AS's wrapper. Adds `action` to each TimeStep."""

    def __init__(self, env: "SafetyBiGymCQNAdapter"):
        self._env = env

    def reset(self):
        ts = self._env.reset()
        return self._augment(ts)

    def step(self, action):
        ts = self._env.step(action)
        return self._augment(ts, action)

    def _augment(self, ts: TimeStep, action: Optional[np.ndarray] = None):
        if action is None:
            spec = self.action_spec()
            action = np.zeros(spec.shape, dtype=spec.dtype)
        return ExtendedTimeStep(
            rgb_obs=ts.rgb_obs,
            low_dim_obs=ts.low_dim_obs,
            step_type=ts.step_type,
            action=action,
            reward=ts.reward,
            discount=ts.discount,
            demo=ts.demo,
            info=ts.info,
        )

    def low_dim_observation_spec(self):
        return self._env.low_dim_observation_spec()

    def rgb_observation_spec(self):
        return self._env.rgb_observation_spec()

    def low_dim_raw_observation_spec(self):
        return self._env.low_dim_raw_observation_spec()

    def rgb_raw_observation_spec(self):
        return self._env.rgb_raw_observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class SafetyBiGymCQNAdapter:
    """CQN-AS-compatible adapter around SafetyBiGymEnv.

    Mirrors the public API of CQN-AS/bigym_src/bigym_env.py:BiGym
    (TimeStep step/reset, action normalisation, frame stacking,
    action_spec / *_observation_spec) but builds the env through
    SafetyBiGymEnvFactory so all safety wrappers compose.

    Parameters
    ----------
    cfg
        Hydra DictConfig compatible with safety_bigym_factory._create_env
        (i.e. carries env.task_name, env.cameras, env.disruptions, etc).
    frame_stack
        How many past observations to stack. Default 1 (no stack).
    state_keys
        Low-dim observation keys to concatenate from the obs dict, before
        any human_pos_estimate injection. Defaults match CQN-AS's BiGym.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        frame_stack: int = 1,
        state_keys: Tuple[str, ...] = _DEFAULT_STATE_KEYS,
        normalize_low_dim_obs: bool = False,
    ):
        self._cfg = cfg
        self._frame_stack = frame_stack
        self._state_keys = tuple(state_keys)
        self._normalize_low_dim_obs = normalize_low_dim_obs
        self._low_dim_obs_stats: Optional[Dict[str, np.ndarray]] = None

        bs = cfg.env.get("bodyslam") if hasattr(cfg, "env") else None
        self._bodyslam_mode = str(bs.get("mode", "off")) if bs is not None else "off"
        self._inject_human_pos = self._bodyslam_mode != "off"

        # Gate cameras on the top-level pixels flag — when pixels=False, the
        # factory builds the env with cameras=[] and the obs dict has no
        # rgb_<name> keys; reading them would KeyError.
        pixels_on = bool(cfg.get("pixels", False))
        self._camera_keys: Tuple[str, ...] = (
            tuple(cfg.env.get("cameras", [])) if pixels_on else ()
        )
        cam_shape = cfg.get("visual_observation_shape", [84, 84])
        self._camera_shape: Tuple[int, int] = (int(cam_shape[0]), int(cam_shape[1]))

        self._step_counter = 0
        self._episode_length = int(cfg.env.episode_length)
        self._demo_down_sample_rate = int(cfg.env.demo_down_sample_rate)

        self._last_info: Dict[str, Any] = {}

        self._launch()
        self._initialize_frame_stack()
        self._construct_action_and_observation_spaces()

    # ------------------------------------------------------------------
    # CQN-AS-compatible API
    # ------------------------------------------------------------------

    def low_dim_observation_spec(self) -> specs.Array:
        return specs.Array(
            self.low_dim_observation_space.shape, np.float32, "low_dim_obs"
        )

    def low_dim_raw_observation_spec(self) -> specs.Array:
        return specs.Array(
            self.low_dim_raw_observation_space.shape, np.float32, "low_dim_obs"
        )

    def rgb_observation_spec(self) -> specs.Array:
        return specs.Array(
            self.rgb_observation_space.shape, np.uint8, "rgb_obs"
        )

    def rgb_raw_observation_spec(self) -> specs.Array:
        return specs.Array(
            self.rgb_raw_observation_space.shape, np.uint8, "rgb_obs"
        )

    def action_spec(self) -> specs.Array:
        return specs.Array(self.action_space.shape, np.float32, "action")

    def step(self, action: np.ndarray) -> TimeStep:
        raw_action = self._convert_action_to_raw(action)
        gym_obs, reward, terminated, truncated, info = self._env.step(raw_action)
        obs = self._extract_obs(gym_obs)
        self._step_counter += 1
        self._last_info = dict(info) if info else {}

        # Mirror CQN-AS BiGym's hard timelimit: control-rate step budget.
        if self._step_counter >= (self._episode_length // self._demo_down_sample_rate):
            truncated = True

        step_type = StepType.LAST if (terminated or truncated) else StepType.MID
        discount = float(1 - bool(terminated))

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=float(reward),
            discount=discount,
            demo=0.0,
            info=self._last_info,
        )

    def reset(self, **kwargs) -> TimeStep:
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        gym_obs, info = self._env.reset(**kwargs)
        obs = self._extract_obs(gym_obs)
        self._step_counter = 0
        self._last_info = dict(info) if info else {}

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            demo=0.0,
            info=self._last_info,
        )

    def render(self) -> Union[None, np.ndarray]:
        return self._env.render()

    def get_demos(self, num_demos: int):
        """Stub — demos are a follow-up to the A6 smoke gate.

        SafetyBiGymEnvFactory loads demos via the raw (non-safety-wrapped)
        BiGym env, then re-wraps with BodySLAMWrapper in demo_replay mode
        (synthesising human_pos_estimate from an AMASS clip). Porting that
        path into the CQN-AS demo pipeline requires reformatting the
        loaded demos into CQN-AS's ExtendedTimeStep list (see
        CQN-AS/bigym_src/bigym_env.py:BiGym.convert_demo_to_timesteps).

        Raise here so unconfigured demo training fails loudly; for the
        smoke gate, configure with num_demos=0 / disable demo pretraining.
        """
        raise NotImplementedError(
            "CQN-AS demos through SafetyBiGymEnvFactory are not wired yet "
            "(follow-up to A6 smoke gate). For smoke runs set num_demos=0."
        )

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    # ------------------------------------------------------------------
    # Internal — env construction + obs handling
    # ------------------------------------------------------------------

    def _launch(self) -> None:
        """Build the SafetyBiGymEnv via SafetyBiGymEnvFactory.

        Note: SafetyBiGymEnvFactory expects a RoboBase-style cfg with
        ``cfg.pixels`` and ``cfg.visual_observation_shape`` at the top
        level. The cqn_as_config.yaml root config must include those
        keys; we don't synthesise them here.
        """
        factory = SafetyBiGymEnvFactory()
        self._env = factory._create_env(self._cfg)

    def _initialize_frame_stack(self) -> None:
        self._low_dim_obses = deque([], maxlen=self._frame_stack)
        self._frames = {
            camera_key: deque([], maxlen=self._frame_stack)
            for camera_key in self._camera_keys
        }

    def _construct_action_and_observation_spaces(self) -> None:
        # Action space: [-1, 1] normalised, same shape as underlying env.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_space.shape, dtype=np.float32
        )

        # Low-dim obs width: sum of state-key widths + 6 for human_pos_estimate
        # when bodyslam is on. Probe from the env's observation_space.
        obs_space = self._env.observation_space
        low_dim_width = 0
        for key in self._state_keys:
            if key not in obs_space.spaces:
                raise KeyError(
                    f"state_key {key!r} missing from env observation_space "
                    f"(have: {sorted(obs_space.spaces.keys())})"
                )
            low_dim_width += int(obs_space[key].shape[-1])
        if self._inject_human_pos:
            if BODYSLAM_OBS_KEY not in obs_space.spaces:
                raise KeyError(
                    f"bodyslam.mode={self._bodyslam_mode!r} but observation "
                    f"space lacks {BODYSLAM_OBS_KEY!r}. Did the factory "
                    f"insert BodySLAMWrapper?"
                )
            low_dim_width += int(obs_space[BODYSLAM_OBS_KEY].shape[-1])

        self.low_dim_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(low_dim_width * self._frame_stack,),
            dtype=np.float32,
        )
        self.low_dim_raw_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(low_dim_width,), dtype=np.float32
        )

        self.rgb_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                len(self._camera_keys),
                3 * self._frame_stack,
                *self._camera_shape,
            ),
            dtype=np.uint8,
        )
        self.rgb_raw_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(len(self._camera_keys), 3, *self._camera_shape),
            dtype=np.uint8,
        )

        # Default action stats: identity normalisation (overridden by
        # extract_action_stats once demos are wired).
        action_min = -np.ones(self.action_space.shape, dtype=np.float32)
        action_max = np.ones(self.action_space.shape, dtype=np.float32)
        # Match CQN-AS BiGym: last two dims are grippers, range [0, 1].
        if self.action_space.shape[0] >= 2:
            action_min[-2:] = 0
            action_max[-2:] = 1
        self._action_stats: Dict[str, np.ndarray] = {
            "min": action_min, "max": action_max,
        }

    def _convert_action_to_raw(self, action: np.ndarray) -> np.ndarray:
        """[-1, 1] → underlying env's raw range, via action_stats."""
        action = np.asarray(action, dtype=np.float32)
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        scaled = (action + 1.0) / 2.0  # [0, 1]
        scaled = scaled * (action_max - action_min + 1e-8) + action_min
        return scaled.astype(np.float32, copy=False)

    def _convert_action_from_raw(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        scaled = (action - action_min) / (action_max - action_min + 1e-8)
        scaled = scaled * 2.0 - 1.0
        return scaled.astype(np.float32, copy=False)

    def _extract_obs(self, gym_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Translate gym dict obs → (rgb_obs stack, low_dim_obs stack)."""
        # Low-dim: concatenate state_keys + optional human_pos_estimate
        pieces = [gym_obs[key] for key in self._state_keys]
        if self._inject_human_pos:
            pieces.append(gym_obs[BODYSLAM_OBS_KEY])
        low_dim_obs = np.hstack(pieces).astype(np.float32)
        if self._normalize_low_dim_obs and self._low_dim_obs_stats is not None:
            mean = self._low_dim_obs_stats["mean"]
            std = self._low_dim_obs_stats["std"]
            low_dim_obs = (low_dim_obs - mean) / (std + 1e-8)

        if len(self._low_dim_obses) == 0:
            for _ in range(self._frame_stack):
                self._low_dim_obses.append(low_dim_obs)
        else:
            self._low_dim_obses.append(low_dim_obs)

        out: Dict[str, np.ndarray] = {
            "low_dim_obs": np.concatenate(list(self._low_dim_obses), axis=0),
        }

        # RGB: stack per-camera, concat along channel
        if self._camera_keys:
            for camera_key in self._camera_keys:
                pixels = gym_obs[f"rgb_{camera_key}"].copy().astype(np.uint8)
                if len(self._frames[camera_key]) == 0:
                    for _ in range(self._frame_stack):
                        self._frames[camera_key].append(pixels)
                else:
                    self._frames[camera_key].append(pixels)
            out["rgb_obs"] = np.stack(
                [
                    np.concatenate(list(self._frames[camera_key]), axis=0)
                    for camera_key in self._camera_keys
                ],
                axis=0,
            )
        else:
            # Empty pixel placeholder so downstream code can still index it
            out["rgb_obs"] = np.zeros(self.rgb_observation_space.shape, np.uint8)

        return out


def make(cfg: DictConfig, **kwargs) -> ExtendedTimeStepWrapper:
    """Public factory: returns an ExtendedTimeStepWrapper-wrapped adapter.

    Mirrors the entrypoint contract of CQN-AS/bigym_src/bigym_env.py:make
    so the rest of the CQN-AS training loop is a drop-in.
    """
    return ExtendedTimeStepWrapper(SafetyBiGymCQNAdapter(cfg, **kwargs))


__all__ = [
    "SafetyBiGymCQNAdapter",
    "ExtendedTimeStepWrapper",
    "TimeStep",
    "ExtendedTimeStep",
    "make",
]
