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

Demos (CQN-AS is demo-driven RL): get_demos() loads 4-dof BiGym demos
through SafetyBiGymEnvFactory's raw-env + DemoStore path, injects
``human_pos_estimate`` via BodySLAMWrapper demo_replay, and converts them
to ExtendedTimeStep lists (Workstream D). Ported from
CQN-AS/bigym_src/bigym_env.py:BiGym.{get_demos,convert_demo_to_timesteps,
extract_action_stats,rescale_demo_actions}.
"""

from __future__ import annotations

import logging
import os
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
from safety_bigym.filters.cost_signal import (
    COST_FORMS,
    D_BUFFER_DEFAULT,
    PFL_RATIO_THRESHOLD_DEFAULT,
    select_cost,
)
from safety_bigym.perception.bodyslam_wrapper import (
    OBS_KEY as BODYSLAM_OBS_KEY,
    BodySLAMWrapper,
)
from safety_bigym.perception.demo_position_provider import AMASSDemoPositionProvider

logger = logging.getLogger(__name__)


_DEFAULT_STATE_KEYS: Tuple[str, ...] = (
    "proprioception",
    "proprioception_grippers",
    "proprioception_floating_base",
)


class TimeStep(NamedTuple):
    """Mirror of CQN-AS/bigym_src/bigym_env.py:TimeStep.

    ``cost`` is a Phase 3 addition (P3.0c): the per-env-step continuous cost
    ``c_t = max(c_ssm, c_pfl)`` computed by :func:`compute_cost` from
    ``info["safety"]``. Stored on the TimeStep so the replay buffer can pick
    it up per-env-step (not per K-action-chunk), preserving the per-step
    granularity required by ``UPDATED_PROJECT_PLAN.md:348``. Default 0.0 keeps
    construction-site compatibility for tests that don't simulate a human.
    """

    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    demo: Any
    info: Dict[str, Any] = {}  # extra field; ignored by agent, used by logger
    cost: float = 0.0  # Phase 3 per-step cost; see filters/cost_signal.py

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
    """Mirror of CQN-AS/bigym_src/bigym_env.py:ExtendedTimeStep.

    Carries the Phase 3 ``cost`` field through to the replay buffer. See
    :class:`TimeStep` for semantics.
    """

    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    action: Any
    demo: Any
    info: Dict[str, Any] = {}
    cost: float = 0.0  # Phase 3 per-step cost; see filters/cost_signal.py

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
            cost=ts.cost,
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


class _DemoReplayEnv:
    """Minimal in-memory gym-like env that replays a demo's observations.

    Used only as the inner env for :class:`BodySLAMWrapper` during demo
    ``human_pos_estimate`` injection (get_demos). It exposes the gym
    surface BodySLAMWrapper touches — a ``spaces.Dict`` ``observation_space``,
    ``reset() -> (obs, info)`` and ``step(action) -> (obs, 0.0, False, False,
    info)`` — and walks the demo's recorded observations one per step.

    The emitted ``info`` carries no ``"safety"`` key, so BodySLAMWrapper's
    ``_get_true_pos`` falls back to the AMASS ``position_provider`` (the
    demo_replay path), exactly as the factory's ``_maybe_wrap_demo_bodyslam``
    does for the RoboBase BC demo env.
    """

    def __init__(self, observation_space: spaces.Dict):
        self.observation_space = observation_space
        self.unwrapped = self
        self._current_scenario = None
        self._observations: list = []
        self._idx = 0

    def load(self, observations: list) -> None:
        """Point the env at a new demo's observation sequence."""
        self._observations = observations
        self._idx = 0

    def reset(self, **kwargs):
        self._idx = 0
        return self._observations[0], {}

    def step(self, action):
        self._idx += 1
        idx = min(self._idx, len(self._observations) - 1)
        return self._observations[idx], 0.0, False, False, {}

    def close(self) -> None:
        pass


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

        # Phase 3 per-step cost configuration. Surface d_buffer / pfl_threshold
        # so an experiment can sweep them without code edits; defaults match
        # UPDATED_PROJECT_PLAN.md:343-346 and filters/cost_signal.py constants.
        safety_cfg = cfg.env.get("safety") if hasattr(cfg, "env") else None
        if safety_cfg is not None:
            self._cost_d_buffer = float(safety_cfg.get("d_buffer", D_BUFFER_DEFAULT))
            self._cost_pfl_threshold = float(
                safety_cfg.get("pfl_ratio_threshold", PFL_RATIO_THRESHOLD_DEFAULT)
            )
            # E3.1 cost-signal form selector (continuous | binary). `fixed` is
            # not a cost form — it disables the Lagrangian and uses the env
            # reward penalty, so it never sets this.
            self._cost_form = str(safety_cfg.get("cost_form", "continuous"))
        else:
            self._cost_d_buffer = D_BUFFER_DEFAULT
            self._cost_pfl_threshold = PFL_RATIO_THRESHOLD_DEFAULT
            self._cost_form = "continuous"
        if self._cost_form not in COST_FORMS:
            raise ValueError(
                f"env.safety.cost_form must be one of {COST_FORMS}; "
                f"got {self._cost_form!r}"
            )

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

        # Phase 3 per-step cost. info["safety"] is populated by ISO15066Wrapper
        # on every env-step; select_cost handles missing/None fields gracefully
        # and dispatches on the E3.1 cost form (continuous | binary).
        cost = select_cost(
            self._last_info.get("safety", {}),
            cost_form=self._cost_form,
            d_buffer=self._cost_d_buffer,
            pfl_threshold=self._cost_pfl_threshold,
        )

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=float(reward),
            discount=discount,
            demo=0.0,
            info=self._last_info,
            cost=cost,
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
            cost=0.0,
        )

    def render(self) -> Union[None, np.ndarray]:
        return self._env.render()

    def get_demos(self, num_demos: int):
        """Load + convert BiGym demos into CQN-AS ExtendedTimeStep lists.

        Port of CQN-AS/bigym_src/bigym_env.py:BiGym.get_demos adapted to
        the SafetyBiGym pipeline (Workstream D). Steps:

        1. Load raw 4-dof demos via SafetyBiGymEnvFactory's raw-env +
           DemoStore path (``_get_demo_fn``); the raw env is required
           because DemoStore matches by env *class name*.
        2. Truncate each demo at the first rewarding state (so the demo's
           final reward ≈ 1.0 and the demo replay buffer keeps it).
        3. When ``bodyslam != off``, inject a per-step ``human_pos_estimate``
           into each demo observation using the same
           ``AMASSDemoPositionProvider`` + ``BodySLAMWrapper(demo_replay=True)``
           mechanism the factory's ``_maybe_wrap_demo_bodyslam`` uses, so the
           demo low_dim width/statistics match the live env.
        4. Convert each demo to a list of :class:`ExtendedTimeStep` (obs via
           ``_extract_obs``, action from ``info["demo_action"]``, step types,
           ``cost=0.0`` since demos carry no live human).
        5. Override ``self._action_stats`` with demo-derived per-dim stats so
           demo and live actions share one normalisation, then rescale the
           demo actions into [-1, 1] under those stats.

        Returns a ``list`` of demos, each a ``list[ExtendedTimeStep]``.
        ``train_cqn_as.load_demos`` adds each timestep to the (demo) replay
        buffers via ``ReplayBufferStorage.add``.
        """
        if num_demos == 0:
            raise ValueError("get_demos called with num_demos=0")

        factory = SafetyBiGymEnvFactory()
        raw_demos = factory._get_demo_fn(self._cfg, num_demos)

        # Truncate each demo at the first rewarding state (upstream parity).
        filtered = []
        for raw_demo in raw_demos:
            steps = []
            for demostep in raw_demo.timesteps:
                steps.append(demostep)
                if demostep.reward > 0:
                    break
            filtered.append(steps)

        # Inject human_pos_estimate when the channel is on.
        if self._inject_human_pos:
            self._inject_human_pos_into_demos(filtered)

        demos = []
        num_successful = 0.0
        for steps in filtered:
            demo, successful = self._convert_demo_to_timesteps(steps)
            num_successful += float(successful)
            demos.append(demo)
        logger.info(
            "Converted %d demos (%d successful) for CQN-AS.",
            len(demos), int(num_successful),
        )

        # Demo-derived action stats shared with the live path, then rescale.
        self._action_stats = self._extract_action_stats(demos)
        demos = [self._rescale_demo_actions(demo) for demo in demos]
        return demos

    # ------------------------------------------------------------------
    # Demo conversion internals (ported from CQN-AS/bigym_src/bigym_env.py)
    # ------------------------------------------------------------------

    def _inject_human_pos_into_demos(self, demos: list) -> None:
        """Mutate each demostep.observation to add ``human_pos_estimate``.

        Drives a :class:`BodySLAMWrapper` in demo_replay mode (fed by an
        ``AMASSDemoPositionProvider``) over an in-memory replay env, mirroring
        the factory's ``_maybe_wrap_demo_bodyslam``. A fresh provider clip +
        root transform is sampled per demo (the wrapper's reset() calls
        ``provider.reset()``).
        """
        bs = self._cfg.env.get("bodyslam") if hasattr(self._cfg, "env") else None
        motion_dir = self._cfg.env.get(
            "motion_clip_dir", os.environ.get("AMASS_DATA_DIR")
        )
        clip_paths = list(self._cfg.env.get("motion_clip_paths", []))
        if not motion_dir or not clip_paths:
            raise RuntimeError(
                "Demo human_pos injection requires motion_clip_dir + "
                "motion_clip_paths in env config (or AMASS_DATA_DIR env var)."
            )
        provider = AMASSDemoPositionProvider(
            clip_paths=clip_paths,
            motion_dir=motion_dir,
            seed=int(bs.get("seed", 0)) ^ 0xDEAD,
        )

        # Build a Dict observation_space mirroring the demo obs keys so
        # BodySLAMWrapper accepts the inner env.
        sample_obs = demos[0][0].observation
        inner_spaces = {
            key: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=np.asarray(val).shape, dtype=np.float32,
            )
            for key, val in sample_obs.items()
        }
        inner = _DemoReplayEnv(spaces.Dict(inner_spaces))
        wrapper = BodySLAMWrapper(
            inner,
            mode=self._bodyslam_mode,
            ou_alpha=float(bs.get("ou_alpha", 0.9)),
            noise_std=float(bs.get("noise_std", 0.05)),
            latency_steps=int(bs.get("latency_steps", 3)),
            occlusion_noise_mult=float(bs.get("occlusion_noise_mult", 3.0)),
            dropout_prob=float(bs.get("dropout_prob", 0.02)),
            seed=int(bs.get("seed", 0)),
            position_provider=provider,
            demo_replay=True,
        )

        for steps in demos:
            observations = [ds.observation for ds in steps]
            inner.load(observations)
            obs, _ = wrapper.reset()
            steps[0].observation = obs
            for i in range(1, len(steps)):
                obs, _, _, _, _ = wrapper.step(None)
                steps[i].observation = obs

    def _convert_demo_to_timesteps(self, demo: list):
        """Turn a list of raw demosteps into a list of ExtendedTimeStep.

        Port of CQN-AS BiGym.convert_demo_to_timesteps; adds the Phase 3
        ``cost`` field (0.0 — demos have no live human, safe-side placeholder)
        and an empty ``info`` dict.
        """
        timesteps: list = []

        # Reset the frame-stack deques (mirrors upstream).
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        rewards = [ds.reward for ds in demo]
        successful_demo = sum(rewards) > 0.25

        last_timestep = False
        for i, demostep in enumerate(demo):
            obs = self._extract_obs(demostep.observation)
            reward = float(demostep.reward)
            discount = 1.0
            term, trunc = demostep.termination, demostep.truncation
            action = np.asarray(
                demostep.info["demo_action"], dtype=np.float32
            )

            if i == 0:
                step_type = StepType.FIRST
            else:
                if (i == len(demo) - 1) or reward > 0:
                    if not (term or trunc):
                        trunc = True  # timelimit
                    step_type = StepType.LAST
                    if term:
                        discount = 0.0
                    last_timestep = True
                else:
                    step_type = StepType.MID

            timesteps.append(
                ExtendedTimeStep(
                    rgb_obs=obs["rgb_obs"],
                    low_dim_obs=obs["low_dim_obs"],
                    step_type=step_type,
                    action=action,
                    reward=reward,
                    discount=discount,
                    demo=int(successful_demo),
                    info={},
                    cost=0.0,
                )
            )
            if last_timestep:
                break

        return timesteps, successful_demo

    def _extract_action_stats(self, demos: list) -> Dict[str, np.ndarray]:
        """Per-dim action min/max over all demos; gripper tail forced [0,1].

        Port of CQN-AS BiGym.extract_action_stats. Overrides the identity
        default so demo + live actions share one normalisation.
        """
        actions = []
        for demo in demos:
            for ts in demo:
                actions.append(ts.action)
        actions = np.stack(actions)

        action_max = np.hstack([np.max(actions, 0)[:-2], 1, 1]).astype(np.float32)
        action_min = np.hstack([np.min(actions, 0)[:-2], 0, 0]).astype(np.float32)

        env_low = self._env.action_space.low
        env_high = self._env.action_space.high
        if not (np.all(action_min >= env_low) and np.all(action_max <= env_high)):
            logger.warning(
                "Demo action stats exceed env action bounds; clipping to range."
            )
            action_min = np.maximum(action_min, env_low).astype(np.float32)
            action_max = np.minimum(action_max, env_high).astype(np.float32)

        return {"min": action_min, "max": action_max}

    def _rescale_demo_actions(self, demo: list) -> list:
        """Rescale each demo action raw→[-1,1] under the current action_stats."""
        return [
            ts._replace(action=self._convert_action_from_raw(ts.action))
            for ts in demo
        ]

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
