"""
SafetyBiGymEnvFactory — RoboBase EnvFactory for SafetyBiGym

Subclasses robobase's BiGymEnvFactory to inject the SMPL-H human
and ISO 15066 safety monitoring layer around any BiGym task.

Usage:
    from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory
    workspace = Workspace(cfg, env_factory=SafetyBiGymEnvFactory())
"""

import copy
import logging
import os

import numpy as np

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from omegaconf import DictConfig

from robobase.envs.bigym import BiGymEnvFactory
from robobase.envs.utils.bigym_utils import TASK_MAP

from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

from safety_bigym import make_safety_env, SafetyConfig, HumanConfig
from safety_bigym.perception import (
    AMASSDemoPositionProvider,
    BodySLAMWrapper,
    MujocoRayOcclusion,
    NoOcclusion,
)
from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics
from safety_bigym.scenarios.disruption_types import DisruptionType
from safety_bigym.scenarios.scenario_sampler import ParameterSpace, ScenarioSampler

logger = logging.getLogger(__name__)


def _task_name_to_env_class(task_name: str):
    """Resolve task name string to BiGym env class."""
    if task_name not in TASK_MAP:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {list(TASK_MAP.keys())}"
        )
    return TASK_MAP[task_name]


class SafetyBiGymEnvFactory(BiGymEnvFactory):
    """EnvFactory that wraps BiGym tasks with SafetyBiGymEnv.

    Inherits all demo loading, action rescaling, and wrapper logic
    from BiGymEnvFactory. Only overrides _create_env to inject the
    human + safety monitoring layer.

    Demo loading uses the raw BiGym env (not safety-wrapped) because
    BiGym's DemoStore indexes demos by env class name.
    """

    def _create_raw_bigym_env(self, cfg: DictConfig):
        """Create a raw BiGym env (no safety wrapper) for demo lookup.

        BiGym's DemoStore.get_demos() uses Metadata.from_env(env) which
        reads the env's class name. Our safety-wrapped class has a
        different name (SafetyReachTargetSingle vs ReachTargetSingle),
        so DemoStore can't find the demos. This method creates the
        vanilla BiGym env that DemoStore expects.
        """
        return super()._create_env(cfg)

    def _get_demo_fn(self, cfg: DictConfig, num_demos: int):
        """Load demos using a raw BiGym env for correct DemoStore lookup."""
        logging.info("Loading demos via raw BiGym env (for DemoStore compatibility).")

        # Use raw BiGym env so DemoStore sees correct class name
        env = self._create_raw_bigym_env(cfg)

        demo_store = DemoStore()
        if np.isinf(num_demos):
            num_demos = -1

        demos = demo_store.get_demos(
            Metadata.from_env(env),
            amount=num_demos,
            frequency=CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate,
        )

        for demo in demos:
            for ts in demo.timesteps:
                ts.observation = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in ts.observation.items()
                }

        env.close()
        logging.info(f"Loaded {len(demos)} demos.")
        return demos

    def _create_env(self, cfg: DictConfig):
        """Create a SafetyBiGymEnv instead of a raw BiGym env.

        The returned env has the same observation/action spaces as
        the underlying BiGym task, plus an SMPL-H human in the scene
        with ISO 15066 safety monitoring.
        """
        task_cls = _task_name_to_env_class(cfg.env.task_name)

        # Camera configuration (same as parent)
        camera_configs = [
            CameraConfig(
                name=camera_name,
                rgb=True,
                depth=False,
                resolution=cfg.visual_observation_shape,
            )
            for camera_name in cfg.env.cameras
        ]

        # Action mode (same as parent)
        if cfg.env.enable_all_floating_dof:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
                floating_dofs=[
                    PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ
                ],
            )
        else:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
            )

        # Human config from Hydra config
        human_model = cfg.env.get("human_model", "g1")
        smplh_motion = cfg.env.get("smplh_motion", "amass")
        if human_model == "g1" or (
            human_model == "smplh" and smplh_motion == "procedural"
        ):
            # No AMASS clip playback at runtime.
            motion_clip_dir = None
            motion_clip_paths = []
        else:
            motion_clip_dir = cfg.env.get(
                "motion_clip_dir", os.environ.get("AMASS_DATA_DIR")
            )
            motion_clip_paths = list(cfg.env.get("motion_clip_paths", [
                "74/74_01_poses.npz",
                "74/74_02_poses.npz",
                "09/09_01_poses.npz",
                "09/09_03_poses.npz",
                "122/122_04_poses.npz",
            ]))
        inject_human = cfg.env.get("inject_human", True)

        human_config = HumanConfig(
            motion_clip_dir=motion_clip_dir,
            motion_clip_paths=motion_clip_paths,
            human_model=human_model,
            smplh_motion=smplh_motion,
        )

        # Read reward-shaping fields from cfg.env.safety; defaults preserve
        # the pre-Phase-1.4 behaviour (penalty off). Phase 3 P3.0a adds the
        # workspace shaping triple (add_workspace_penalty / radius / beta).
        safety_cfg_block = cfg.env.get("safety", {}) or {}
        safety_config = SafetyConfig(
            log_violations=False,
            terminate_on_violation=False,
            add_violation_penalty=bool(
                safety_cfg_block.get("add_violation_penalty", False)
            ),
            violation_penalty=float(
                safety_cfg_block.get("violation_penalty", 0.05)
            ),
            add_workspace_penalty=bool(
                safety_cfg_block.get("add_workspace_penalty", False)
            ),
            workspace_radius=float(
                safety_cfg_block.get("workspace_radius", 0.4)
            ),
            workspace_beta=float(
                safety_cfg_block.get("workspace_beta", 0.05)
            ),
            workspace_excess_cap=(
                None
                if safety_cfg_block.get("workspace_excess_cap", 1.0) is None
                else float(safety_cfg_block.get("workspace_excess_cap", 1.0))
            ),
            add_progress_reward=bool(
                safety_cfg_block.get("add_progress_reward", False)
            ),
            progress_beta=float(
                safety_cfg_block.get("progress_beta", 1.0)
            ),
            progress_goal=float(
                safety_cfg_block.get("progress_goal", 0.0)
            ),
            progress_gamma=float(
                safety_cfg_block.get("progress_gamma", 0.99)
            ),
        )

        # Build a ParameterSpace honouring any cfg.env.disruptions overrides.
        # YAML may set per-type weights and tightened range fields; we copy
        # them onto the ParameterSpace defaults so the rest of the sampler
        # picks them up unchanged.
        param_space_kwargs: dict = {"clip_paths": motion_clip_paths}

        disruptions_cfg = cfg.env.get("disruptions", None)
        if disruptions_cfg is not None:
            weights_cfg = disruptions_cfg.get("weights", None)
            if weights_cfg is not None:
                weights: dict = {}
                for name, weight in weights_cfg.items():
                    try:
                        weights[DisruptionType[name]] = float(weight)
                    except KeyError as e:
                        raise ValueError(
                            f"disruptions.weights[{name!r}] is not a DisruptionType"
                        ) from e
                param_space_kwargs["disruption_weights"] = weights
            for range_field in (
                "closest_approach_range",
                "pass_by_offset_range",
                "loiter_duration_range",
                "embed_distance_range",
                "walk_speed_range",
                "spawn_distance_range",
                "arc_radius_range",
                # COWORKER continuous knobs (5 axes). Override per-axis
                # from env YAML to widen/narrow the train/eval band.
                "coworker_closest_approach_range",
                "coworker_reach_period_range",
                "coworker_target_mix_p_ee_range",
                "coworker_near_loiter_range",
                "coworker_walk_speed_range",
            ):
                value = disruptions_cfg.get(range_field, None)
                if value is not None:
                    param_space_kwargs[range_field] = tuple(value)
            traj_weights = disruptions_cfg.get("coworker_trajectory_weights", None)
            if traj_weights is not None:
                param_space_kwargs["coworker_trajectory_weights"] = {
                    str(k): float(v) for k, v in traj_weights.items()
                }

        # Optional eval knob: force every episode to use one disruption type.
        # Used by baseline_sweep.py to evaluate a trained DP against each
        # disruption type independently. Overrides any YAML weights.
        forced = cfg.env.get("disruption_type", None)
        if forced:
            try:
                dtype = DisruptionType[forced]
            except KeyError as e:
                raise ValueError(
                    f"env.disruption_type={forced!r} is not a DisruptionType "
                    f"(expected one of {[d.name for d in DisruptionType]})"
                ) from e
            param_space_kwargs["disruption_weights"] = {dtype: 1.0}
            logger.info(f"Forcing disruption_type={dtype.name} for every episode.")

        scenario_sampler = ScenarioSampler(
            parameter_space=ParameterSpace(**param_space_kwargs),
            motion_dir=motion_clip_dir,
        )

        logger.info(
            f"Creating SafetyBiGymEnv: task={task_cls.__name__}, "
            f"inject_human={inject_human}, clips={len(motion_clip_paths)}"
        )

        env = make_safety_env(
            task_cls=task_cls,
            action_mode=action_mode,
            safety_config=safety_config,
            human_config=human_config,
            scenario_sampler=scenario_sampler,
            inject_human=inject_human,
            render_mode=cfg.env.render_mode,
            observation_config=ObservationConfig(
                cameras=camera_configs if cfg.pixels else [],
                proprioception=True,
                privileged_information=False if cfg.pixels else True,
            ),
            control_frequency=CONTROL_FREQUENCY_MAX
            // cfg.env.demo_down_sample_rate,
        )
        env = self._maybe_wrap_bodyslam(env, cfg)
        return EpisodeSafetyMetrics(env)

    def _maybe_wrap_bodyslam(self, env, cfg: DictConfig):
        """Insert BodySLAMWrapper between SafetyBiGymEnv and EpisodeSafetyMetrics.

        Driven by the `bodyslam` Hydra config group (cfgs/bodyslam/{off,
        oracle,noisy}.yaml). When `mode='off'` or the block is missing, the
        env is returned untouched — preserves baseline behaviour for runs
        that don't opt into the perception layer.
        """
        bs = cfg.env.get("bodyslam") if hasattr(cfg, "env") else None
        if bs is None:
            return env
        mode = str(bs.get("mode", "off"))
        if mode == "off":
            return env
        if mode not in ("oracle", "noisy"):
            raise ValueError(
                f"env.bodyslam.mode must be one of 'off', 'oracle', 'noisy'; got {mode!r}"
            )

        occlusion_fn = NoOcclusion
        if mode == "noisy" and bool(bs.get("use_occlusion", False)):
            occlusion_fn = self._build_ray_occlusion(env) or NoOcclusion

        logger.info(
            f"Inserting BodySLAMWrapper(mode={mode}, "
            f"occlusion={'ray' if occlusion_fn is not NoOcclusion else 'none'})."
        )
        return BodySLAMWrapper(
            env,
            mode=mode,
            ou_alpha=float(bs.get("ou_alpha", 0.9)),
            noise_std=float(bs.get("noise_std", 0.05)),
            latency_steps=int(bs.get("latency_steps", 3)),
            occlusion_noise_mult=float(bs.get("occlusion_noise_mult", 3.0)),
            dropout_prob=float(bs.get("dropout_prob", 0.02)),
            seed=int(bs.get("seed", 0)),
            occlusion_fn=occlusion_fn,
        )

    def _wrap_env(self, env, cfg, demo_env=False, train=True, return_raw_spaces=False):
        """Override parent wrap to inject BodySLAMWrapper into the demo path.

        The demo env is a `DemoEnv` replaying recorded BiGym timesteps; it
        emits no `info["safety"]`, so the wrapper switches to demo_replay
        mode and pulls per-step pelvis positions from an AMASS clip via
        AMASSDemoPositionProvider. Without this override, the demo and
        train envs disagree on `low_dim_state` width and the replay buffer
        rejects demos at load time.
        """
        if demo_env:
            env = self._maybe_wrap_demo_bodyslam(env, cfg)
        return super()._wrap_env(
            env, cfg,
            demo_env=demo_env,
            train=train,
            return_raw_spaces=return_raw_spaces,
        )

    def _maybe_wrap_demo_bodyslam(self, env, cfg: DictConfig):
        bs = cfg.env.get("bodyslam") if hasattr(cfg, "env") else None
        if bs is None:
            return env
        mode = str(bs.get("mode", "off"))
        if mode == "off":
            return env
        if mode not in ("oracle", "noisy"):
            raise ValueError(
                f"env.bodyslam.mode must be one of 'off', 'oracle', 'noisy'; got {mode!r}"
            )

        motion_dir = cfg.env.get(
            "motion_clip_dir", os.environ.get("AMASS_DATA_DIR")
        )
        clip_paths = list(cfg.env.get("motion_clip_paths", []))
        if not motion_dir or not clip_paths:
            raise RuntimeError(
                "BodySLAM demo replay requires motion_clip_dir + motion_clip_paths "
                "in env config (or AMASS_DATA_DIR env var)."
            )
        provider = AMASSDemoPositionProvider(
            clip_paths=clip_paths,
            motion_dir=motion_dir,
            seed=int(bs.get("seed", 0)) ^ 0xDEAD,
        )
        logger.info(
            f"Inserting BodySLAMWrapper(mode={mode}, demo_replay=True) "
            f"around DemoEnv with AMASS provider ({len(clip_paths)} clips)."
        )
        return BodySLAMWrapper(
            env,
            mode=mode,
            ou_alpha=float(bs.get("ou_alpha", 0.9)),
            noise_std=float(bs.get("noise_std", 0.05)),
            latency_steps=int(bs.get("latency_steps", 3)),
            occlusion_noise_mult=float(bs.get("occlusion_noise_mult", 3.0)),
            dropout_prob=float(bs.get("dropout_prob", 0.02)),
            seed=int(bs.get("seed", 0)),
            occlusion_fn=NoOcclusion,
            position_provider=provider,
            demo_replay=True,
        )

    def _build_ray_occlusion(self, env):
        """Best-effort MujocoRayOcclusion factory; returns None on failure."""
        try:
            import mujoco
            unwrapped = env.unwrapped
            model = getattr(unwrapped, "_mojo", None)
            model = model.physics.model._model if model is not None else None
            data = unwrapped._mojo.physics.data._data
            cam_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "head"
            )
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "Pelvis_col"
            )
            if cam_id < 0 or geom_id < 0:
                logger.warning(
                    "BodySLAM occlusion lookup failed (cam_id=%d, geom_id=%d); "
                    "falling back to no-occlusion.", cam_id, geom_id,
                )
                return None
            return MujocoRayOcclusion(model, data, cam_id, geom_id)
        except Exception as exc:
            logger.warning(
                "BodySLAM ray occlusion init failed: %s; falling back.", exc,
            )
            return None

