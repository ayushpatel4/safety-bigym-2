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
        )

        safety_config = SafetyConfig(
            log_violations=False,
            terminate_on_violation=False,
        )

        logger.info(
            f"Creating SafetyBiGymEnv: task={task_cls.__name__}, "
            f"inject_human={inject_human}, clips={len(motion_clip_paths)}"
        )

        return make_safety_env(
            task_cls=task_cls,
            action_mode=action_mode,
            safety_config=safety_config,
            human_config=human_config,
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

