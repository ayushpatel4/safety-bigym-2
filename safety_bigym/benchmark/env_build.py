"""G1-capable eval-env builders for the benchmark harness.

``build_g1_gym_env`` mirrors ``scripts/svf_collect_dataset.py::_build_live_env`` (the
validated 4-dof / G1 / BodySLAM construction the SVF dataset was collected through) but
additionally wraps with :class:`EpisodeSafetyMetrics` so ``info["episode_safety"]`` — the
canonical ``ep_*`` source every results table reads — is emitted in this path too. (The
collector omits that wrapper because it labels transitions itself.)

Heavy imports (safety_bigym envs, bigym, mujoco) are kept lazy inside the function so the
pure benchmark modules (stats/records/schema/aggregate) stay importable without them.

G1 is AMASS-free (``human_model="g1"`` → ``motion_clip_dir=None``); the SMPL-H branch
requires ``AMASS_DATA_DIR`` exactly like the collector.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Sequence, Tuple

# (module_path.ClassName, task_id) — mirrors svf_collect_dataset.TASK_REGISTRY.
TASK_REGISTRY: Dict[str, Tuple[str, int]] = {
    "reach_target_single": ("bigym.envs.reach_target.ReachTargetSingle", 0),
    "dishwasher_close": ("bigym.envs.dishwasher.DishwasherClose", 1),
    "dishwasher_load_plates": ("bigym.envs.dishwasher_plates.DishwasherLoadPlates", 2),
    "saucepan_to_hob": ("bigym.envs.pick_and_place.SaucepanToHob", 3),
    "drawers_open_all": ("bigym.envs.cupboards.DrawersAllOpen", 4),
}

DEFAULT_CLIPS = ("74/74_01_poses.npz",)

__all__ = [
    "TASK_REGISTRY",
    "DEFAULT_CLIPS",
    "build_g1_gym_env",
    "build_cqn_cfg",
    "build_cqn_adapter",
    "make_cqn_agent",
]


def _import_task(task_key: str):
    if task_key not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {task_key!r}; choose from {sorted(TASK_REGISTRY)}")
    module_path, cls_name = TASK_REGISTRY[task_key][0].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def build_g1_gym_env(
    task_key: str,
    disruption: str,
    obs_mode: str,
    *,
    human_model: str = "g1",
    cameras: Sequence[str] = (),
    camera_resolution: Tuple[int, int] = (84, 84),
    motion_clips: Sequence[str] = DEFAULT_CLIPS,
):
    """Build ``EpisodeSafetyMetrics([BodySLAMWrapper(]SafetyBiGymEnv[)])`` for one cell.

    ``obs_mode``: ``off`` skips BodySLAM (env emits no ``human_pos_estimate``);
    ``oracle``/``noisy`` wrap it. ``disruption``: ``coworker_train``/``coworker_eval``
    use the strict-superset COWORKER ParameterSpace factories, anything else is treated
    as a single ``DisruptionType`` name.
    """
    from safety_bigym import SafetyConfig, HumanConfig, make_safety_env
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper
    from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics
    from safety_bigym.scenarios.disruption_types import DisruptionType
    from safety_bigym.scenarios.scenario_sampler import (
        ParameterSpace,
        ScenarioSampler,
        make_coworker_train_space,
        make_coworker_eval_space,
    )
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.utils.observation_config import CameraConfig, ObservationConfig

    if obs_mode not in ("off", "oracle", "noisy"):
        raise ValueError(f"obs_mode must be off/oracle/noisy, got {obs_mode!r}")

    if human_model == "g1":
        motion_clip_dir = None
        motion_clip_paths: list = []
    else:
        amass = os.environ.get("AMASS_DATA_DIR")
        if not amass:
            raise RuntimeError(
                "AMASS_DATA_DIR is not set (required for human_model=smplh). Export it:\n"
                "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
            )
        motion_clip_dir = amass
        motion_clip_paths = list(motion_clips)

    task_cls = _import_task(task_key)
    human_config = HumanConfig(
        motion_clip_dir=motion_clip_dir,
        motion_clip_paths=motion_clip_paths,
        human_model=human_model,
    )

    if disruption == "coworker_train":
        parameter_space = make_coworker_train_space(clip_paths=human_config.motion_clip_paths)
    elif disruption == "coworker_eval":
        parameter_space = make_coworker_eval_space(clip_paths=human_config.motion_clip_paths)
    else:
        parameter_space = ParameterSpace(
            clip_paths=human_config.motion_clip_paths,
            disruption_weights={DisruptionType[disruption]: 1.0},
        )
    sampler = ScenarioSampler(parameter_space=parameter_space, motion_dir=motion_clip_dir)

    make_env_kwargs: Dict[str, Any] = {}
    if cameras:
        make_env_kwargs["observation_config"] = ObservationConfig(
            cameras=[
                CameraConfig(name=name, rgb=True, depth=False,
                             resolution=tuple(camera_resolution))
                for name in cameras
            ],
            proprioception=True,
            privileged_information=False,
        )

    # 4-dof floating base (X, Y, Z, RZ) — matches RoboBase's
    # enable_all_floating_dof=True regime (action_dim=16) the SVF critic + Phase-0
    # snapshots were sized against. See svf_collect_dataset._build_live_env.
    env = make_safety_env(
        task_cls=task_cls,
        action_mode=JointPositionActionMode(
            absolute=True,
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        safety_config=SafetyConfig(terminate_on_violation=False, log_violations=False),
        human_config=human_config,
        scenario_sampler=sampler,
        inject_human=True,
        render_mode="rgb_array",
        **make_env_kwargs,
    )
    if obs_mode != "off":
        env = BodySLAMWrapper(env, mode=obs_mode)
    # EpisodeSafetyMetrics outermost (of the non-filter stack) so it sees the executed
    # action; the runtime filter, if any, is attached OUTSIDE this by filter_attach.
    env = EpisodeSafetyMetrics(env)
    return env


# ----------------------------------------------------------------------------------------
# CQN-AS path: build the adapter from the snapshot's embedded config + CLI overrides, then
# the agent from the payload. Mirrors train_cqn_as.py (make_agent + _setup_env + get_demos
# action-stat derivation). Validated on the GPU box / a real snapshot — see docs.
# ----------------------------------------------------------------------------------------

def build_cqn_cfg(snapshot_config, *, task, disruption, obs_mode, human_model, device=None):
    """Build the OmegaConf cfg for ``env_adapter.make`` from a snapshot's resolved config.

    Starts from the snapshot's own (fully-resolved) config so obs/action spaces match what
    the agent was trained against, then applies CLI overrides:
      * ``human_model`` and ``env.bodyslam.mode`` are set directly;
      * ``disruption`` / ``bodyslam`` config-group yamls are merged when present (so a
        ``coworker_eval`` OOD band or an obs-mode change pulls in the right env knobs);
      * ``task`` is overridden only when it differs from the snapshot's (with a warning) —
        the headline CQN-AS use evaluates a policy on its trained task.
    """
    import logging

    from omegaconf import OmegaConf

    logger = logging.getLogger("benchmark.env_build")
    cfg = OmegaConf.create(dict(snapshot_config))
    repo_cfgs = _repo_root() / "cfgs"

    # device (eval may run on CPU even if trained on cuda)
    if device is not None:
        OmegaConf.update(cfg, "device", device, force_add=True)
        OmegaConf.update(cfg, "agent.device", device, force_add=True)

    # task override (rare — usually keep the snapshot's trained task)
    snap_task = str(cfg.env.get("task_name", "")) if cfg.get("env") else ""
    if task and snap_task and task != snap_task:
        logger.warning("CQN-AS snapshot trained on task=%r but --task=%r; "
                       "merging env/safety_bigym/%s.yaml.", snap_task, task, task)
        env_yaml = repo_cfgs / "env" / "safety_bigym" / f"{task}.yaml"
        if env_yaml.is_file():
            cfg = OmegaConf.merge(cfg, OmegaConf.load(env_yaml))

    # disruption override via the config group (env.disruption_type/disruptions)
    if disruption:
        dpath = repo_cfgs / "disruption" / f"{disruption}.yaml"
        if dpath.is_file():
            cfg = OmegaConf.merge(cfg, OmegaConf.load(dpath))
        else:
            logger.warning("No disruption config %s; leaving snapshot disruption as-is.", dpath)

    # bodyslam / obs-mode override
    if obs_mode:
        bpath = repo_cfgs / "bodyslam" / f"{obs_mode}.yaml"
        if bpath.is_file():
            cfg = OmegaConf.merge(cfg, OmegaConf.load(bpath))
        OmegaConf.update(cfg, "env.bodyslam.mode", obs_mode, force_add=True)

    OmegaConf.update(cfg, "env.human_model", human_model, force_add=True)

    # Rebase the (machine-specific) AMASS dir baked into the snapshot config onto the
    # local AMASS_DATA_DIR so demo human-pos injection (get_demos) finds the clips on
    # this machine — motion_clip_paths are relative, so only the dir needs rebasing.
    amass = os.environ.get("AMASS_DATA_DIR")
    if amass and cfg.get("env") is not None and cfg.env.get("motion_clip_dir"):
        OmegaConf.update(cfg, "env.motion_clip_dir", amass, force_add=True)

    return cfg


def build_cqn_adapter(cfg, *, num_demos_for_stats: int = 0):
    """Return ``(wrapped_adapter, adapter)`` and set demo-derived action stats.

    The CQN snapshot does not carry ``action_stats`` (they are demo-derived). Replays
    ``get_demos`` purely for its side effect of setting ``adapter._action_stats`` — exactly
    as ``train_cqn_as`` does — so the eval-time action de-normalisation matches training.
    Requires DemoStore (+ AMASS for demo human-pos injection when bodyslam != off).
    """
    from safety_bigym.agents.cqn_as import env_adapter

    wrapped = env_adapter.make(cfg, frame_stack=int(cfg.get("frame_stack", 1)),
                               normalize_low_dim_obs=False)
    adapter = wrapped._env
    # Demo count for the action-stat side effect. Default: the snapshot's training count
    # (faithful normalisation). ``num_demos_for_stats > 0`` caps it — pixel demos are
    # memory-heavy (3 cameras x 84x84 x frame_stack per step), so a constrained machine
    # can derive approximate stats from a subset. 0 -> use the snapshot's count.
    cfg_n = int(cfg.get("num_demos", 0))
    if num_demos_for_stats and int(num_demos_for_stats) > 0:
        n = min(cfg_n, int(num_demos_for_stats)) if cfg_n > 0 else int(num_demos_for_stats)
    else:
        n = cfg_n or 5
    adapter.get_demos(max(n, 1))  # side effect: adapter._action_stats <- demo-derived
    return wrapped, adapter


def make_cqn_agent(cfg, wrapped_adapter, payload):
    """Instantiate the CQN-AS agent from cfg + load weights (mirrors train_cqn_as.make_agent)."""
    import hydra

    acfg = cfg.agent
    acfg.rgb_obs_shape = list(wrapped_adapter.rgb_observation_spec().shape)
    acfg.low_dim_obs_shape = list(wrapped_adapter.low_dim_observation_spec().shape)
    acfg.action_shape = [int(cfg.action_sequence), *wrapped_adapter.action_spec().shape]
    agent = hydra.utils.instantiate(acfg)
    agent.load_state_dict(payload["agent_state"])
    return agent


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
