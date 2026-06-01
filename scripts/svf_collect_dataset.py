#!/usr/bin/env python
"""SVF transition dataset collector.

Phase 2 — collects ``(obs, action, next_obs, r_safe, done, ssm_margin, ...)``
transitions from rollouts in ``SafetyBiGymEnv + BodySLAMWrapper`` and writes
them as sharded ``.npz`` for the offline CQL trainer.

Sources (CLI flag, repeatable):
  ``random``    — uniform actions sampled from ``env.action_space``
  ``demo``      — replay BiGym demonstrations via DemoStore; ``human_pos_estimate``
                  is synthesised from an AMASS clip (no live human in the recorded
                  demo). All transitions labelled ``r_safe=1`` (demos are safe by
                  construction; ``ssm_margin`` is set to a constant placeholder).
  ``snapshot``  — load an ACT/DP snapshot (workspace.py drift) and roll it out.
                  Requires --snapshot-path. Phase-0 ACT snapshots are pending
                  GPU retrain; until then this path is exercised only in error
                  branches by the smoke test.

Disruption space (Phase 0.5 onward):
  Default ``--disruption-space coworker_train`` collects a single cell using
  ``make_coworker_train_space()`` — the strict-superset COWORKER ParameterSpace
  with five continuous parameter axes (closest approach, reach period, target
  mix, near loiter, walk speed) in moderate bands. ``coworker_eval`` is the
  wider eval ParameterSpace; ``legacy_multi`` reinstates the pre-0.5 mixture
  of {INCIDENTAL, SHARED_GOAL, DIRECT, OBSTRUCTION, RANDOM_PERTURBED} and
  iterates over the --disruptions string list.

Smoke mode runs 1 task × coworker_train × 2 episodes × ≤50 steps from the
random source and writes a single ``_smoke_shard.npz``.

Usage:
    python scripts/svf_collect_dataset.py --smoke
    python scripts/svf_collect_dataset.py --source random --source demo \\
        --tasks reach_target_single --episodes-per-cell 50
    # legacy escape hatch:
    python scripts/svf_collect_dataset.py --disruption-space legacy_multi \\
        --disruptions INCIDENTAL DIRECT
    python scripts/svf_collect_dataset.py --source snapshot \\
        --snapshot-path exp_local/act_safety/.../snapshots/60000_snapshot.pt

Hand-off to GPU for the full ~310k-transition collection.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("svf_collect_dataset")


AMASS_DATA_DIR = os.environ.get("AMASS_DATA_DIR")


SOURCE_CODES: Dict[str, int] = {"demo": 0, "random": 1, "snapshot": 2}

TASK_REGISTRY: Dict[str, Tuple[str, int]] = {
    "reach_target_single": ("bigym.envs.reach_target.ReachTargetSingle", 0),
    "dishwasher_close": ("bigym.envs.dishwasher.DishwasherClose", 1),
    "dishwasher_load_plates": ("bigym.envs.dishwasher_plates.DishwasherLoadPlates", 2),
    "saucepan_to_hob": ("bigym.envs.pick_and_place.SaucepanToHob", 3),
    "drawers_open_all": ("bigym.envs.cupboards.DrawersAllOpen", 4),
}

DEFAULT_DISRUPTIONS = (
    "INCIDENTAL",
    "SHARED_GOAL",
    "DIRECT",
    "OBSTRUCTION",
    "RANDOM_PERTURBED",
)

# Sentinel cell labels for the coworker ParameterSpace factories. When a
# disruption string is one of these, _build_live_env dispatches to
# make_coworker_{train,eval}_space() instead of looking up DisruptionType[name].
COWORKER_CELL_LABELS = ("coworker_train", "coworker_eval")

DISRUPTION_SPACE_CHOICES = ("coworker_train", "coworker_eval", "legacy_multi")

DEFAULT_CLIPS = ("74/74_01_poses.npz",)

# Keys ConcatDim excludes from low_dim_state at training time
# ([robobase/envs/bigym.py:95]). The snapshot adapter must replicate this
# exclusion exactly or the actor's first layer receives a wrong-dim vector.
LOW_DIM_KEYS_TO_IGNORE: Tuple[str, ...] = ("proprioception_floating_base_actions",)

HUMAN_POS_ESTIMATE_KEY = "human_pos_estimate"

# Demos are recorded without a live human, so we synthesise per-demo human
# pelvis positions from an AMASS clip. Constant safe placeholder margin since
# the recorded trajectories never touch the human.
DEMO_PLACEHOLDER_MARGIN = 1.0
DEMO_HUMAN_OCCLUDED = 0.0
DEMO_HUMAN_STALENESS = 0.0
DEMO_HUMAN_CONFIDENCE = 1.0


@dataclass
class CollectionPlan:
    sources: Tuple[str, ...]
    tasks: Tuple[str, ...]
    disruptions: Tuple[str, ...]
    episodes_per_cell: int
    max_steps: int
    bodyslam_mode: str
    output_dir: Path
    seed: int = 0
    motion_clips: Tuple[str, ...] = DEFAULT_CLIPS
    demos_per_task: int = 30  # used only by the demo source
    # Coworker embodiment. "g1" (default) is AMASS-free; "smplh" requires
    # AMASS_DATA_DIR and replays motion clips. Flows into every _build_live_env.
    human_model: str = "g1"
    # Per-task snapshot path overrides — populated by --snapshot-override CLI
    # flags. Empty by default; the resolver falls through to
    # safety_bigym.filters.snapshots.SNAPSHOTS.
    snapshot_overrides: Mapping[str, str] = field(default_factory=dict)
    # Geometric near-contact threshold (metres) used by label_transition. The
    # SVF's binary r_safe = (min_separation >= proximity_threshold). Default
    # matches labeling.label_transition's own default; surface as a CLI knob
    # because it's the most likely thing to sweep.
    proximity_threshold: float = 0.10

    @classmethod
    def smoke(cls, output_dir: Path) -> "CollectionPlan":
        return cls(
            sources=("random",),
            tasks=("dishwasher_close",),
            disruptions=("coworker_train",),
            episodes_per_cell=2,
            max_steps=50,
            bodyslam_mode="oracle",
            output_dir=output_dir,
            seed=0,
        )


def _import_task(task_key: str):
    if task_key not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {task_key!r}; known: {sorted(TASK_REGISTRY)}")
    dotted, _ = TASK_REGISTRY[task_key]
    module_path, _, cls_name = dotted.rpartition(".")
    import importlib
    return getattr(importlib.import_module(module_path), cls_name)


def _resolve_demo_down_sample_rate(task_key: str, default: int = 20) -> int:
    """Read ``demo_down_sample_rate`` for a task from its env config yaml.

    The collection/sweep env MUST run at the SAME control rate as deployment
    (the RoboBase factory ``_create_env`` sets ``control_frequency =
    CONTROL_FREQUENCY_MAX // demo_down_sample_rate``). The base
    ``cfgs/env/safety_bigym.yaml`` defaults the rate; the per-task yaml
    (e.g. ``saucepan_to_hob.yaml`` -> 25) overrides it.
    """
    from omegaconf import OmegaConf

    rate = default
    base = REPO_ROOT / "cfgs" / "env" / "safety_bigym.yaml"
    task = REPO_ROOT / "cfgs" / "env" / "safety_bigym" / f"{task_key}.yaml"
    for p in (base, task):
        try:
            if p.is_file():
                r = OmegaConf.select(OmegaConf.load(p), "env.demo_down_sample_rate")
                if r is not None:
                    rate = int(r)
        except Exception:  # noqa: BLE001 — fall back to the running default
            pass
    return rate


def _resolve_coworker_overrides(disruption: str) -> Dict[str, Any]:
    """Read coworker_* ParameterSpace overrides from a disruption config yaml.

    The deployment factory (``safety_bigym_factory._create_env``) builds the
    COWORKER ParameterSpace from ``cfg.env.disruptions`` (the merged disruption
    yaml). The Python presets (``_COWORKER_*_RANGES``) drifted from those yamls,
    so collection must read the yaml to match deployment. Returns a dict of
    overrides (``coworker_*_range`` as tuples + ``coworker_trajectory_weights``
    as a dict), or ``{}`` if the yaml/section is absent (caller falls back to
    the Python preset).
    """
    from omegaconf import OmegaConf

    yaml = REPO_ROOT / "cfgs" / "disruption" / f"{disruption}.yaml"
    if not yaml.is_file():
        return {}
    try:
        sect = OmegaConf.select(OmegaConf.load(yaml), "env.disruptions")
        if sect is None:
            return {}
        sect = OmegaConf.to_container(sect, resolve=True)
    except Exception:  # noqa: BLE001 — fall back to the preset
        return {}
    out: Dict[str, Any] = {}
    for key, val in (sect or {}).items():
        if key.endswith("_range") and isinstance(val, (list, tuple)):
            out[key] = tuple(val)
        elif key == "coworker_trajectory_weights" and isinstance(val, dict):
            out[key] = dict(val)
    return out


def _build_live_env(
    task_key: str,
    disruption: str,
    mode: str,
    motion_clips: Sequence[str],
    *,
    cameras: Sequence[str] = (),
    camera_resolution: Tuple[int, int] = (84, 84),
    human_model: str = "g1",
    demo_down_sample_rate: Optional[int] = None,
):
    """Construct ``[BodySLAMWrapper(]SafetyBiGymEnv[)]`` — used by random/snapshot.

    ``mode``: "oracle" | "noisy" wraps with BodySLAMWrapper; "off" skips the
    wrapper entirely so the env emits no ``human_pos_estimate`` (required for
    Phase 0 ACT snapshots, which were trained without BodySLAMWrapper —
    instantiating the agent against a space that contains the extra key
    sizes the input layer wrong and silent state_dict mismatches result).

    ``cameras`` defaults to empty (bare env, no rendering) so random/demo
    callers don't pay the MuJoCo render cost. Pixel-trained snapshot policies
    must pass the cameras list from their snapshot's ``cfg.env.cameras`` so
    the obs dict carries the ``rgb_<name>`` keys the actor's encoder expects.

    ``human_model``: "g1" (default) loads the AMASS-free Unitree G1 coworker;
    "smplh" loads the AMASS-driven SMPL-H coworker. Mirrors the G1 branch in
    ``safety_bigym_factory._create_env`` — for G1 the AMASS requirement and
    motion-clip injection are skipped entirely (``motion_clip_dir=None``).
    """
    # AMASS is only needed for the SMPL-H clip-playback path; G1 is AMASS-free.
    if human_model == "g1":
        motion_clip_dir = None
        motion_clip_paths: list = []
    else:
        if not AMASS_DATA_DIR:
            raise RuntimeError(
                "AMASS_DATA_DIR is not set. Export it before running:\n"
                "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
            )
        motion_clip_dir = AMASS_DATA_DIR
        motion_clip_paths = list(motion_clips)

    from safety_bigym import SafetyConfig, HumanConfig, make_safety_env
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper
    from safety_bigym.scenarios.disruption_types import DisruptionType
    from safety_bigym.scenarios.scenario_sampler import (
        ParameterSpace,
        ScenarioSampler,
        _coworker_space,
        make_coworker_train_space,
        make_coworker_eval_space,
    )
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.bigym_env import CONTROL_FREQUENCY_MAX
    from bigym.utils.observation_config import CameraConfig, ObservationConfig

    task_cls = _import_task(task_key)
    human_config = HumanConfig(
        motion_clip_dir=motion_clip_dir,
        motion_clip_paths=motion_clip_paths,
        human_model=human_model,
    )
    # Cell label dispatch. For coworker_train/coworker_eval, build the COWORKER
    # ParameterSpace from the disruption YAML — the SAME source the deployment
    # factory (_create_env) reads via cfg.env.disruptions. The Python presets
    # (_COWORKER_*_RANGES used by make_coworker_*_space) drifted from those yamls
    # (coworker_train was tightened 2026-05-27: closest 0.60-0.95 / fast reach
    # 1.3-2.2, but the preset stayed loose 0.9-1.4 / slow 4.5-6.5), so a
    # preset-built coworker stays too far and the collection sees ~0 proximity
    # while deployment sees 0.30. Reading the yaml makes collection==deployment.
    # Any other disruption is a legacy DisruptionType pinned to a 1.0-weight space.
    if disruption in ("coworker_train", "coworker_eval"):
        _ov = _resolve_coworker_overrides(disruption)
        if _ov:
            _ranges = {k: v for k, v in _ov.items() if k.endswith("_range")}
            _extra = {k: v for k, v in _ov.items() if not k.endswith("_range")}
            parameter_space = _coworker_space(
                _ranges, clip_paths=human_config.motion_clip_paths, **_extra
            )
            logger.info(
                "Coworker scenario for %s from disruption yaml (matches "
                "deployment): %s", disruption, {k: _ov[k] for k in sorted(_ov)},
            )
        else:
            logger.warning(
                "Disruption yaml for %s not found/empty; falling back to the "
                "Python preset (may drift from deployment).", disruption,
            )
            _builder = (
                make_coworker_train_space
                if disruption == "coworker_train"
                else make_coworker_eval_space
            )
            parameter_space = _builder(clip_paths=human_config.motion_clip_paths)
    else:
        parameter_space = ParameterSpace(
            clip_paths=human_config.motion_clip_paths,
            disruption_weights={DisruptionType[disruption]: 1.0},
        )
    sampler = ScenarioSampler(
        parameter_space=parameter_space,
        motion_dir=motion_clip_dir,
    )
    make_env_kwargs: Dict[str, Any] = {}
    if cameras:
        make_env_kwargs["observation_config"] = ObservationConfig(
            cameras=[
                CameraConfig(
                    name=name, rgb=True, depth=False,
                    resolution=tuple(camera_resolution),
                )
                for name in cameras
            ],
            proprioception=True,
            privileged_information=False,
        )
    # 4-dof floating base (X, Y, Z, RZ) mirrors RoboBase's BiGym factory under
    # `cfg.env.enable_all_floating_dof=True` — the regime the Phase-0 ACT
    # snapshots were trained under (action_dim=16, qpos=66). The bare-BiGym
    # default of 3 dofs (X, Y, RZ) gives action_dim=15 and silent
    # state_dict shape mismatches at snapshot load. See B1.4 / B2.3 debug.
    # Match the DEPLOYMENT env's control rate. The RoboBase factory builds the
    # env with control_frequency = CONTROL_FREQUENCY_MAX // demo_down_sample_rate
    # (safety_bigym_factory._create_env). Omitting it left the collection env at
    # the full 500 Hz (1 physics substep/step, action_scale=1) vs deployment's
    # downsampled rate (e.g. 20 Hz / 25 substeps / action_scale=25 for saucepan).
    # That single mismatch moved the robot ~25x less per step (policy never
    # completed the task: 0% success vs deployment's 85%) and made 1000 steps
    # cover ~1/25 the wall-time (coworker barely approached: proximity 0.05 vs
    # 0.30) — the collection<->deployment divergence diagnosed 2026-06-01.
    _ds_rate = (
        int(demo_down_sample_rate)
        if demo_down_sample_rate is not None
        else _resolve_demo_down_sample_rate(task_key)
    )
    _control_freq = max(1, CONTROL_FREQUENCY_MAX // _ds_rate)
    logger.info(
        "Collection env control_frequency=%d Hz (demo_down_sample_rate=%d) — "
        "matches deployment factory._create_env.",
        _control_freq, _ds_rate,
    )
    env = make_safety_env(
        task_cls=task_cls,
        action_mode=JointPositionActionMode(
            absolute=True,
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        # Suppress the per-step "SSM Violation!" WARNING spam: Phase 2 labels
        # transitions by geometric proximity (B2.7), so the ISO 15066-based
        # SSM-violation flag is now informational and fires on essentially
        # every step at kitchen scale. The warnings drown out useful output
        # during multi-task collection. ssm_margin / min_separation remain
        # populated on info["safety"] for traceability and shard storage.
        safety_config=SafetyConfig(
            terminate_on_violation=False,
            log_violations=False,
        ),
        human_config=human_config,
        scenario_sampler=sampler,
        inject_human=True,
        control_frequency=_control_freq,
        **make_env_kwargs,
    )
    if mode == "off":
        return env
    if mode not in ("oracle", "noisy"):
        raise ValueError(
            f"_build_live_env: bodyslam mode must be one of off/oracle/noisy; "
            f"got {mode!r}"
        )
    return BodySLAMWrapper(env, mode=mode)


# ---------- random policy -----------------------------------------------------


def random_policy(env, rng: np.random.Generator) -> Callable[[dict], np.ndarray]:
    space = env.action_space
    low = space.low.astype(np.float32)
    high = space.high.astype(np.float32)

    def _act(_obs: dict) -> np.ndarray:
        return rng.uniform(low=low, high=high).astype(np.float32)

    return _act


# ---------- live-env rollout (random + snapshot share this path) --------------


def _filter_obs(obs: dict, obs_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(obs[k], dtype=np.float32) for k in obs_keys}


def rollout_episode(
    env,
    policy: Callable[[dict], np.ndarray],
    *,
    max_steps: int,
    obs_keys: Sequence[str],
    use_pfl: bool,
    proximity_threshold: float,
) -> Optional[Dict[str, Any]]:
    from safety_bigym.filters.labeling import label_transition

    obs, _info = env.reset()
    # Stateful policies (e.g. CQN-AS frame-stacked snapshots) must clear their
    # per-episode buffers at the episode boundary. Stateless policies (random,
    # ACT _SnapshotPolicy) have no reset() and are unaffected.
    if hasattr(policy, "reset"):
        policy.reset()
    obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in obs_keys}
    next_obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in obs_keys}
    actions: List[np.ndarray] = []
    r_safe_list: List[float] = []
    done_list: List[bool] = []
    margins: List[float] = []
    min_seps: List[float] = []
    pfl_ratios: List[float] = []

    for _ in range(max_steps):
        action = policy(obs).astype(np.float32, copy=False)
        prev = _filter_obs(obs, obs_keys)

        obs, _reward, terminated, truncated, info = env.step(action)
        nxt = _filter_obs(obs, obs_keys)

        # Physics instability (e.g. random actions colliding with the close
        # coworker at the 20 Hz control rate can blow up MuJoCo: NaN/Inf QACC ->
        # BiGym's EnvHealth truncates). Drop the bad step WITHOUT storing it, so
        # the SVF dataset stays finite; the prior good transitions in this
        # episode are kept and the outer loop resets for the next episode.
        _finite = all(
            bool(np.all(np.isfinite(np.asarray(nxt[k], dtype=np.float32))))
            for k in obs_keys
        ) and bool(np.all(np.isfinite(np.asarray(action, dtype=np.float32))))
        if not _finite:
            logger.warning(
                "Non-finite obs/action at step %d (physics instability); "
                "truncating episode without storing this step.",
                len(actions),
            )
            break

        if "safety" not in info:
            continue

        r_safe, viol_terminal = label_transition(
            info, use_pfl=use_pfl, proximity_threshold=proximity_threshold,
        )
        for k in obs_keys:
            obs_buf[k].append(prev[k])
            next_obs_buf[k].append(nxt[k])
        actions.append(action)
        r_safe_list.append(r_safe)
        done_list.append(bool(viol_terminal or terminated or truncated))
        safety_info = info["safety"]
        margins.append(float(safety_info.get("ssm_margin", 0.0)))
        # Store the raw signals so r_safe can be recomputed later (proximity
        # threshold sweep, PFL retrofit) without re-collecting transitions.
        # ``pfl_force_ratio`` is currently identically zero; see CLAUDE.md.
        min_seps.append(float(safety_info.get("min_separation", float("inf"))))
        pfl_ratios.append(float(safety_info.get("pfl_force_ratio", 0.0)))

        if terminated or truncated:
            break

    if not actions:
        return None

    return {
        "obs": {k: np.stack(v, axis=0) for k, v in obs_buf.items()},
        "next_obs": {k: np.stack(v, axis=0) for k, v in next_obs_buf.items()},
        "action": np.stack(actions, axis=0),
        "r_safe": np.asarray(r_safe_list, dtype=np.float32),
        "done": np.asarray(done_list, dtype=np.bool_),
        "ssm_margin": np.asarray(margins, dtype=np.float32),
        "min_separation": np.asarray(min_seps, dtype=np.float32),
        "pfl_force_ratio": np.asarray(pfl_ratios, dtype=np.float32),
    }


# ---------- demo source -------------------------------------------------------


def _build_human_pos_estimate(pelvis_xyz: np.ndarray) -> np.ndarray:
    """Construct the 6-D ``human_pos_estimate`` vector for demo replay.

    Mirrors ``BodySLAMWrapper._build_estimate``; demos have no occlusion or
    staleness, so the auxiliary fields are pinned at their ``oracle`` defaults.
    """
    return np.array(
        [
            float(pelvis_xyz[0]),
            float(pelvis_xyz[1]),
            float(pelvis_xyz[2]),
            DEMO_HUMAN_OCCLUDED,
            DEMO_HUMAN_STALENESS,
            DEMO_HUMAN_CONFIDENCE,
        ],
        dtype=np.float32,
    )


def _make_amass_provider(motion_clips: Sequence[str], seed: int):
    from safety_bigym.perception.demo_position_provider import AMASSDemoPositionProvider

    if not AMASS_DATA_DIR:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it before running:\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
        )
    return AMASSDemoPositionProvider(
        clip_paths=list(motion_clips),
        motion_dir=AMASS_DATA_DIR,
        seed=int(seed),
    )


def fetch_demos(task_key: str, num_demos: int, frequency: int = 50):
    """Load BiGym demos for one task via ``DemoStore``.

    Returns the list of ``Demo`` objects (each with ``.timesteps``). The env is
    constructed only for ``Metadata.from_env`` and closed immediately after.

    If the store has no demos matching this task's metadata signature, returns
    an empty list and logs a warning (the caller skips the task rather than
    crashing the whole collection run).
    """
    from demonstrations.demo_store import DemoStore, DemoNotFoundError
    from demonstrations.utils import Metadata
    from bigym.action_modes import JointPositionActionMode

    task_cls = _import_task(task_key)
    env = task_cls(
        action_mode=JointPositionActionMode(absolute=True, floating_base=True),
    )
    try:
        store = DemoStore()
        try:
            demos = store.get_demos(
                Metadata.from_env(env),
                amount=num_demos,
                frequency=frequency,
            )
        except DemoNotFoundError as e:
            logger.warning(
                f"DemoStore has no demos matching {task_key} metadata "
                f"(action_mode=JointPositionActionMode, floating_base=True). "
                f"Skipping demo source for this task. "
                f"Underlying error: {e}"
            )
            return []
        for demo in demos:
            for ts in demo.timesteps:
                ts.observation = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in ts.observation.items()
                }
        return demos
    finally:
        env.close()


def demo_episode_to_transitions(
    timesteps: Sequence[Any],
    *,
    spec_obs_keys: Sequence[str],
    amass_provider,
) -> Optional[Dict[str, np.ndarray]]:
    """Turn one demo's timesteps into the per-shard array bundle.

    Demos contain ``(observation, action)`` pairs but no live safety physics —
    so we set ``r_safe = 1`` and ``ssm_margin = DEMO_PLACEHOLDER_MARGIN`` for
    every transition. ``human_pos_estimate`` is synthesised by the AMASS
    provider so the channel is non-degenerate.
    """
    if len(timesteps) < 2:
        return None

    amass_provider.reset()
    obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in spec_obs_keys}
    next_obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in spec_obs_keys}
    actions: List[np.ndarray] = []

    for step_idx, (ts, next_ts) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # ``DemoStep.executed_action`` reads from ``info[ACTION_KEY]``; the
        # bare ``.action`` attribute does not exist (see bigym/demonstrations/demo.py).
        raw_action = ts.executed_action
        if raw_action is None:
            return None
        action = np.asarray(raw_action, dtype=np.float32)
        prev_obs = dict(ts.observation)
        next_obs = dict(next_ts.observation)
        prev_obs["human_pos_estimate"] = _build_human_pos_estimate(
            amass_provider(step_idx)
        )
        next_obs["human_pos_estimate"] = _build_human_pos_estimate(
            amass_provider(step_idx + 1)
        )
        for k in spec_obs_keys:
            if k not in prev_obs:
                # Demo obs is missing a key the spec expects — skip this episode
                # rather than corrupting the dataset.
                logger.warning(
                    f"Demo timestep missing obs key {k!r}; skipping episode."
                )
                return None
            obs_buf[k].append(np.asarray(prev_obs[k], dtype=np.float32))
            next_obs_buf[k].append(np.asarray(next_obs[k], dtype=np.float32))
        actions.append(action)

    n = len(actions)
    return {
        "obs": {k: np.stack(v, axis=0) for k, v in obs_buf.items()},
        "next_obs": {k: np.stack(v, axis=0) for k, v in next_obs_buf.items()},
        "action": np.stack(actions, axis=0),
        "r_safe": np.ones(n, dtype=np.float32),
        "done": np.concatenate([np.zeros(n - 1, dtype=np.bool_), [True]])
        if n > 0
        else np.zeros(0, dtype=np.bool_),
        "ssm_margin": np.full(n, DEMO_PLACEHOLDER_MARGIN, dtype=np.float32),
        # Demos have no live human-robot physics; use a safe-side placeholder
        # large enough to never trip any plausible proximity threshold and a
        # zero PFL ratio (no contact recorded). Demo source is currently
        # disabled in B3 (see CLAUDE.md), but keep the schema consistent.
        "min_separation": np.full(n, 10.0, dtype=np.float32),
        "pfl_force_ratio": np.zeros(n, dtype=np.float32),
    }


# ---------- snapshot source ---------------------------------------------------


_RAW_TANH_WARNED = False


def _warn_raw_tanh_once() -> None:
    """Log a single warning per process when a snapshot policy fires without
    action_stats — i.e. we're producing raw tanh-space actions (the v1
    behaviour, see B4.2 caveat). New collections should set action_stats so
    actions span the env's true range."""
    global _RAW_TANH_WARNED
    if _RAW_TANH_WARNED:
        return
    _RAW_TANH_WARNED = True
    logger.warning(
        "Snapshot policy missing action_stats — emitting raw tanh-space "
        "actions. This is the v1 behaviour; env will clip grippers and "
        "body joints. Re-collect with a snapshot whose payload contains "
        "action_stats (workspace.py drift) to fix. See "
        "docs/phase2_results.md §B5.5."
    )


@dataclass
class _SnapshotPolicy:
    """Wraps a RoboBase-loaded agent into a policy callable.

    The agent expects post-wrap observations: ``low_dim_state`` (concat of
    1-D proprio keys via ConcatDim, optionally including
    ``human_pos_estimate`` when the policy was trained with BodySLAMWrapper)
    plus one ``rgb_<camera>`` torch tensor per camera in
    ``(B=1, T=1, C=3, H, W)`` shape. ``adapt_obs`` replicates ConcatDim's
    concatenation rule exactly (iterate obs in insertion order, skip keys
    in ``LOW_DIM_KEYS_TO_IGNORE``, gate ``human_pos_estimate`` on
    ``includes_human_pos``). The encoder applies ``/255`` + ImageNet
    normalize on pixels internally.

    ``includes_human_pos`` is set by ``load_snapshot_policy`` from the
    snapshot's ``cfg.env.bodyslam.mode``: oracle/noisy ⇒ True (Phase 1+),
    off/missing ⇒ False (Phase 0). Phase 0 snapshots get a shorter
    low_dim_state that omits the 6-D human pose estimate; Phase 1
    snapshots include it.

    **Action denormalization (B5.5, 2026-05-20).** ACT/DP actors return raw
    tanh-space outputs in [-1, 1]; RoboBase training wraps the env with
    :class:`RescaleFromTanhWithMinMax`, so the deployed policy sees actions
    mapped back to the env's true ±π body-joint range and [0, 1] gripper
    range. The SVF collection path doesn't replicate that wrapper, so we
    apply the same transform in :meth:`__call__` when ``action_stats`` is
    set. ``action_stats`` is read by :func:`load_snapshot_policy` from
    ``payload["action_stats"]`` (written by the FYP3 robobase workspace
    drift); ``min_max_margin`` comes from the snapshot's ``cfg.min_max_margin``
    (default 0.0). Without the stats the policy returns raw tanh outputs and
    logs a warning once, preserving the v1 behaviour for old snapshots.
    """

    agent: Any
    cameras: Tuple[str, ...] = ()  # empty ⇒ no-pixel policy
    camera_resolution: Tuple[int, int] = (84, 84)
    includes_human_pos: bool = False
    # B5.5: tanh-denormalization payload. None ⇒ legacy raw-tanh behaviour.
    action_stats: Optional[Dict[str, np.ndarray]] = None
    min_max_margin: float = 0.0

    @property
    def expects_pixels(self) -> bool:
        return len(self.cameras) > 0

    def adapt_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        import torch

        # Replicate ConcatDim: iterate obs in insertion order (matches the
        # env's emit order, which is what ConcatDim sees at training), skip
        # keys in LOW_DIM_KEYS_TO_IGNORE, gate human_pos_estimate on the
        # policy's bodyslam-trained flag, and concat all remaining 1-D
        # values in order.
        low_dim_pieces: list[np.ndarray] = []
        for key, value in obs.items():
            if key in LOW_DIM_KEYS_TO_IGNORE:
                continue
            if key == HUMAN_POS_ESTIMATE_KEY and not self.includes_human_pos:
                continue
            arr = np.asarray(value)
            if arr.ndim != 1:
                continue  # pixels and other multi-D keys go through their own paths
            low_dim_pieces.append(arr.astype(np.float32, copy=False).reshape(-1))

        if not low_dim_pieces:
            # Fall back to a pre-concat'd key if the env was already wrapped
            # (e.g. shipped post-ConcatDim form). Rare; mostly defensive.
            low_dim_pieces.append(
                np.asarray(obs.get("low_dim_state", np.zeros(0)), dtype=np.float32).reshape(-1)
            )
        low_dim = np.concatenate(low_dim_pieces)

        out: Dict[str, Any] = {
            "low_dim_state": torch.from_numpy(low_dim).unsqueeze(0).unsqueeze(0)
        }
        for cam in self.cameras:
            key = f"rgb_{cam}"
            if key not in obs:
                raise KeyError(
                    f"Snapshot policy expects obs key {key!r} but it's missing. "
                    f"Did you build the env with cameras={list(self.cameras)!r}?"
                )
            arr = np.asarray(obs[key])
            # Bare bigym emits (H, W, 3) uint8. Permute HWC → CHW, add B + T.
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.ndim == 3 and arr.shape[0] == 3:
                pass  # already CHW
            else:
                raise ValueError(
                    f"Unexpected pixel shape for {key!r}: got {arr.shape}, "
                    "expected (H, W, 3) or (3, H, W)."
                )
            # ACT's encoder reads uint8 and applies /255 + ImageNet normalize.
            out[key] = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0)
        return out

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        import torch

        with torch.no_grad():
            adapted = self.adapt_obs(obs)
            action = self.agent.act(adapted, step=0, eval_mode=True)
        action_np = action.detach().cpu().numpy()
        # ACT returns a chunk; take the first step.
        if action_np.ndim >= 2:
            action_np = action_np.reshape(-1, action_np.shape[-1])[0]
        action_np = action_np.astype(np.float32, copy=False)
        # B5.5: replicate RoboBase's RescaleFromTanhWithMinMax. Without this
        # the env silently clips gripper dims to [0, 1] and body-joint dims
        # explore only the inner [-1, 1] band of the ±π range. See B4.2
        # caveat in docs/phase2_results.md.
        if self.action_stats is not None:
            from robobase.envs.wrappers.rescale_from_tanh import (
                RescaleFromTanhWithMinMax,
            )

            action_np = RescaleFromTanhWithMinMax.transform_from_tanh(
                action_np, self.action_stats, self.min_max_margin
            )
            action_np = np.asarray(action_np, dtype=np.float32)
        else:
            _warn_raw_tanh_once()
        return action_np


def _peek_snapshot_cfg(snapshot_path: Path):
    """Load just the cfg field out of a snapshot payload."""
    if not snapshot_path or not Path(snapshot_path).is_file():
        raise FileNotFoundError(f"snapshot_path={snapshot_path!r} not found.")
    import torch
    from omegaconf import DictConfig, OmegaConf

    payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    # RoboBase ACT/DP snapshots store the Hydra cfg under "cfg"; CQN-AS
    # (train_cqn_as.py) stores a resolved container under "config". Accept both.
    cfg = payload.get("cfg")
    if cfg is None:
        cfg = payload.get("config")
    if cfg is None:
        raise KeyError(
            f"snapshot at {snapshot_path} has neither a 'cfg' (RoboBase) nor a "
            "'config' (CQN-AS) field — was it produced after the workspace.py "
            "drift fix? See CLAUDE.md."
        )
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    return cfg


def peek_snapshot_bodyslam_mode(snapshot_path: Path) -> str:
    """Return the BodySLAM mode the snapshot was trained with.

    ``"off"`` ⇒ Phase 0 ACT (no human-state observation). ``"oracle"`` or
    ``"noisy"`` ⇒ Phase 1+ ACT (trained with BodySLAMWrapper). Determines
    whether ``_SnapshotPolicy.adapt_obs`` should append ``human_pos_estimate``
    to ``low_dim_state`` AND what BodySLAM mode the eval env should use so
    the input noise distribution matches training.
    """
    cfg = _peek_snapshot_cfg(snapshot_path)
    bs = cfg.env.get("bodyslam") if "env" in cfg else None
    if bs is None:
        return "off"
    mode = str(bs.get("mode", "off"))
    if mode not in ("off", "oracle", "noisy"):
        raise ValueError(
            f"Unexpected bodyslam.mode={mode!r} in snapshot {snapshot_path}; "
            "expected one of off/oracle/noisy."
        )
    return mode


def peek_snapshot_cameras(
    snapshot_path: Path,
) -> Tuple[Tuple[str, ...], Tuple[int, int]]:
    """Read just the camera config out of a snapshot's embedded cfg.

    Returns ``(camera_names, (H, W))``. ``camera_names`` is empty for
    non-pixel snapshots. Used by ``_collect_live_env_source`` to build a
    camera-equipped env before fully loading the agent.
    """
    cfg = _peek_snapshot_cfg(snapshot_path)
    if not bool(cfg.get("pixels", False)):
        return (), (0, 0)
    cameras = tuple(str(c) for c in cfg.env.get("cameras", []))
    shape = cfg.get("visual_observation_shape")
    if shape is None:
        resolution = (84, 84)
    else:
        resolution = (int(shape[0]), int(shape[1]))
    return cameras, resolution


def _synthesize_snapshot_obs_space(
    env,
    cameras: Sequence[str],
    resolution: Tuple[int, int],
    includes_human_pos: bool,
):
    """Build the post-wrap observation_space the snapshot agent was trained against.

    The env we hand to ``hydra.utils.instantiate`` must look like what RoboBase's
    BiGym factory produced at training: ``ConcatDim(shape_length=1)`` collapses
    all 1-D obs keys into a single ``low_dim_state`` channel (skipping
    ``LOW_DIM_KEYS_TO_IGNORE``), and ``FrameStack(frame_stack=1)`` prepends a
    ``T=1`` axis to every remaining key (so rgb becomes ``(T, C, H, W)`` —
    [robobase/method/bc.py:155] multiplies the first two dims, asserting 4-D).

    For Phase-0 snapshots (``includes_human_pos=False``) we omit
    ``human_pos_estimate`` from the low_dim_state sum so the synthesized dim
    matches the actor's training-time input size, even though the wrapped env
    *does* emit the key (the runtime adapter strips it via the same flag).
    """
    from gymnasium import spaces

    low_dim_total = 0
    for key, space in env.observation_space.spaces.items():
        if not isinstance(space, spaces.Box):
            continue
        if len(space.shape) != 1:
            continue
        if key in LOW_DIM_KEYS_TO_IGNORE:
            continue
        if key == HUMAN_POS_ESTIMATE_KEY and not includes_human_pos:
            continue
        low_dim_total += int(space.shape[0])

    out: Dict[str, Any] = {}
    if low_dim_total > 0:
        out["low_dim_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, low_dim_total),
            dtype=np.float32,
        )
    for cam in cameras:
        out[f"rgb_{cam}"] = spaces.Box(
            low=0,
            high=255,
            shape=(1, 3, int(resolution[0]), int(resolution[1])),
            dtype=np.uint8,
        )
    return spaces.Dict(out)


def load_snapshot_policy(
    snapshot_path: Path, env, *, rollout_max_steps: Optional[int] = None
) -> _SnapshotPolicy:
    """Load a RoboBase snapshot and return a policy callable.

    Builds the agent via ``hydra.utils.instantiate`` from the cfg embedded in
    the payload. EMA shadow params are restored explicitly for ACT (see
    workspace.py drift bullet 4 in CLAUDE.md). Pure CPU; no W&B.

    The caller should build ``env`` with cameras matching what
    :func:`peek_snapshot_cameras` returned so the actor's encoder gets the
    pixel keys it was trained with.
    """
    if not snapshot_path or not Path(snapshot_path).is_file():
        raise FileNotFoundError(
            f"snapshot_path={snapshot_path!r} not found. Phase-0 ACT snapshots are "
            "still pending GPU retrain — see CLAUDE.md."
        )

    import hydra
    import torch
    from omegaconf import DictConfig, OmegaConf

    payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)

    # CQN-AS snapshots (train_cqn_as.py) carry "agent_state" + "config" and a
    # vendored CQNASAgent whose act() takes split (rgb_obs, low_dim_obs) args —
    # incompatible with the RoboBase instantiate path below. Dispatch early.
    if "agent_state" in payload:
        return _load_cqn_as_snapshot_policy(
            payload, env, rollout_max_steps=rollout_max_steps
        )

    cfg = payload.get("cfg")
    if cfg is None:
        raise KeyError(
            f"snapshot at {snapshot_path} has no 'cfg' field — was it produced "
            "after the workspace.py drift fix? See CLAUDE.md."
        )
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    if int(cfg.get("frame_stack", 1)) != 1:
        raise NotImplementedError(
            "Snapshot policy adapter only supports frame_stack=1. "
            f"Snapshot's cfg.frame_stack={cfg.get('frame_stack')!r}. "
            "Extend _SnapshotPolicy.adapt_obs with a frame-stack deque first."
        )

    cameras: Tuple[str, ...] = ()
    resolution: Tuple[int, int] = (84, 84)
    if bool(cfg.get("pixels", False)):
        cameras = tuple(str(c) for c in cfg.env.get("cameras", []))
        shape = cfg.get("visual_observation_shape")
        if shape is not None:
            resolution = (int(shape[0]), int(shape[1]))

    # Phase 0 vs Phase 1 detection: was BodySLAMWrapper applied at training?
    bs = cfg.env.get("bodyslam") if "env" in cfg else None
    bs_mode = str(bs.get("mode", "off")) if bs is not None else "off"
    includes_human_pos = bs_mode in ("oracle", "noisy")

    # Synthesize the observation_space the agent was instantiated against at
    # training. The raw env (post-BodySLAMWrapper) has per-key 1-D obs and
    # rgb as (3, H, W); the agent expects ConcatDim+FrameStack output.
    synthesized_obs_space = _synthesize_snapshot_obs_space(
        env, cameras, resolution, includes_human_pos
    )

    method_cfg = cfg.method
    intrinsic_reward_module = None
    agent = hydra.utils.instantiate(
        method_cfg,
        device="cpu",
        observation_space=synthesized_obs_space,
        action_space=env.action_space,
        num_train_envs=0,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        intrinsic_reward_module=intrinsic_reward_module,
    )
    agent.load_state_dict(payload["agent"], strict=False)
    if "actor_ema" in payload and hasattr(agent, "actor"):
        actor_ema = payload["actor_ema"]
        if hasattr(agent.actor, "ema") and hasattr(agent.actor.ema, "shadow_params"):
            for p, sp in zip(agent.actor.ema.shadow_params, actor_ema):
                p.data.copy_(sp)
    agent.train(False)

    # B5.5: lift the action denormalization payload off the snapshot so the
    # SVF collection path matches RoboBase's RescaleFromTanhWithMinMax wrap.
    # Both fields are saved by the FYP3 robobase workspace.py drift; legacy
    # snapshots without them fall back to raw tanh + a one-shot warning.
    action_stats = payload.get("action_stats")
    if action_stats is not None:
        action_stats = {
            "min": np.asarray(action_stats["min"], dtype=np.float32),
            "max": np.asarray(action_stats["max"], dtype=np.float32),
        }
    min_max_margin = float(cfg.get("min_max_margin", 0.0))

    return _SnapshotPolicy(
        agent=agent,
        cameras=cameras,
        camera_resolution=resolution,
        includes_human_pos=includes_human_pos,
        action_stats=action_stats,
        min_max_margin=min_max_margin,
    )


class _CQNASSnapshotPolicy:
    """Roll out a CQN-AS snapshot as a ``policy(gym_obs) -> raw_action`` callable.

    CQN-AS snapshots differ from RoboBase ACT/DP ones: the vendored
    ``CQNASAgent`` is not a RoboBase Method, and ``act`` takes split
    ``(rgb_obs, low_dim_obs)`` arrays rather than an obs dict. This wrapper
    replicates :meth:`SafetyBiGymCQNAdapter._extract_obs` (state-key concat,
    optional ``human_pos_estimate`` injection, per-camera frame-stacked rgb)
    and ``_convert_action_to_raw`` (``[-1, 1]`` → env range) so the agent sees
    exactly its training-time inputs while the SVF collector keeps driving the
    plain gym env from :func:`_build_live_env`.

    ``includes_human_pos`` comes from the *snapshot's* trained bodyslam mode
    (so a bodyslam=off policy is fed low_dim without the 6-D human channel),
    independently of the env's own bodyslam mode (which is ``noisy`` so the
    dataset still records ``human_pos_estimate``). This mirrors the ACT path's
    ``includes_human_pos`` decoupling.

    Action execution **mirrors the deployment runner** (benchmark
    ``CQNASRunner.step``): re-plan every ``action_sequence`` steps and execute
    the open-loop sub-actions, or — when the snapshot's cfg sets
    ``temporal_ensemble`` — query every step and execute the exp-weighted
    ensemble blend. This is load-bearing: training the SVF on receding-horizon
    ``chunk[0]`` actions while deploying ensemble/open-loop actions makes every
    deployed action OOD for the critic (the 2026-06-01 ~89%-spurious-veto bug:
    `action_sequence=16`, `temporal_ensemble=true`, so raw `chunk[0]` ≠ the
    blended action executed at deploy). Frame-stack deques + chunk/ensemble
    state are reset per episode via :meth:`reset`, which
    :func:`rollout_episode` calls after ``env.reset()``.
    """

    def __init__(
        self,
        *,
        agent: Any,
        state_keys: Sequence[str],
        includes_human_pos: bool,
        camera_keys: Sequence[str],
        frame_stack: int,
        action_sequence: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        rgb_placeholder_shape: Optional[Sequence[int]] = None,
        temporal_ensemble: Any = None,
    ):
        self.agent = agent
        self._state_keys = tuple(state_keys)
        self._includes_human_pos = bool(includes_human_pos)
        self._camera_keys = tuple(camera_keys)
        self._frame_stack = int(frame_stack)
        self._action_sequence = int(action_sequence)
        # Mirror the deployment runner's execution mode (see class docstring).
        self._temporal_ensemble = temporal_ensemble
        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)
        self._rgb_placeholder_shape = (
            tuple(int(x) for x in rgb_placeholder_shape)
            if rgb_placeholder_shape is not None
            else None
        )
        self._low_dim_frames: Any = None
        self._rgb_frames: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        from collections import deque

        self._low_dim_frames = deque(maxlen=self._frame_stack)
        self._rgb_frames = {
            cam: deque(maxlen=self._frame_stack) for cam in self._camera_keys
        }
        # Per-episode action-execution state (mirrors CQNASRunner.reset).
        self._action_chunk = None
        self._episode_step = 0
        if self._temporal_ensemble is not None:
            self._temporal_ensemble.reset()

    def _stack_into(self, dq, frame: np.ndarray) -> None:
        # Match the adapter: prime the deque by repeating the first frame so a
        # fresh episode's stack is full from step 0; thereafter append normally.
        if len(dq) == 0:
            for _ in range(self._frame_stack):
                dq.append(frame)
        else:
            dq.append(frame)

    def _build_low_dim(self, gym_obs: Dict[str, np.ndarray]) -> np.ndarray:
        pieces = [
            np.asarray(gym_obs[k], dtype=np.float32).reshape(-1)
            for k in self._state_keys
        ]
        if self._includes_human_pos:
            pieces.append(
                np.asarray(gym_obs[HUMAN_POS_ESTIMATE_KEY], dtype=np.float32).reshape(-1)
            )
        low_dim = np.hstack(pieces).astype(np.float32)
        self._stack_into(self._low_dim_frames, low_dim)
        return np.concatenate(list(self._low_dim_frames), axis=0)

    def _build_rgb(self, gym_obs: Dict[str, np.ndarray]) -> np.ndarray:
        if not self._camera_keys:
            if self._rgb_placeholder_shape is None:
                raise ValueError(
                    "CQN-AS snapshot has no cameras and no rgb_obs_shape to "
                    "build a placeholder from."
                )
            return np.zeros(self._rgb_placeholder_shape, dtype=np.uint8)
        for cam in self._camera_keys:
            px = np.asarray(gym_obs[f"rgb_{cam}"]).astype(np.uint8, copy=False)
            # Bare bigym emits (H, W, 3); CQN-AS expects channel-first (C, H, W).
            if px.ndim == 3 and px.shape[-1] == 3:
                px = np.transpose(px, (2, 0, 1))
            self._stack_into(self._rgb_frames[cam], px)
        # (num_cameras, C * frame_stack, H, W) — mirrors adapter._extract_obs.
        return np.stack(
            [
                np.concatenate(list(self._rgb_frames[cam]), axis=0)
                for cam in self._camera_keys
            ],
            axis=0,
        )

    def __call__(self, gym_obs: Dict[str, np.ndarray]) -> np.ndarray:
        import torch

        low_dim_obs = self._build_low_dim(gym_obs)
        rgb_obs = self._build_rgb(gym_obs)
        # Mirror benchmark CQNASRunner.step EXACTLY so the collected action
        # distribution == the deployed one: re-plan every action_sequence steps
        # (or every step under temporal ensemble), then execute the ensemble
        # blend or the open-loop sub-action.
        if self._temporal_ensemble is not None or (
            self._episode_step % self._action_sequence == 0
        ):
            with torch.no_grad():
                # step is irrelevant under eval_mode (it only gates the stddev
                # schedule / exploration, both bypassed when eval_mode=True).
                raw = self.agent.act(rgb_obs, low_dim_obs, step=10**9, eval_mode=True)
            self._action_chunk = np.asarray(raw, dtype=np.float32).reshape(
                self._action_sequence, -1
            )
            if self._temporal_ensemble is not None:
                self._temporal_ensemble.register_action_sequence(self._action_chunk)
        if self._temporal_ensemble is not None:
            norm_action = np.asarray(
                self._temporal_ensemble.get_action(), dtype=np.float32
            )
        else:
            norm_action = self._action_chunk[
                self._episode_step % self._action_sequence
            ]
        self._episode_step += 1
        # [-1, 1] → raw env range (mirror adapter._convert_action_to_raw).
        scaled = (norm_action + 1.0) / 2.0
        raw_action = scaled * (self._action_high - self._action_low + 1e-8)
        raw_action = raw_action + self._action_low
        return raw_action.astype(np.float32, copy=False)


# Cap the temporal-ensemble history when the rollout length is unknown, so a
# long-horizon env's episode_length never triggers a multi-GB allocation. The
# blend is local (±action_sequence), so any size >= rollout length is exact.
_SAFE_ENS_CAP = 2048


def _load_cqn_as_snapshot_policy(
    payload: Dict[str, Any], env, *, rollout_max_steps: Optional[int] = None
) -> _CQNASSnapshotPolicy:
    """Build a CQN-AS policy from a ``train_cqn_as.py`` snapshot payload.

    The agent's input shapes (``rgb_obs_shape`` / ``low_dim_obs_shape`` /
    ``action_shape``) are baked into ``config.agent`` by ``make_agent`` at
    train time, so the agent is rebuilt with ``hydra.utils.instantiate`` and
    loaded via the vendored ``CQNASAgent.load_state_dict``.
    """
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from safety_bigym.agents.cqn_as.env_adapter import _DEFAULT_STATE_KEYS

    cfg = payload.get("config")
    if cfg is None:
        raise KeyError(
            "CQN-AS snapshot payload has 'agent_state' but no 'config' field."
        )
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    agent_cfg = cfg.get("agent")
    if agent_cfg is None or agent_cfg.get("rgb_obs_shape") is None:
        raise KeyError(
            "CQN-AS snapshot config.agent is missing or lacks the baked "
            "rgb_obs_shape/low_dim_obs_shape/action_shape that make_agent "
            "writes at train time — cannot rebuild the agent."
        )

    # Device portability: snapshots trained on the GPU box carry device="cuda";
    # fall back to CPU where CUDA is absent (local smoke) per the repo's
    # "works on this device or the GPU device" convention.
    import torch as _torch

    OmegaConf.set_struct(cfg, False)
    agent_cfg.device = "cuda" if _torch.cuda.is_available() else "cpu"

    agent = hydra.utils.instantiate(agent_cfg)
    agent.load_state_dict(payload["agent_state"])
    if hasattr(agent, "train"):
        agent.train(False)

    bs = cfg.env.get("bodyslam") if "env" in cfg else None
    bs_mode = str(bs.get("mode", "off")) if bs is not None else "off"
    includes_human_pos = bs_mode in ("oracle", "noisy")

    pixels_on = bool(cfg.get("pixels", False))
    camera_keys = tuple(str(c) for c in cfg.env.get("cameras", [])) if pixels_on else ()

    # --- Deployment-matched action de-normalisation (2026-06-01 fix) ---------
    # The CQN agent was TRAINED with the adapter's DEMO-derived action stats
    # (`extract_action_stats`), NOT env.action_space. The snapshot policy must
    # de-normalise the SAME way, or the actions it executes + stores in the SVF
    # dataset are mis-scaled and the runtime filter critic later sees OOD actions
    # (the benchmark Q-collapse / ~100%-intervention bug). Recover the exact
    # stats by building a throwaway adapter and replaying demos — the identical
    # path `train_cqn_as` and `benchmark.build_cqn_adapter` use. Loud fallback to
    # env.action_space (the OLD, buggy behaviour) so collection never crashes.
    from safety_bigym.agents.cqn_as import env_adapter as _ea  # noqa: E402

    action_low, action_high = env.action_space.low, env.action_space.high
    n_demos = int(cfg.get("num_demos", 0) or cfg.env.get("demos", 0) or 0)
    if n_demos > 0:
        try:
            _wrap = _ea.make(
                cfg, frame_stack=int(cfg.get("frame_stack", 1)),
                normalize_low_dim_obs=False,
            )
            _stats_adapter = _wrap._env
            _stats_adapter.get_demos(n_demos)  # populates demo-derived _action_stats
            action_low = np.asarray(_stats_adapter._action_stats["min"], dtype=np.float32)
            action_high = np.asarray(_stats_adapter._action_stats["max"], dtype=np.float32)
            logger.info(
                "Snapshot policy de-norm: DEMO-derived action stats (matches "
                "deployment); range=[%.3f..%.3f]", float(action_low.min()),
                float(action_high.max()),
            )
            try:
                _wrap.close()
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not derive demo action stats (%s); snapshot policy FALLS "
                "BACK to env.action_space de-norm — the SVF critic will see OOD "
                "actions. DO NOT trust a dataset collected via this fallback.", e,
            )
    else:
        logger.warning(
            "cfg has no num_demos; snapshot policy uses env.action_space de-norm "
            "(may mismatch deployment — see the 2026-06-01 de-norm fix)."
        )

    # Match the deployment runner's action-execution mode (open-loop chunks +
    # optional temporal ensemble) so collected actions == deployed actions.
    # Without this, the SVF trains on receding-horizon chunk[0] but deploys
    # ensemble/open-loop actions -> OOD -> ~100% spurious veto (2026-06-01 bug).
    temporal_ensemble = None
    if bool(cfg.get("temporal_ensemble", False)):
        from dm_env import specs as _specs

        from safety_bigym.agents.cqn_as import utils as _cqn_utils

        _act_dim = int(np.asarray(action_low).shape[0])
        _seq = int(cfg.get("action_sequence", 1))
        _ep_len = int(cfg.env.get("episode_length")) if "env" in cfg else None
        if _ep_len is None:
            raise KeyError(
                "temporal_ensemble=true but cfg.env.episode_length is missing — "
                "cannot size TemporalEnsembleControl to match deployment."
            )
        # The ensemble blend at step t only reads the last `action_sequence`
        # chunks (rows t-seq+1..t at column t), so the result is INDEPENDENT of
        # the history's total length. Size to the actual rollout cap, not
        # episode_length: on long-horizon envs (saucepan episode_length=25000)
        # the full [L, L+seq, dim] array is ~40 GB and would OOM, while the
        # blended actions are byte-identical at any size >= rollout length.
        if rollout_max_steps is not None:
            _ens_len = min(_ep_len, int(rollout_max_steps) + _seq + 2)
        else:
            _ens_len = min(_ep_len, _SAFE_ENS_CAP)
        _ens_len = max(_ens_len, _seq + 2)
        _act_spec = _specs.BoundedArray(
            shape=(_act_dim,), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
        temporal_ensemble = _cqn_utils.TemporalEnsembleControl(_ens_len, _act_spec, _seq)
        logger.info(
            "Snapshot policy uses TemporalEnsembleControl (history_len=%d for "
            "rollout_max_steps=%s, action_sequence=%d) to match deployment.",
            _ens_len, rollout_max_steps, _seq,
        )

    return _CQNASSnapshotPolicy(
        agent=agent,
        state_keys=_DEFAULT_STATE_KEYS,
        includes_human_pos=includes_human_pos,
        camera_keys=camera_keys,
        frame_stack=int(cfg.get("frame_stack", 1)),
        action_sequence=int(cfg.get("action_sequence", 1)),
        action_low=action_low,
        action_high=action_high,
        rgb_placeholder_shape=tuple(agent_cfg.get("rgb_obs_shape")),
        temporal_ensemble=temporal_ensemble,
    )


# ---------- main orchestration ------------------------------------------------


def _smoke_shard_name(plan: CollectionPlan) -> Optional[str]:
    if plan.episodes_per_cell == 2 and plan.max_steps == 50:
        return "_smoke_shard"
    return None


def _collect_live_env_source(
    plan: CollectionPlan,
    *,
    source: str,
    rng: np.random.Generator,
    spec,
    writer,
    use_pfl: bool,
) -> Tuple[int, int]:
    """Random / snapshot collection — both use a live env + a policy callable.

    For ``source="snapshot"`` the per-task path is resolved via
    :func:`safety_bigym.filters.snapshots.resolve_snapshot`; tasks whose
    SNAPSHOTS entry is ``None`` are skipped with a warning.
    """
    from safety_bigym.filters.feature_extractor import CriticFeatureSpec
    from safety_bigym.filters.dataset import TransitionShardWriter
    from safety_bigym.filters.snapshots import resolve_snapshot

    source_code = SOURCE_CODES[source]
    shard_idx = 0
    total = 0

    for task_key in plan.tasks:
        task_id = TASK_REGISTRY[task_key][1]

        # Snapshot path is per-task — resolve before doing any heavy env setup.
        snapshot_path: Optional[Path] = None
        snapshot_cameras: Tuple[str, ...] = ()
        snapshot_resolution: Tuple[int, int] = (84, 84)
        bodyslam_mode = plan.bodyslam_mode
        if source == "snapshot":
            snapshot_path = resolve_snapshot(
                task_key, overrides=plan.snapshot_overrides
            )
            if snapshot_path is None:
                logger.warning(
                    f"No snapshot configured for task {task_key!r} "
                    "(SNAPSHOTS entry is None and no --snapshot-override given). "
                    "Skipping snapshot source for this task."
                )
                continue
            # Peek at the snapshot's cfg so the env we build matches the
            # actor's *pixel* training format (same cameras + resolution).
            # The actor's low_dim_state schema (Phase 0 off vs Phase 1
            # oracle/noisy) is handled separately by _SnapshotPolicy.adapt_obs
            # and the synthesized observation_space inside load_snapshot_policy.
            snapshot_cameras, snapshot_resolution = peek_snapshot_cameras(snapshot_path)
            snap_bs = peek_snapshot_bodyslam_mode(snapshot_path)
            # Env wrapping is governed by plan.bodyslam_mode regardless of
            # what the snapshot was trained with — the SVF dataset must
            # always carry human_pos_estimate so the critic can learn the
            # SSM signal. Phase 0 snapshots (bodyslam=off) still work as
            # action samplers because adapt_obs strips the human channel
            # before feeding the actor.
            logger.info(
                f"Snapshot at {snapshot_path}: trained bodyslam={snap_bs}, "
                f"cameras={list(snapshot_cameras) or 'none'} "
                f"@ {snapshot_resolution[0]}x{snapshot_resolution[1]}; "
                f"env wrapping uses plan.bodyslam_mode={bodyslam_mode!r}."
            )

        for disruption in plan.disruptions:
            logger.info(
                f"Collecting source={source} task={task_key} disruption={disruption} "
                f"({plan.episodes_per_cell} episodes, max {plan.max_steps} steps)"
            )
            env = _build_live_env(
                task_key, disruption, bodyslam_mode, plan.motion_clips,
                cameras=snapshot_cameras,
                camera_resolution=snapshot_resolution,
                human_model=plan.human_model,
            )

            if source == "random":
                policy = random_policy(env, rng)
            elif source == "snapshot":
                # rollout_max_steps sizes the CQN-AS temporal-ensemble history to
                # the rollout cap (not episode_length) — see _load_cqn_as_snapshot_policy.
                policy = load_snapshot_policy(
                    snapshot_path, env, rollout_max_steps=plan.max_steps
                )
            else:
                raise ValueError(f"_collect_live_env_source got source={source!r}")

            # Spec is built off the first env we construct so the obs key set
            # is consistent across all sources.
            if spec[0] is None:
                spec[0] = CriticFeatureSpec.from_spaces(
                    env.observation_space, env.action_space
                )
                writer[0] = TransitionShardWriter(spec[0], plan.output_dir)
                logger.info(
                    f"Critic feature spec frozen: input_dim={spec[0].input_dim} "
                    f"(obs_keys={spec[0].obs_keys})"
                )

            for _ep in range(plan.episodes_per_cell):
                payload = rollout_episode(
                    env,
                    policy,
                    max_steps=plan.max_steps,
                    obs_keys=spec[0].obs_keys,
                    use_pfl=use_pfl,
                    proximity_threshold=plan.proximity_threshold,
                )
                if payload is None:
                    continue
                n = len(payload["action"])
                name = _smoke_shard_name(plan) or (
                    f"{source}__{task_key}__{disruption}__{shard_idx:04d}"
                )
                writer[0].write_shard(
                    name=name,
                    obs=payload["obs"],
                    action=payload["action"],
                    next_obs=payload["next_obs"],
                    r_safe=payload["r_safe"],
                    done=payload["done"],
                    ssm_margin=payload["ssm_margin"],
                    min_separation=payload["min_separation"],
                    pfl_force_ratio=payload["pfl_force_ratio"],
                    source=np.full(n, source_code, dtype=np.uint8),
                    task_id=np.full(n, task_id, dtype=np.uint8),
                )
                shard_idx += 1
                total += n

    return shard_idx, total


def _collect_demo_source(
    plan: CollectionPlan,
    *,
    spec,
    writer,
) -> Tuple[int, int]:
    """Load BiGym demos once per task and lay them out as transitions.

    Demos do not require a live env or any disruption iteration — they are
    pre-recorded action sequences. ``demos_per_task`` controls the demo count;
    each demo becomes a single shard.
    """
    from safety_bigym.filters.feature_extractor import CriticFeatureSpec
    from safety_bigym.filters.dataset import TransitionShardWriter

    if spec[0] is None:
        # Demos can't define the spec on their own (they don't expose
        # action_space.low/high in a useful way for sampling). Require either
        # demo to come AFTER a live source, or — for demo-only collection —
        # build a probe env once.
        probe = _build_live_env(
            plan.tasks[0],
            plan.disruptions[0] if plan.disruptions else "coworker_train",
            plan.bodyslam_mode,
            plan.motion_clips,
            human_model=plan.human_model,
        )
        spec[0] = CriticFeatureSpec.from_spaces(
            probe.observation_space, probe.action_space
        )
        writer[0] = TransitionShardWriter(spec[0], plan.output_dir)
        probe.close() if hasattr(probe, "close") else None
        logger.info(
            f"Critic feature spec frozen via demo probe: input_dim={spec[0].input_dim} "
            f"(obs_keys={spec[0].obs_keys})"
        )

    shard_idx = 0
    total = 0
    source_code = SOURCE_CODES["demo"]

    for task_key in plan.tasks:
        task_id = TASK_REGISTRY[task_key][1]
        logger.info(
            f"Collecting source=demo task={task_key} "
            f"(up to {plan.demos_per_task} demos)"
        )
        amass_provider = _make_amass_provider(plan.motion_clips, plan.seed ^ 0xD)
        try:
            demos = fetch_demos(task_key, plan.demos_per_task)
        except Exception as e:  # noqa: BLE001 — surface DemoStore failures clearly
            logger.error(f"DemoStore fetch failed for {task_key}: {e}")
            raise
        if not demos:
            logger.warning(
                f"No demos available for {task_key}; skipping demo source for this task."
            )
            continue

        for demo_idx, demo in enumerate(demos):
            payload = demo_episode_to_transitions(
                demo.timesteps,
                spec_obs_keys=spec[0].obs_keys,
                amass_provider=amass_provider,
            )
            if payload is None:
                continue
            n = len(payload["action"])
            name = _smoke_shard_name(plan) or f"demo__{task_key}__{demo_idx:04d}"
            writer[0].write_shard(
                name=name,
                obs=payload["obs"],
                action=payload["action"],
                next_obs=payload["next_obs"],
                r_safe=payload["r_safe"],
                done=payload["done"],
                ssm_margin=payload["ssm_margin"],
                min_separation=payload["min_separation"],
                pfl_force_ratio=payload["pfl_force_ratio"],
                source=np.full(n, source_code, dtype=np.uint8),
                task_id=np.full(n, task_id, dtype=np.uint8),
            )
            shard_idx += 1
            total += n

    return shard_idx, total


def run_collection(plan: CollectionPlan, *, use_pfl: bool = False) -> Path:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(plan.seed)

    # Mutable wrappers so the per-source helpers can lazy-init the spec/writer.
    spec_holder: List[Any] = [None]
    writer_holder: List[Any] = [None]
    total = 0

    for source in plan.sources:
        if source == "demo":
            _, n = _collect_demo_source(plan, spec=spec_holder, writer=writer_holder)
        elif source in ("random", "snapshot"):
            _, n = _collect_live_env_source(
                plan,
                source=source,
                rng=rng,
                spec=spec_holder,
                writer=writer_holder,
                use_pfl=use_pfl,
            )
        else:
            raise ValueError(f"Unknown source {source!r}; known: {sorted(SOURCE_CODES)}")
        total += n

    if total == 0:
        raise RuntimeError(
            "Collection produced 0 transitions; check task / disruption / "
            "AMASS clip availability."
        )
    logger.info(f"Wrote {total} transitions to {plan.output_dir}")
    return plan.output_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="≤200-transition smoke run.")
    p.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_CODES),
        help="Repeatable. Defaults to ['random'].",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_REGISTRY)[:1],
        choices=sorted(TASK_REGISTRY),
    )
    p.add_argument(
        "--disruption-space",
        choices=DISRUPTION_SPACE_CHOICES,
        default="coworker_train",
        help=(
            "coworker_train (default): single cell using make_coworker_train_space() "
            "— 5 continuous parameter axes in moderate bands. "
            "coworker_eval: single cell using make_coworker_eval_space() — wider "
            "bands that strictly contain the train ranges. "
            "legacy_multi: iterate over --disruptions string list (pre-Phase-0.5 "
            "5-disruption mixture). Retained as an escape hatch."
        ),
    )
    p.add_argument(
        "--disruptions",
        nargs="+",
        default=list(DEFAULT_DISRUPTIONS),
        help="Only used when --disruption-space=legacy_multi.",
    )
    p.add_argument("--episodes-per-cell", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--demos-per-task", type=int, default=30)
    p.add_argument("--bodyslam-mode", choices=("oracle", "noisy"), default="oracle")
    p.add_argument(
        "--human-model",
        choices=("smplh", "g1"),
        default="g1",
        help=(
            "Coworker embodiment. 'g1' (default) is AMASS-free; 'smplh' "
            "requires AMASS_DATA_DIR and replays motion clips."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "svf_v1",
    )
    p.add_argument(
        "--snapshot-override",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "Override a snapshot path for one task; takes precedence over the "
            "SNAPSHOTS dict in safety_bigym/filters/snapshots.py. Repeatable."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--proximity-threshold",
        type=float,
        default=0.10,
        help=(
            "Geometric near-contact bar (metres) used by label_transition. "
            "Any human-joint / robot-link pair closer than this counts as a "
            "safety violation. Default 0.10 m. Surfaced for sweep; see "
            "safety_bigym/filters/labeling.py for the rationale."
        ),
    )
    p.add_argument("--use-pfl", action="store_true",
                   help="Set once the PFL contact-detection bug is fixed.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    snapshot_overrides = _parse_snapshot_overrides(args.snapshot_override)

    if args.smoke:
        plan = CollectionPlan.smoke(args.output_dir)
    else:
        if args.disruption_space == "legacy_multi":
            disruptions = tuple(args.disruptions)
        else:
            # coworker_train / coworker_eval: single cell labelled by the space
            # name; _build_live_env dispatches to the factory on this label.
            disruptions = (args.disruption_space,)
        plan = CollectionPlan(
            sources=tuple(args.source or ("random",)),
            tasks=tuple(args.tasks),
            disruptions=disruptions,
            episodes_per_cell=args.episodes_per_cell,
            max_steps=args.max_steps,
            bodyslam_mode=args.bodyslam_mode,
            output_dir=args.output_dir,
            snapshot_overrides=snapshot_overrides,
            seed=args.seed,
            demos_per_task=args.demos_per_task,
            proximity_threshold=args.proximity_threshold,
            human_model=args.human_model,
        )

    run_collection(plan, use_pfl=args.use_pfl)
    return 0


def _parse_snapshot_overrides(raw: Sequence[str]) -> Dict[str, str]:
    """Turn ``--snapshot-override TASK=PATH`` flags into a mapping."""
    out: Dict[str, str] = {}
    for entry in raw:
        if "=" not in entry:
            raise SystemExit(
                f"--snapshot-override expects TASK=PATH; got {entry!r}"
            )
        task, _, path = entry.partition("=")
        if not task or not path:
            raise SystemExit(
                f"--snapshot-override expects TASK=PATH; got {entry!r}"
            )
        out[task] = path
    return out


if __name__ == "__main__":
    raise SystemExit(main())
