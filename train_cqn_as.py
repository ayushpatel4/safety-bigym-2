"""CQN-AS training entrypoint for SafetyBiGym.

Modelled on CQN-AS/train_cqn_as_bigym.py but routes env construction
through safety_bigym.agents.cqn_as.env_adapter so all safety wrappers
compose. Handles num_demos=0 cleanly (skips the demo replay buffer
entirely) so the A6 smoke gate can exercise composition without
the demo pipeline (deferred to a follow-up).

Usage:
    export AMASS_DATA_DIR=/path/to/CMU/CMU
    export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0  # headless GPU box
    python train_cqn_as.py \\
        env=safety_bigym/dishwasher_close \\
        disruption=coworker_train \\
        bodyslam=oracle \\
        num_train_frames=20000 \\
        wandb.use=true \\
        wandb.name=cqn_as_dishwasher_close_smoke
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Default to EGL on headless boxes; user can override.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import hydra
import numpy as np
import torch
from dm_env import specs
from omegaconf import DictConfig, OmegaConf

from safety_bigym.agents.cqn_as import env_adapter, utils
from safety_bigym.agents.cqn_as.eval_video import render_frame, write_eval_video
from safety_bigym.agents.cqn_as.replay_buffer import (
    ReplayBufferStorage,
    make_replay_loader,
)

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


def make_agent(rgb_obs_spec, low_dim_obs_spec, action_spec, action_sequence, cfg):
    cfg.rgb_obs_shape = list(rgb_obs_spec.shape)
    cfg.low_dim_obs_shape = list(low_dim_obs_spec.shape)
    cfg.action_shape = [action_sequence, *action_spec.shape]
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg: DictConfig):
        self.work_dir = Path.cwd()
        logger.info(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._setup_env()
        self._setup_replay()
        self._setup_wandb()

        self.agent = make_agent(
            self.train_env.rgb_observation_spec(),
            self.train_env.low_dim_observation_spec(),
            self.train_env.action_spec(),
            self.cfg.action_sequence,
            self.cfg.agent,
        )

        self.timer = utils.Timer()
        self._update_step = 0
        self._global_step = 0
        self._global_episode = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_env(self) -> None:
        self.train_env = env_adapter.make(
            self.cfg,
            frame_stack=self.cfg.frame_stack,
            normalize_low_dim_obs=False,
        )
        if self.cfg.temporal_ensemble:
            self.train_temporal_ensemble = utils.TemporalEnsembleControl(
                self.cfg.env.episode_length,
                self.train_env.action_spec(),
                self.cfg.action_sequence,
            )
            self.eval_temporal_ensemble = utils.TemporalEnsembleControl(
                self.cfg.env.episode_length,
                self.train_env.action_spec(),
                self.cfg.action_sequence,
            )

    def _setup_replay(self) -> None:
        data_specs = (
            self.train_env.rgb_raw_observation_spec(),
            self.train_env.low_dim_raw_observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
            specs.Array((1,), np.float32, "demo"),
            # Phase 3 P3.0c: per-step cost c_t carried into the episode shards.
            # ReplayBufferStorage.add() reads `time_step["cost"]` via the
            # NamedTuple __getitem__ shim; env_adapter populates it from
            # info["safety"] each env-step.
            specs.Array((1,), np.float32, "cost"),
        )
        self.replay_storage = ReplayBufferStorage(
            data_specs, self.work_dir / "buffer", self.cfg.use_relabeling
        )
        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.action_sequence,
            self.cfg.frame_stack,
            fill_action="last_action",
        )
        # Demo plumbing only when num_demos>0; smoke gate runs with 0.
        self._demos_enabled = int(self.cfg.num_demos) > 0
        if self._demos_enabled:
            self.demo_replay_storage = ReplayBufferStorage(
                data_specs,
                self.work_dir / "demo_buffer",
                self.cfg.use_relabeling,
                is_demo_buffer=True,
            )
            self.demo_replay_loader = make_replay_loader(
                self.work_dir / "demo_buffer",
                self.cfg.replay_buffer_size,
                self.cfg.demo_batch_size,
                self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot,
                self.cfg.nstep,
                self.cfg.discount,
                self.cfg.action_sequence,
                self.cfg.frame_stack,
                fill_action="last_action",
            )
        else:
            self.demo_replay_storage = None
            self.demo_replay_loader = None
        self._replay_iter = None

    def _setup_wandb(self) -> None:
        self._wandb_run = None
        wb_cfg = self.cfg.get("wandb", None)
        if wb_cfg is None or not bool(wb_cfg.get("use", False)):
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb requested but not installed; skipping.")
            return
        self._wandb_run = wandb.init(
            project=str(wb_cfg.get("project", "safety-critic")),
            entity=wb_cfg.get("entity"),
            name=str(wb_cfg.get("name", "cqn_as_run")),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            dir=str(self.work_dir),
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, metrics, step: int, ty: str = "train") -> None:
        # `metrics` may be a TensorDict (returned by agent.update()) or a
        # plain dict. Three things need handling:
        #  1. TensorDict refuses bool conversion — can't use ``if not metrics``.
        #  2. ``len(TensorDict)`` returns the first batch-size dim, not the
        #     number of keys. ``agent.update()`` returns a TD built from 0-d
        #     ``.detach()`` tensors, so its ``batch_size == torch.Size([])``
        #     and ``len(td) == 0`` even when 4+ keys are present. Using
        #     ``len(metrics) == 0`` as an emptiness check silently swallows
        #     every per-update train log — that's the regression flagged in
        #     ``docs/IMPLEMENTATION_STATUS.md`` 2026-05-23 notes
        #     (``grep q_critic_loss`` returning nothing). Check ``items()``.
        #  3. ``TensorDict.items()`` can be a single-use generator (the dict
        #     comprehension below would exhaust it, leaving the format-
        #     string join silent). Materialise items into a list once.
        if metrics is None:
            return
        try:
            items = list(metrics.items())
        except Exception:
            return
        if not items:
            return
        materialised = []
        for k, v in items:
            if hasattr(v, "item"):  # 0-d tensor → python scalar
                try:
                    v = v.item()
                except (ValueError, RuntimeError):
                    pass  # non-scalar tensor; format() will str() it
            materialised.append((k, v))
        prefixed = {f"{ty}/{k}": v for k, v in materialised}
        if self._wandb_run is not None:
            self._wandb_run.log(prefixed, step=step)
        parts = []
        for k, v in materialised:
            try:
                parts.append(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                )
            except (TypeError, ValueError):
                parts.append(f"{k}={v!r}")
        logger.info(f"[{ty}] step={step} " + " ".join(parts))

    def _safety_payload(self, info: dict) -> dict:
        """Extract per-step + episode-end safety metrics from env info."""
        out: dict = {}
        step_safety = info.get("safety") if info else None
        if step_safety is not None:
            for key in ("ssm_margin", "pfl_force_ratio",
                        "ssm_violation", "pfl_violation"):
                if key in step_safety:
                    val = step_safety[key]
                    if isinstance(val, bool):
                        val = float(val)
                    out[f"safety/{key}"] = val
        ep_safety = info.get("episode_safety") if info else None
        if ep_safety is not None:
            for key, val in ep_safety.items():
                if isinstance(val, (int, float, bool)):
                    out[f"episode_safety/{key}"] = float(val)
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def global_episode(self) -> int:
        return self._global_episode

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            base = iter(self.replay_loader)
            if self._demos_enabled:
                demo = iter(self.demo_replay_loader)
                self._replay_iter = utils.DemoMergedIterator(base, demo)
            else:
                self._replay_iter = base
        return self._replay_iter

    # ------------------------------------------------------------------
    # Demos
    # ------------------------------------------------------------------

    def load_demos(self) -> None:
        if not self._demos_enabled:
            logger.warning(
                "num_demos=0 — running without demonstrations. CQN-AS is "
                "designed for demo-driven RL; this mode is intended for the "
                "A6 smoke gate only."
            )
            return
        demos = self.train_env.get_demos(self.cfg.num_demos)
        for demo in demos:
            for time_step in demo:
                self.replay_storage.add(time_step)
                self.demo_replay_storage.add(time_step)
        logger.info(
            f"Loaded {len(demos)} demos; replay size now {len(self.replay_storage)}"
        )

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(
            self.cfg.num_seed_frames, self.cfg.action_repeat
        )
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )
        snapshot_every_step = utils.Every(
            self.cfg.snapshot_every_frames, self.cfg.action_repeat
        )
        do_eval = False

        time_step = self.train_env.reset()
        if self.cfg.temporal_ensemble:
            self.train_temporal_ensemble.reset()
        self.replay_storage.add(time_step)
        if self._demos_enabled:
            self.demo_replay_storage.add(time_step)

        episode_step = 0
        episode_reward = 0.0
        action = None
        metrics: dict = {}

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                ep_safety_metrics = self._safety_payload(time_step.info or {})
                if ep_safety_metrics:
                    self._log(ep_safety_metrics, self.global_step, ty="episode")
                self._log(
                    {
                        "episode_reward": episode_reward,
                        "episode_length": episode_step,
                        "episode": self._global_episode,
                        "buffer_size": len(self.replay_storage),
                    },
                    self.global_step,
                    ty="train",
                )
                if do_eval:
                    self.eval()
                    do_eval = False

                time_step = self.train_env.reset()
                if self.cfg.temporal_ensemble:
                    self.train_temporal_ensemble.reset()
                self.replay_storage.add(time_step)
                if self._demos_enabled:
                    self.demo_replay_storage.add(time_step)
                episode_step = 0
                episode_reward = 0.0

            if (
                self.global_step >= self.cfg.eval_every_frames
                and eval_every_step(self.global_step)
            ):
                do_eval = True

            if (
                self.cfg.temporal_ensemble
                or episode_step % self.cfg.action_sequence == 0
            ):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    raw_action = self.agent.act(
                        time_step.rgb_obs,
                        time_step.low_dim_obs,
                        self.global_step,
                        eval_mode=True,
                    )
                action = raw_action.reshape([self.cfg.action_sequence, -1])
                if self.cfg.temporal_ensemble:
                    self.train_temporal_ensemble.register_action_sequence(action)

            # Worker-aware update gate. The CQN-AS replay loader stripes
            # episode files by ``eps_idx % num_workers`` (replay_buffer.py
            # _try_fetch). The DataLoader rotates across workers, so a
            # worker that hasn't seen its first eligible episode yet raises
            # IndexError on sample. Upstream CQN-AS never hits this because
            # demos pre-fill every worker; in the A6 smoke gate we run with
            # num_demos=0 and have to wait until each worker is guaranteed
            # at least one episode (i.e. global_episode >= num_workers).
            num_replay_workers = max(1, int(self.cfg.replay_buffer_num_workers))
            if (
                not seed_until_step(self.global_step)
                and self.global_step % self.cfg.agent.update_every_steps == 0
                and self._global_episode >= num_replay_workers
            ):
                for _ in range(self.cfg.num_update_steps):
                    batch = next(self.replay_iter)
                    batch = utils.to_torch_pixel_tensor_dict(batch, self.device)
                    # Guard: a non-finite tensor anywhere in the batch propagates
                    # NaN into the network -> NaN action / projection target ->
                    # floor(NaN) garbage index -> index_add_/gather dies with an
                    # opaque async CUDA device-side assert (dstIndex < dstAddDimSize)
                    # that mis-points at a later op (e.g. cumsum). Fail here instead,
                    # naming the offending field. Covers observations (low_dim_obs
                    # carries the demo-injected human_pos_estimate) + reward/action.
                    for _k in (
                        "reward", "discount", "action", "cost",
                        "low_dim_obs", "next_low_dim_obs",
                    ):
                        if _k in batch and not torch.isfinite(batch[_k]).all():
                            bad = batch[_k]
                            n_bad = int((~torch.isfinite(bad)).sum().item())
                            raise ValueError(
                                f"non-finite values in batch['{_k}'] before "
                                f"agent.update ({n_bad} of {bad.numel()} bad; "
                                f"finite range [{bad[torch.isfinite(bad)].min().item() if torch.isfinite(bad).any() else float('nan')}, "
                                f"{bad[torch.isfinite(bad)].max().item() if torch.isfinite(bad).any() else float('nan')}]); "
                                f"this would crash the C51 critic. Likely a "
                                f"malformed demo (e.g. human_pos_estimate) or env "
                                f"signal — check the demo set / obs pipeline."
                            )
                    metrics = self.agent.update(batch)
                    self._update_step += 1
                    self.agent.update_target_critic(self._update_step)
                # `metrics` is a TensorDict; truthiness raises. Defer the
                # filled-vs-empty check to _log (which handles both shapes).
                self._log(metrics, self.global_step, ty="train")

            if self.cfg.temporal_ensemble:
                sub_action = self.train_temporal_ensemble.get_action()
            else:
                sub_action = action[episode_step % self.cfg.action_sequence]
            sub_action = self.agent.add_noise_to_action(sub_action, self.global_step)
            time_step = self.train_env.step(sub_action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            if self._demos_enabled:
                self.demo_replay_storage.add(time_step)

            # Per-step safety payload (gate A6.3 — confirms info["safety"]
            # arrives per env step, not per K-step chunk).
            step_safety = self._safety_payload(time_step.info or {})
            if step_safety and self.global_step % 50 == 0:
                self._log(step_safety, self.global_step, ty="safety")

            episode_step += 1
            self._global_step += 1

            # Snapshot cadence: fire every snapshot_every_frames steps,
            # independent of episode boundaries. Previously this check lived
            # inside `if time_step.last():` which only triggered when an
            # episode end coincidentally aligned with a multiple of
            # snapshot_every_frames — for COWORKER train rollouts on
            # saucepan_to_hob (episodes ~150 steps, terminated stochastically
            # by violations), that alignment essentially never happens and
            # 200k-step runs landed zero snapshots on disk.
            if self.cfg.save_snapshot and snapshot_every_step(self.global_step):
                self.save_snapshot()

        # Final-state snapshot when the loop exits at num_train_frames.
        # Without this, the trailing partial period (last snapshot landed at
        # 190000, train_until_step exits at 200000) is unrecoverable —
        # the eval-curve peak is usually near the end, so this is the most
        # load-bearing single checkpoint of the run.
        if self.cfg.save_snapshot:
            self.save_snapshot()

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval(self) -> None:
        step, episode, total_reward = 0, 0, 0.0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        # cfg.save_video gates per-eval-cycle video recording. Only the first
        # eval episode is captured per cycle (disk is cheap but not infinite,
        # and the second/third episode look very similar early in training).
        # Rendering goes through self.train_env.render() which delegates down
        # through the wrapper stack to SafetyBiGymEnv's MuJoCo renderer.
        record_video = bool(self.cfg.get("save_video", False))
        frames: list = []
        video_dir = self.work_dir / "eval_videos"

        while eval_until_episode(episode):
            episode_step = 0
            time_step = self.train_env.reset()
            if self.cfg.temporal_ensemble:
                self.eval_temporal_ensemble.reset()
            action = None
            if record_video and episode == 0:
                frame = render_frame(self.train_env, global_step=self.global_step)
                if frame is not None:
                    frames.append(frame)
            while not time_step.last():
                if (
                    self.cfg.temporal_ensemble
                    or episode_step % self.cfg.action_sequence == 0
                ):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        raw_action = self.agent.act(
                            time_step.rgb_obs,
                            time_step.low_dim_obs,
                            self.global_step,
                            eval_mode=True,
                        )
                    action = raw_action.reshape([self.cfg.action_sequence, -1])
                    if self.cfg.temporal_ensemble:
                        self.eval_temporal_ensemble.register_action_sequence(action)
                if self.cfg.temporal_ensemble:
                    sub_action = self.eval_temporal_ensemble.get_action()
                else:
                    sub_action = action[episode_step % self.cfg.action_sequence]
                time_step = self.train_env.step(sub_action)
                total_reward += time_step.reward
                step += 1
                episode_step += 1
                if record_video and episode == 0:
                    frame = render_frame(self.train_env, global_step=self.global_step)
                    if frame is not None:
                        frames.append(frame)
            episode += 1

        if record_video and frames:
            write_eval_video(
                video_dir,
                frames,
                global_step=self.global_step,
                wandb_run=self._wandb_run,
            )

        self._log(
            {
                "episode_reward": total_reward / max(episode, 1),
                "episode_length": step / max(episode, 1),
                "episode": self._global_episode,
            },
            self.global_step,
            ty="eval",
        )


    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def save_snapshot(self) -> None:
        snapshot = self.work_dir / f"snapshot_{self.global_step}.pt"
        payload = {
            "agent_state": self.agent.state_dict()
            if hasattr(self.agent, "state_dict") else None,
            "step": self._global_step,
            "episode": self._global_episode,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        with snapshot.open("wb") as f:
            torch.save(payload, f)
        logger.info(f"saved snapshot: {snapshot}")

    def load_snapshot(self, path) -> None:
        """Eager-load a saved agent state into ``self.agent``.

        Called from ``main`` before ``train()`` when the user passes
        ``+snapshot_path=...`` on the Hydra CLI. The snapshot must have
        been produced by a run with a compatible model architecture
        (same bodyslam mode, same task, same agent config). Mismatches
        surface as a strict-load KeyError or shape RuntimeError, which
        is the right failure mode — we'd rather fail loudly than
        silently run with mostly-random weights.

        ``weights_only=False`` is required because snapshot payloads
        also carry the run config (an OmegaConf DictConfig); torch
        ≥2.6's default safe-globals list doesn't include it.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"snapshot_path {path} not found")
        logger.info(f"loading snapshot: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        agent_state = payload.get("agent_state")
        if agent_state is None:
            raise ValueError(
                f"snapshot {path} has no 'agent_state' field "
                f"(keys: {sorted(payload.keys())})"
            )
        # The vendored CQNASAgent isn't an nn.Module — it owns sub-modules
        # (encoder, critic, target_critic, optimizers). Its state_dict()
        # returns a dict-of-dicts; load_state_dict needs the same shape.
        self.agent.load_state_dict(agent_state)
        loaded_step = int(payload.get("step", 0))
        loaded_ep = int(payload.get("episode", 0))
        logger.info(
            f"snapshot loaded: trained for {loaded_step} steps / "
            f"{loaded_ep} episodes"
        )


@hydra.main(version_base=None, config_path="cfgs", config_name="cqn_as_config")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    # Eval-only mode: load a saved policy, run eval() once, exit.
    # phase1_reward_pilot_cqn_as.py --eval uses this path with
    # num_train_frames=0 and +snapshot_path=<peak snapshot>.
    snapshot_path = cfg.get("snapshot_path") if hasattr(cfg, "get") else None
    if snapshot_path is not None:
        workspace.load_snapshot(snapshot_path)
    if int(cfg.num_train_frames) <= 0:
        if snapshot_path is None:
            logger.warning(
                "num_train_frames<=0 with no snapshot_path — agent is "
                "freshly initialised; eval results will be ~random."
            )
        workspace.eval()
        return
    workspace.load_demos()
    workspace.train()


if __name__ == "__main__":
    main()
