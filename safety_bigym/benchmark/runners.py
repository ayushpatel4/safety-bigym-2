"""Rollout runners — a uniform per-step interface across the random / ACT / CQN-AS paths.

Every runner exposes ``reset(seed) -> info`` and ``step() -> StepRecord`` (it owns its own
policy + env and pulls its own action), plus ``intervention_count`` / ``filter_step_count``
(0 when unfiltered). :func:`run_episode` then drives any runner identically into one
:class:`EpisodeRecord`, so the aggregation code never branches on policy kind.

``apply_veto`` is the pure SVF-veto kernel (no torch/mujoco of its own) used by the CQN-AS
in-loop filter; factoring it out lets it be unit-tested with a stub critic + identity
transforms, de-risking the locally-untestable CQN-AS path.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Protocol, Tuple, runtime_checkable

import numpy as np

from safety_bigym.benchmark.records import EpisodeRecord, StepRecord

__all__ = [
    "RolloutRunner",
    "GymRunner",
    "CQNASRunner",
    "run_episode",
    "apply_veto",
    "make_random_policy",
    "build_cell_runner",
]


@runtime_checkable
class RolloutRunner(Protocol):
    def reset(self, seed: int) -> Dict[str, Any]: ...
    def step(self) -> StepRecord: ...
    @property
    def intervention_count(self) -> int: ...
    @property
    def filter_step_count(self) -> int: ...
    def close(self) -> None: ...


def make_random_policy(action_space, seed: int = 0) -> Callable[[Mapping[str, Any]], np.ndarray]:
    """Uniform sampler over a Box action space (deterministic given ``seed``)."""
    rng = np.random.default_rng(seed)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    def _act(_obs):
        return rng.uniform(low=low, high=high).astype(np.float32)

    return _act


class GymRunner:
    """Runner for any gym env (random / ACT). When the env is filter-wrapped, the
    ``intervention_count`` / ``step_count`` of :class:`SafetyFilterWrapper` are surfaced."""

    def __init__(self, env, policy: Callable[[Mapping[str, Any]], np.ndarray]):
        self.env = env
        self.policy = policy
        self._obs = None
        self.n_steps = 0

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed: int) -> Dict[str, Any]:
        self._obs, info = self.env.reset(seed=seed)
        self.n_steps = 0
        return info or {}

    def step(self) -> StepRecord:
        from safety_bigym.filters.cost_signal import compute_cost

        action = self.policy(self._obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs = obs
        self.n_steps += 1
        safety = info.get("safety") or {}
        return StepRecord(
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            min_separation=float(safety.get("min_separation", float("inf"))),
            c_t=float(compute_cost(safety)),
        )

    @property
    def intervention_count(self) -> int:
        return int(getattr(self.env, "intervention_count", 0))

    @property
    def filter_step_count(self) -> int:
        return int(getattr(self.env, "step_count", 0))

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # pragma: no cover — best-effort
            pass


def apply_veto(
    *,
    critic,
    fallback,
    obs: Mapping[str, Any],
    sub_action: np.ndarray,
    to_raw: Callable[[np.ndarray], np.ndarray],
    from_raw: Callable[[np.ndarray], np.ndarray],
    threshold_R: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """SVF veto kernel for the CQN-AS path.

    ``sub_action`` is the agent's NORMALISED per-step action. ``to_raw``/``from_raw`` are
    the adapter's de/normalisation maps so the critic sees the same raw action the env
    will execute, and the fallback (built on the raw action space) is mapped back to the
    normalised space the adapter expects. Returns ``(executed_normalised_action, info)``
    where ``info`` matches ``SafetyFilterWrapper``'s ``info["safety_filter"]`` schema.
    """
    raw_proposed = np.asarray(to_raw(sub_action), dtype=np.float32)
    q = critic.q_value(obs, raw_proposed)
    q_scalar = float(q if not isinstance(q, np.ndarray) else q.item())
    intervened = q_scalar < float(threshold_R)
    if intervened:
        fallback_raw = fallback.compute(obs=obs, proposed=raw_proposed)
        executed = np.asarray(from_raw(np.asarray(fallback_raw, dtype=np.float32)), dtype=np.float32)
    else:
        executed = np.asarray(sub_action, dtype=np.float32)
    info = {
        "intervened": bool(intervened),
        "q_value": q_scalar,
        "threshold_R": float(threshold_R),
    }
    return executed, info


class CQNASRunner:
    """Runner for a CQN-AS snapshot — replays train_cqn_as.eval's action-selection
    (action_sequence chunking + optional temporal ensemble) and applies the SVF veto
    in-loop (the gym filter wrapper can't sit inside the adapter)."""

    def __init__(self, adapter, agent, *, action_sequence, temporal_ensemble=None,
                 filter_state=None, cbf_filter=None, global_step=0):
        self.adapter = adapter  # SafetyBiGymCQNAdapter (its _env may be ObsCacheWrapper)
        self.agent = agent
        self.action_sequence = int(action_sequence)
        self.temporal_ensemble = temporal_ensemble
        self.filter_state = filter_state  # (critic, fallback, threshold_R) or None
        # Geometric CBF directional-dodge filter (always-on, base-XY only). Mutually
        # exclusive with the SVF veto path above; only one is set at a time.
        self.cbf_filter = cbf_filter  # CBFDodgeFilter or None
        self.global_step = int(global_step)
        self._ts = None
        self._episode_step = 0
        self._action_chunk = None
        self.intervention_count = 0
        self.filter_step_count = 0

    def reset(self, seed: int) -> Dict[str, Any]:
        from safety_bigym.agents.cqn_as import utils

        utils.set_seed_everywhere(int(seed))
        self._ts = self.adapter.reset(seed=int(seed))
        self._episode_step = 0
        self._action_chunk = None
        self.intervention_count = 0
        self.filter_step_count = 0
        if self.temporal_ensemble is not None:
            self.temporal_ensemble.reset()
        return self._ts.info or {}

    def step(self) -> StepRecord:
        import torch

        from safety_bigym.agents.cqn_as import utils

        ts = self._ts
        if self.temporal_ensemble is not None or (self._episode_step % self.action_sequence == 0):
            with torch.no_grad(), utils.eval_mode(self.agent):
                raw_action = self.agent.act(
                    ts.rgb_obs, ts.low_dim_obs, self.global_step, eval_mode=True
                )
            self._action_chunk = np.asarray(raw_action).reshape([self.action_sequence, -1])
            if self.temporal_ensemble is not None:
                self.temporal_ensemble.register_action_sequence(self._action_chunk)

        if self.temporal_ensemble is not None:
            sub_action = self.temporal_ensemble.get_action()
        else:
            sub_action = self._action_chunk[self._episode_step % self.action_sequence]
        sub_action = np.asarray(sub_action, dtype=np.float32)

        filter_info = None
        if self.filter_state is not None:
            critic, fallback, threshold_R = self.filter_state
            obs_dict = self.adapter._env.last_obs  # ObsCacheWrapper cache (raw gym obs)
            sub_action, filter_info = apply_veto(
                critic=critic, fallback=fallback, obs=obs_dict, sub_action=sub_action,
                to_raw=self.adapter._convert_action_to_raw,
                from_raw=self.adapter._convert_action_from_raw,
                threshold_R=threshold_R,
            )
            self.filter_step_count += 1
            if filter_info["intervened"]:
                self.intervention_count += 1
        elif self.cbf_filter is not None:
            # Always-on geometric dodge: de-normalise, correct base-XY, re-normalise.
            obs_dict = self.adapter._env.last_obs  # ObsCacheWrapper cache (raw gym obs)
            raw_proposed = self.adapter._convert_action_to_raw(sub_action)
            raw_corrected, filter_info = self.cbf_filter.apply(obs_dict, raw_proposed)
            sub_action = np.asarray(
                self.adapter._convert_action_from_raw(raw_corrected), dtype=np.float32
            )
            self.filter_step_count += 1
            if filter_info["intervened"]:
                self.intervention_count += 1

        nts = self.adapter.step(sub_action)
        self._ts = nts
        self._episode_step += 1

        info = dict(nts.info) if nts.info else {}
        if filter_info is not None:
            info["safety_filter"] = filter_info
        terminated = bool(nts.last() and float(nts.discount) == 0.0)
        truncated = bool(nts.last() and not terminated)
        safety = info.get("safety") or {}
        return StepRecord(
            reward=float(nts.reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
            min_separation=float(safety.get("min_separation", float("inf"))),
            c_t=float(getattr(nts, "cost", 0.0)),
        )

    def close(self) -> None:
        try:
            self.adapter.close()
        except Exception:  # pragma: no cover
            pass


def build_cell_runner(
    meta,
    payload,
    *,
    snapshot_path,
    task: str,
    disruption: str,
    obs_mode: str,
    human_model: str,
    filter_critic=None,
    filter_threshold: float = 4.0,
    fallback_name: str = "zero_velocity",
    num_demos_for_stats: int = 5,
    cbf_config=None,
):
    """Build ``(runner, renderable_env)`` for one cell, branching on ``meta.kind``.

    Gym paths (random / ACT) build via :func:`env_build.build_g1_gym_env` and attach the
    filter as an outer wrapper. The CQN-AS path builds its own adapter from the snapshot's
    config and applies the veto in-loop.

    Exactly one of the safety filters may be active per cell:
      - ``filter_critic`` (+ ``filter_threshold`` / ``fallback_name``) -> learned SVF veto.
      - ``cbf_config`` (a dict of CBFDodgeFilter kwargs) -> geometric directional dodge.
    The CBF path is CQN-AS-only and needs no critic, so it skips
    :func:`filter_attach.assert_critic_covers_obs`.
    """
    from safety_bigym.benchmark import env_build, filter_attach

    if filter_critic is not None and cbf_config is not None:
        raise ValueError(
            "build_cell_runner: pass only one of filter_critic (SVF) or cbf_config (CBF)."
        )

    if meta.kind in ("random", "act"):
        env = env_build.build_g1_gym_env(
            task, disruption, obs_mode, human_model=human_model,
            cameras=meta.cameras, camera_resolution=meta.camera_resolution,
        )
        if filter_critic is not None:
            env = filter_attach.attach_filter_gym(
                env, critic=filter_critic, threshold_R=filter_threshold,
                fallback_name=fallback_name,
            )
        if meta.kind == "random":
            policy = make_random_policy(env.action_space, seed=0)
        else:
            from safety_bigym.benchmark.loader import _act_policy_from_snapshot  # noqa

            policy = _act_policy_from_snapshot(snapshot_path, env)
        return GymRunner(env, policy), env

    if meta.kind == "cqn_as":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = env_build.build_cqn_cfg(
            payload["config"], task=task, disruption=disruption,
            obs_mode=obs_mode, human_model=human_model, device=device,
        )
        wrapped, adapter = env_build.build_cqn_adapter(cfg, num_demos_for_stats=num_demos_for_stats)
        agent = env_build.make_cqn_agent(cfg, wrapped, payload)

        filter_state = None
        cbf_filter = None
        if filter_critic is not None:
            raw_env = adapter._env
            keys = raw_env.observation_space.spaces.keys() if hasattr(raw_env.observation_space, "spaces") else ()
            filter_attach.assert_critic_covers_obs(filter_critic, keys)
            from safety_bigym.filters.fallback import FallbackRegistry

            adapter._env = filter_attach.ObsCacheWrapper(adapter._env)
            fb = FallbackRegistry.build(fallback_name, adapter._env.action_space)
            filter_state = (filter_critic, fb, filter_threshold)
        elif cbf_config is not None:
            # Geometric CBF: no critic, so no assert_critic_covers_obs. Still needs the
            # raw obs cache so the in-loop dodge can read human_pos_estimate / base XY.
            from safety_bigym.filters.cbf_filter import CBFDodgeFilter

            adapter._env = filter_attach.ObsCacheWrapper(adapter._env)
            cbf_filter = CBFDodgeFilter(adapter._env.action_space, **dict(cbf_config))

        temporal_ensemble = None
        if bool(cfg.get("temporal_ensemble", False)):
            from safety_bigym.agents.cqn_as import utils

            temporal_ensemble = utils.TemporalEnsembleControl(
                int(cfg.env.episode_length), wrapped.action_spec(), int(cfg.action_sequence)
            )
        runner = CQNASRunner(
            adapter, agent, action_sequence=int(cfg.action_sequence),
            temporal_ensemble=temporal_ensemble, filter_state=filter_state,
            cbf_filter=cbf_filter,
        )
        return runner, adapter

    raise ValueError(f"Unknown policy kind {meta.kind!r}")


def run_episode(
    runner: "RolloutRunner",
    *,
    seed: int,
    episode_index: int,
    max_steps: int,
    filtered: bool,
    on_step: Callable[[], None] | None = None,
) -> EpisodeRecord:
    """Drive one runner for one episode into an :class:`EpisodeRecord`.

    Success uses ``info["task_success"]`` when present (matches train_cqn_as), falling
    back to cumulative-reward > 0 (BiGym sparse reward) otherwise. ``steps_to_completion``
    is the 1-based env-step index of the first success.

    ``on_step`` (if given) is called once after reset and after every step — used to
    capture a render frame per step without a second rollout.
    """
    info0 = runner.reset(seed)
    ep_safety: Dict[str, Any] = dict(info0.get("episode_safety", {})) if isinstance(info0, dict) else {}
    if on_step is not None:
        on_step()

    cum_reward = 0.0
    cost_integral = 0.0
    sum_q = 0.0
    first_success_step = -1
    last_task_success = None
    n = 0

    for _ in range(max_steps):
        rec = runner.step()
        if on_step is not None:
            on_step()
        n += 1
        cum_reward += rec.reward
        cost_integral += rec.c_t
        info = rec.info or {}

        sf = info.get("safety_filter")
        if isinstance(sf, dict):
            try:
                sum_q += float(sf.get("q_value", 0.0))
            except (TypeError, ValueError):  # pragma: no cover
                pass

        ts = info.get("task_success")
        if ts is not None:
            last_task_success = float(ts)
            if first_success_step == -1 and float(ts) > 0.0:
                first_success_step = n
        elif first_success_step == -1 and cum_reward > 0.0:
            first_success_step = n

        es = info.get("episode_safety")
        if isinstance(es, dict):
            ep_safety = es

        if rec.done:
            break

    success = first_success_step != -1
    return EpisodeRecord(
        seed=int(seed),
        episode_index=int(episode_index),
        success=bool(success),
        episode_reward=float(cum_reward),
        n_steps=int(n),
        steps_to_completion=(float(first_success_step) if success else float("nan")),
        ep_safety=dict(ep_safety),
        ep_cost_integral=float(cost_integral),
        filtered=bool(filtered),
        n_interventions=int(runner.intervention_count) if filtered else 0,
        filter_steps=int(runner.filter_step_count) if filtered else 0,
        sum_q_value=float(sum_q),
        task_success_raw=last_task_success,
    )
