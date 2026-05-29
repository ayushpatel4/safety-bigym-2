"""Phase 3 (P3.1) constrained-RL extension of the vendored CQN-AS agent.

This is the "Lagrangian glue" that turns the staged cost pipeline (per-step
``c_t`` in ``batch["cost"]`` / ``batch["max_cost"]``, wired in P3.0c) into
actual constrained RL. It composes -- never edits -- the vendored
``CQNASAgent`` (see the no-edit directive in ``agent.py`` header).

Three pieces:

1. :class:`LagrangianPID` -- a PID controller on the rolling-mean cost that
   drives the Lagrange multiplier ``lambda``. (Lives in :mod:`lagrangian`,
   torch-only, so it is unit-testable without ``tensordict``.)
2. :func:`dual_select` -- the cost-aware argmax ``argmax_a [Q_r - lambda * Q_c]``
   over the per-bin coarse-to-fine candidates. Factored out so it is unit
   testable on stub tensors (lambda=0 reduces to the plain Q_r argmax). (Also
   in :mod:`lagrangian`.)
3. :class:`LagrangianCQNASAgent` -- subclass of ``CQNASAgent`` that adds a
   second C51 C2F cost critic ``Q_c`` (its own encoder + target), trains it by
   per-env-step Bellman regression on ``batch["cost"]``, and selects actions
   with the dual-Q rule.

Design decisions (confirmed 2026-05-20, see docs/IMPLEMENTATION_STATUS.md):
- ``Q_c`` is a verbatim clone of ``C2FCritic`` (kept C51); expected cost is the
  atom-weighted sum, exactly as ``C2FCritic.get_action`` already computes it,
  over a cost-range support ``[cost_v_min, cost_v_max]``. SVF warm-start
  (``filters/cost_critic.py``) is deferred to a future B-value-CVaR variant.
- The ``Q_c`` Bellman backup evaluates the *dual* policy
  ``a' = argmax_a [Q_r - lambda * Q_c]`` from the target nets (the action the
  deployed constrained policy actually takes). The vendored reward critic keeps
  its greedy ``argmax Q_r`` backup untouched.
- ``Q_c`` gets its OWN ``MultiViewCNNEncoder`` + optimizer so cost gradients
  never corrupt the reward features and vice versa.

Correctness invariants:
- ``lambda`` enters ONLY at action selection (:func:`dual_select`), never in any
  critic's regression target -- both Q-networks keep stationary targets.
- ``Q_c`` has its own target network + soft update.
- The cost backup is PER-ENV-STEP: ``batch["cost"]`` already carries the n-step
  discounted per-step cost (replay_buffer.py), NOT a per-K-chunk average. We do
  not regress that granularity.
- ``update()`` returns a ``TensorDict`` -- never bool-test it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from . import utils
from .agent import C2FCritic, CQNASAgent, MultiViewCNNEncoder
from .cqn_utils import random_action_if_within_delta, zoom_in
from .lagrangian import LagrangianPID, dual_select

__all__ = ["LagrangianPID", "dual_select", "LagrangianCQNASAgent"]


class LagrangianCQNASAgent(CQNASAgent):
    """CQN-AS with a Lagrangian cost critic + dual-Q action selection."""

    def __init__(
        self,
        *,
        cost_v_min: float = 0.0,
        cost_v_max: float = 10.0,
        lambda_init: float = 0.0,
        lambda_k_i: float = 1e-3,
        lambda_k_p: float = 1e-2,
        lambda_k_d: float = 0.0,
        lambda_max: float = 100.0,
        cost_budget: float = 0.01,
        rolling_cost_momentum: float = 0.99,
        **base_kwargs,
    ):
        # NOTE: super().__init__ calls self.train() (agent.py:700) BEFORE our
        # cost nets exist; the overridden train() guards on hasattr.
        super().__init__(**base_kwargs)

        device = base_kwargs["device"]
        self._build_cost_critic(base_kwargs, cost_v_min, cost_v_max, device)

        self._pid = LagrangianPID(
            k_i=lambda_k_i,
            k_p=lambda_k_p,
            k_d=lambda_k_d,
            lambda_max=lambda_max,
            cost_budget=cost_budget,
            lambda_init=lambda_init,
        )
        self._lambda = float(lambda_init)
        self._rolling_cost = 0.0
        self._rolling_cost_momentum = float(rolling_cost_momentum)

    def _build_cost_critic(self, base_kwargs, cost_v_min, cost_v_max, device):
        self.cost_encoder = MultiViewCNNEncoder(base_kwargs["rgb_obs_shape"]).to(device)
        critic_args = (
            base_kwargs["action_shape"],
            self.cost_encoder.repr_dim,
            base_kwargs["low_dim_obs_shape"][-1],
            base_kwargs["feature_dim"],
            base_kwargs["hidden_dim"],
            base_kwargs["levels"],
            base_kwargs["bins"],
            base_kwargs["atoms"],
            cost_v_min,
            cost_v_max,
            base_kwargs["gru_layers"],
            base_kwargs["rgb_encoder_layers"],
            base_kwargs["use_parallel_impl"],
        )
        self.cost_critic = C2FCritic(*critic_args).to(device)
        self.cost_critic_target = C2FCritic(*critic_args).to(device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())

        self.cost_encoder_opt = torch.optim.AdamW(
            self.cost_encoder.parameters(),
            lr=base_kwargs["lr"],
            weight_decay=base_kwargs["weight_decay"],
        )
        self.cost_critic_opt = torch.optim.AdamW(
            self.cost_critic.parameters(),
            lr=base_kwargs["lr"],
            weight_decay=base_kwargs["weight_decay"],
        )

        self.cost_encoder.train()
        self.cost_critic.train()
        self.cost_critic_target.eval()

    @property
    def lam(self) -> float:
        return self._lambda

    def train(self, training: bool = True):
        super().train(training)
        # Guard: super().__init__ calls this before the cost nets are built.
        if getattr(self, "cost_encoder", None) is not None:
            self.cost_encoder.train(training)
            self.cost_critic.train(training)

    def state_dict(self):
        sd = super().state_dict()
        sd.update(
            {
                "cost_encoder": self.cost_encoder.state_dict(),
                "cost_critic": self.cost_critic.state_dict(),
                "cost_critic_target": self.cost_critic_target.state_dict(),
                "cost_encoder_opt": self.cost_encoder_opt.state_dict(),
                "cost_critic_opt": self.cost_critic_opt.state_dict(),
                "lagrangian_pid": self._pid.state_dict(),
                "lambda": self._lambda,
                "rolling_cost": self._rolling_cost,
            }
        )
        return sd

    def load_state_dict(self, state_dict):
        """Load -- including warm-start from a plain CQN-AS (no-cost) snapshot.

        Cost-side keys (``cost_encoder``, ``cost_critic``, ``cost_critic_target``,
        their opts, ``lagrangian_pid``, ``lambda``, ``rolling_cost``) are
        OPTIONAL. When absent (the snapshot was produced by ``agent=cqn_as``,
        not ``cqn_as_lagrangian``), the reward side is restored from the base
        snapshot and the cost critic + lambda stay at their fresh init -- the
        intended P3.1 warm-start path (load the stage-1 base policy, then learn
        Q_c + lambda from scratch under the full disruption). When present, the
        full constrained-agent state is restored (snapshot resume).
        """
        super().load_state_dict(state_dict)
        if "cost_critic" in state_dict:
            self.cost_encoder.load_state_dict(state_dict["cost_encoder"])
            self.cost_critic.load_state_dict(state_dict["cost_critic"])
            self.cost_critic_target.load_state_dict(state_dict["cost_critic_target"])
        for key, opt in (
            ("cost_encoder_opt", self.cost_encoder_opt),
            ("cost_critic_opt", self.cost_critic_opt),
        ):
            if key in state_dict:
                try:
                    opt.load_state_dict(state_dict[key])
                except (ValueError, KeyError):
                    pass
        if "lagrangian_pid" in state_dict:
            self._pid.load_state_dict(state_dict["lagrangian_pid"])
        self._lambda = float(state_dict.get("lambda", self._lambda))
        self._rolling_cost = float(state_dict.get("rolling_cost", self._rolling_cost))

    # ------------------------------------------------------------------ #
    # Dual-Q action selection
    # ------------------------------------------------------------------ #
    def _dual_get_action(
        self,
        rew_feat: torch.Tensor,
        cost_feat: torch.Tensor,
        low_dim_obs: torch.Tensor,
        lam: float,
        *,
        use_target: bool,
    ) -> torch.Tensor:
        """Coarse-to-fine action selection with the dual-Q rule.

        Mirrors ``C2FCritic.get_action`` (agent.py:415-432) but at each level
        computes both the reward and cost expected-Q over the candidate bins
        and zooms in on ``argmax_a [Q_r - lam * Q_c]`` -- one shared zoom path.

        ``rew_feat`` / ``cost_feat`` are already-encoded CNN features (outputs
        of ``self.encoder`` / ``self.cost_encoder``).
        """
        critic_r = self.critic_target if use_target else self.critic
        critic_c = self.cost_critic_target if use_target else self.cost_critic

        low = critic_r.initial_low.repeat(rew_feat.shape[0], 1).detach()
        high = critic_r.initial_high.repeat(rew_feat.shape[0], 1).detach()

        feats_r = critic_r.network.encode(rew_feat, low_dim_obs)
        feats_c = critic_c.network.encode(cost_feat, low_dim_obs)

        for level in range(critic_r.levels):
            mid = (low + high) / 2

            q_logits_r = critic_r.network.forward_each_level(level, feats_r, mid)
            qs_r = (
                F.softmax(q_logits_r, 3)
                * critic_r.support.expand_as(q_logits_r).detach()
            ).sum(3)

            q_logits_c = critic_c.network.forward_each_level(level, feats_c, mid)
            qs_c = (
                F.softmax(q_logits_c, 3)
                * critic_c.support.expand_as(q_logits_c).detach()
            ).sum(3)

            combined = qs_r - lam * qs_c
            argmax_q = random_action_if_within_delta(combined)
            if argmax_q is None:  # pragma: no cover - delta path rarely taken
                argmax_q = dual_select(qs_r, qs_c, lam)

            low, high = zoom_in(low, high, argmax_q, critic_r.bins)

        return (high + low) / 2.0

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rew_feat = self.encoder(rgb_obs)
        cost_feat = self.cost_encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self._dual_get_action(
            rew_feat, cost_feat, low_dim_obs, self._lambda, use_target=True
        )
        stddev = torch.ones_like(action) * stddev
        dist = utils.TruncatedNormal(action, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        action = self.critic.encode_decode_action(action)
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------ #
    # Cost critic update + lambda step
    # ------------------------------------------------------------------ #
    def update_cost_critic(self, batch) -> TensorDict:
        """Per-env-step C51 Bellman regression of Q_c on ``batch["cost"]``."""
        rgb_obs = batch["rgb_obs"]
        low_dim_obs = batch["low_dim_obs"]
        action = batch["action"]
        cost = batch["cost"]
        discount = batch["discount"]
        next_rgb_obs = batch["next_rgb_obs"]
        next_low_dim_obs = batch["next_low_dim_obs"]

        # Independent augmentation for the cost critic (its own encoder).
        rgb_obs = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
        )
        next_rgb_obs = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
        )

        cost_feat = self.cost_encoder(rgb_obs)
        with torch.no_grad():
            next_cost_feat = self.cost_encoder(next_rgb_obs)
            # Reward features (for the dual next-action) come from the reward
            # encoder; no grad flows back to it from the cost loss.
            next_rew_feat = self.encoder(next_rgb_obs)
            next_action = self._dual_get_action(
                next_rew_feat, next_cost_feat, next_low_dim_obs,
                self._lambda, use_target=True,
            )
            # lambda does NOT enter the regression target -- only the next-action
            # selection above. The target value below is pure discounted cost.
            target_c_probs_a = self.cost_critic_target.compute_target_q_dist(
                next_cost_feat, next_low_dim_obs, next_action, cost, discount
            )

        _, _, _, log_q_probs_a = self.cost_critic(cost_feat, low_dim_obs, action)
        q_c_loss = -torch.sum(target_c_probs_a * log_q_probs_a, 3).mean()

        self.cost_encoder_opt.zero_grad(set_to_none=True)
        self.cost_critic_opt.zero_grad(set_to_none=True)
        q_c_loss.backward()
        self.cost_critic_opt.step()
        self.cost_encoder_opt.step()

        return TensorDict(q_c_loss=q_c_loss.detach())

    def _update_lambda(self, batch) -> dict:
        batch_cost = batch["cost"].mean().item()
        m = self._rolling_cost_momentum
        self._rolling_cost = m * self._rolling_cost + (1.0 - m) * batch_cost
        self._lambda = self._pid.update(self._rolling_cost)
        return {
            "lambda": self._lambda,
            "rolling_cost": self._rolling_cost,
            "cost_violation": self._rolling_cost - self._pid.cost_budget,
            "batch_cost": batch_cost,
        }

    def update(self, batch):
        # 1) Reward critic + reward encoder -- vendored path, untouched.
        metrics = super().update(batch)
        # 2) Cost critic. Set the key directly (mirrors the base agent's
        #    ``metrics["batch_reward"] = ...`` idiom) rather than TensorDict.update.
        metrics["q_c_loss"] = self.update_cost_critic(batch)["q_c_loss"]
        # 3) Lagrange multiplier.
        for key, val in self._update_lambda(batch).items():
            metrics[key] = torch.as_tensor(val, device=self.device)
        return metrics

    def update_target_critic(self, step):
        super().update_target_critic(step)
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.cost_critic, self.cost_critic_target, self.critic_target_tau
            )
