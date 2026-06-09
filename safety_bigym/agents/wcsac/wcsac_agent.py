"""WCSAC agent (Worst-Case Soft Actor-Critic).

Reference: Q. Yang, T. D. Simao, S. H. Tindemans, M. T. J. Spaan,
"WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement
Learning", AAAI 2021.

This is a faithful, self-contained reimplementation that plugs into the
``train_cqn_as.py`` training stack as ``agent=wcsac``. It implements the
duck-typed agent interface that loop expects (``act``,
``add_noise_to_action``, ``update``, ``update_target_critic``, ``train``,
``state_dict`` / ``load_state_dict``) and exposes ``_lambda`` so the
Lagrangian W&B payload logs the multiplier just like the CQN-AS Lagrangian
agent.

Algorithm (per env step / replay batch):
  * Reward side  : SAC -- stochastic squashed-Gaussian actor, twin reward
    critics with clipped-double-Q targets, automatic entropy temperature.
  * Safety side  : a Gaussian safety critic predicting the *mean* QC and
    *variance* VC of the discounted cost return; the worst-case constraint is
    CVaR_alpha(cost return) = QC + k(alpha)*sqrt(VC).
  * Constraint   : a learnable Lagrange multiplier lambda enforces
    CVaR_alpha <= d via projected dual gradient ascent.

The per-step cost ``c_t`` and its n-step discounted return arrive in
``batch["cost"]`` already (the CQN-AS replay carries it); nothing in the env /
replay / RoboBase needs changing.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..cqn_as import utils
from ..cqn_as.agent import MultiViewCNNEncoder, RandomShiftsAug
from .nets import GaussianSafetyCritic, SquashedGaussianActor, TwinQCritic


def gaussian_cvar_coefficient(alpha: float) -> float:
    """k(alpha) with CVaR_alpha(N(mu, sigma^2)) = mu + k(alpha) * sigma.

    For the *upper* tail at confidence ``alpha`` (i.e. the mean over the worst
    ``1 - alpha`` fraction of the cost return),
        CVaR_alpha = mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha),
    with phi the standard-normal pdf and Phi^{-1} its inverse CDF.
    """
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"cvar_alpha must be in (0, 1), got {alpha}")
    # Phi^{-1}(a) = sqrt(2) * erfinv(2a - 1); torch.erfinv is always available.
    z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * a - 1.0)).item()
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return phi / (1.0 - a)


class WCSACAgent:
    def __init__(
        self,
        rgb_obs_shape,
        low_dim_obs_shape,
        action_shape,
        device,
        actor_lr,
        critic_lr,
        safety_lr,
        alpha_lr,
        init_temperature,
        critic_target_tau,
        critic_target_interval,
        update_every_steps,
        feature_dim,
        hidden_dim,
        weight_decay,
        cvar_alpha,
        cost_budget,
        lambda_lr,
        lambda_init,
        lambda_max,
        num_expl_steps,
    ):
        self.device = torch.device(device)
        self.critic_target_tau = critic_target_tau
        self.critic_target_interval = critic_target_interval
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps

        # action_shape = [action_sequence, *act_dim]; the actor emits the whole
        # flattened chunk, which train_cqn_as.py reshapes to [action_sequence, -1].
        self.action_seq = int(action_shape[0])
        self.total_action_dim = int(np.prod(action_shape))
        low_dim = int(low_dim_obs_shape[-1])

        # Separate reward / safety pixel encoders (mirrors the CQN-AS Lagrangian
        # agent's split). Reused verbatim from the vendored CQN-AS front-end.
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(self.device)
        self.cost_encoder = MultiViewCNNEncoder(rgb_obs_shape).to(self.device)
        repr_dim = self.encoder.repr_dim

        self.actor = SquashedGaussianActor(
            repr_dim, low_dim, self.total_action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.critic = TwinQCritic(
            repr_dim, low_dim, self.total_action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.critic_target = TwinQCritic(
            repr_dim, low_dim, self.total_action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.safety_critic = GaussianSafetyCritic(
            repr_dim, low_dim, self.total_action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.safety_critic_target = GaussianSafetyCritic(
            repr_dim, low_dim, self.total_action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())

        # Encoders are trained by their respective critic losses (the actor uses
        # detached features), so they ride along in the critic optimisers.
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            list(self.critic.parameters()) + list(self.encoder.parameters()),
            lr=critic_lr,
            weight_decay=weight_decay,
        )
        self.safety_opt = torch.optim.AdamW(
            list(self.safety_critic.parameters())
            + list(self.cost_encoder.parameters()),
            lr=safety_lr,
            weight_decay=weight_decay,
        )

        # Automatic entropy temperature (SAC).
        self.log_alpha = torch.tensor(
            math.log(float(init_temperature)), device=self.device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(self.total_action_dim)

        # Worst-case constraint machinery.
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_coef = gaussian_cvar_coefficient(cvar_alpha)
        self.cost_budget = float(cost_budget)
        self.lambda_lr = float(lambda_lr)
        self.lambda_max = float(lambda_max)
        self._lambda = float(lambda_init)
        self._last_cvar = float("nan")

        self.aug = RandomShiftsAug(pad=4)

        # Tells train_cqn_as.py's collection loop to call act(eval_mode=False)
        # so exploration comes from the stochastic policy (the loop's external
        # additive-noise path, add_noise_to_action, is a no-op for us).
        self.stochastic_act = True

        self.train()
        self.critic_target.eval()
        self.safety_critic_target.eval()

    # ------------------------------------------------------------------
    # mode / persistence
    # ------------------------------------------------------------------
    def train(self, training=True):
        self.training = training
        for m in (
            self.encoder,
            self.cost_encoder,
            self.actor,
            self.critic,
            self.safety_critic,
        ):
            m.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "cost_encoder": self.cost_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "safety_critic": self.safety_critic.state_dict(),
            "safety_critic_target": self.safety_critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "safety_opt": self.safety_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "lambda": self._lambda,
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.cost_encoder.load_state_dict(state_dict["cost_encoder"])
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.safety_critic.load_state_dict(state_dict["safety_critic"])
        self.safety_critic_target.load_state_dict(state_dict["safety_critic_target"])
        if "log_alpha" in state_dict:
            with torch.no_grad():
                self.log_alpha.copy_(state_dict["log_alpha"].to(self.device))
        if "lambda" in state_dict:
            self._lambda = float(state_dict["lambda"])
        # Optimisers are best-effort (eval-only runs don't need them).
        for opt_key, opt in (
            ("actor_opt", self.actor_opt),
            ("critic_opt", self.critic_opt),
            ("safety_opt", self.safety_opt),
            ("alpha_opt", self.alpha_opt),
        ):
            if opt_key in state_dict:
                try:
                    opt.load_state_dict(state_dict[opt_key])
                except (ValueError, KeyError):
                    pass

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(
            low_dim_obs, device=self.device, dtype=torch.float
        ).unsqueeze(0)
        if (not eval_mode) and step < self.num_expl_steps:
            action = torch.empty(
                1, self.total_action_dim, device=self.device
            ).uniform_(-1.0, 1.0)
        else:
            features = self.encoder(rgb_obs)
            if eval_mode:
                action = self.actor.mean_action(features, low_dim_obs)
            else:
                action, _ = self.actor(features, low_dim_obs, sample=True)
        return action.detach().cpu().numpy()[0]

    def add_noise_to_action(self, action: np.ndarray, step: int) -> np.ndarray:
        # Exploration is already injected by the stochastic policy in act().
        return action

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def _aug_encode_pair(self, rgb, next_rgb):
        rgb = torch.stack([self.aug(rgb[:, v]) for v in range(rgb.shape[1])], 1)
        next_rgb = torch.stack(
            [self.aug(next_rgb[:, v]) for v in range(next_rgb.shape[1])], 1
        )
        return rgb, next_rgb

    def update(self, batch):
        low_dim = batch["low_dim_obs"]
        next_low_dim = batch["next_low_dim_obs"]
        bsz = low_dim.shape[0]
        action = batch["action"].reshape(bsz, -1)
        if action.shape[1] != self.total_action_dim:
            raise ValueError(
                f"WCSAC expected flattened action dim {self.total_action_dim} "
                f"(action_sequence={self.action_seq}); got {action.shape[1]}. "
                f"Run WCSAC with action_sequence=1 (see cfgs/agent/wcsac.yaml)."
            )
        reward = batch["reward"].reshape(bsz, 1)
        cost = batch["cost"].reshape(bsz, 1)
        discount = batch["discount"].reshape(bsz, 1)

        # Independent augmentation per encoder (reward vs safety).
        rgb_r, next_rgb_r = self._aug_encode_pair(batch["rgb_obs"], batch["next_rgb_obs"])
        rgb_c, next_rgb_c = self._aug_encode_pair(batch["rgb_obs"], batch["next_rgb_obs"])

        f_r = self.encoder(rgb_r)
        f_c = self.cost_encoder(rgb_c)
        with torch.no_grad():
            f_r_next = self.encoder(next_rgb_r)
            f_c_next = self.cost_encoder(next_rgb_c)

        metrics = {}
        metrics.update(
            self._update_critics(
                f_r, f_c, f_r_next, f_c_next,
                low_dim, next_low_dim, action, reward, cost, discount,
            )
        )
        metrics.update(self._update_actor_and_alpha(f_r.detach(), f_c.detach(), low_dim))
        metrics.update(self._update_lambda())
        # Return a plain dict, NOT a scalar TensorDict: train_cqn_as.py's _log
        # early-returns on len(metrics)==0, and a batch_size=[] TensorDict has
        # len 0 -- which would silently drop every WCSAC training curve (lambda,
        # CVaR, actor/critic losses) from W&B. _log handles plain dicts.
        return metrics

    def _update_critics(
        self, f_r, f_c, f_r_next, f_c_next,
        low_dim, next_low_dim, action, reward, cost, discount,
    ):
        with torch.no_grad():
            next_action, next_logp = self.actor(f_r_next, next_low_dim, sample=True)

            # Reward target (clipped double-Q + entropy).
            q1_t, q2_t = self.critic_target(f_r_next, next_low_dim, next_action)
            q_next = torch.min(q1_t, q2_t) - self.alpha * next_logp
            y_r = reward + discount * q_next

            # Safety mean target and the second-moment piece for the variance.
            qc_next, vc_next = self.safety_critic_target(
                f_c_next, next_low_dim, next_action
            )
            y_qc = cost + discount * qc_next
            # E[(c + gamma * D')^2] for D' ~ N(QC', VC').
            second_moment = (
                cost**2
                + 2.0 * discount * cost * qc_next
                + (discount**2) * (qc_next**2 + vc_next)
            )

        # Reward critic.
        q1, q2 = self.critic(f_r, low_dim, action)
        reward_critic_loss = F.mse_loss(q1, y_r) + F.mse_loss(q2, y_r)
        self.critic_opt.zero_grad(set_to_none=True)
        reward_critic_loss.backward()
        self.critic_opt.step()

        # Safety critic. Variance target uses the *bootstrapped* mean QC(s,a)
        # (network output, detached) so it captures transition variance, not
        # just gamma^2 * VC' -- this is the WCSAC variance recursion.
        qc, vc = self.safety_critic(f_c, low_dim, action)
        y_vc = (second_moment - qc.detach() ** 2).clamp(min=0.0)
        safety_mean_loss = F.mse_loss(qc, y_qc)
        safety_var_loss = F.mse_loss(vc, y_vc)
        safety_critic_loss = safety_mean_loss + safety_var_loss
        self.safety_opt.zero_grad(set_to_none=True)
        safety_critic_loss.backward()
        self.safety_opt.step()

        return {
            "reward_critic_loss": reward_critic_loss.detach(),
            "safety_mean_loss": safety_mean_loss.detach(),
            "safety_var_loss": safety_var_loss.detach(),
            "batch_reward": reward.mean().detach(),
            "batch_cost": cost.mean().detach(),
            "qc_mean": qc.mean().detach(),
            "vc_mean": vc.mean().detach(),
        }

    def _update_actor_and_alpha(self, f_r, f_c, low_dim):
        # f_r, f_c are detached features -- the actor never trains the encoders.
        action, log_prob = self.actor(f_r, low_dim, sample=True)
        q1_pi, q2_pi = self.critic(f_r, low_dim, action)
        q_pi = torch.min(q1_pi, q2_pi)
        qc_pi, vc_pi = self.safety_critic(f_c, low_dim, action)
        cvar_pi = qc_pi + self.cvar_coef * torch.sqrt(vc_pi.clamp(min=1e-8))

        actor_loss = (
            self.alpha.detach() * log_prob - q_pi + self._lambda * cvar_pi
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Entropy temperature.
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._last_cvar = float(cvar_pi.mean().detach())
        return {
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha.detach(),
            "entropy": (-log_prob.mean()).detach(),
            "cvar": cvar_pi.mean().detach(),
        }

    def _update_lambda(self):
        # Projected dual gradient ascent: lambda <- clip(lambda + eta*(CVaR - d)).
        violation = self._last_cvar - self.cost_budget
        self._lambda = float(
            min(max(self._lambda + self.lambda_lr * violation, 0.0), self.lambda_max)
        )
        return {
            "lambda": self._lambda,
            "cost_violation": violation,
        }

    def update_target_critic(self, step):
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )
            utils.soft_update_params(
                self.safety_critic, self.safety_critic_target, self.critic_target_tau
            )
