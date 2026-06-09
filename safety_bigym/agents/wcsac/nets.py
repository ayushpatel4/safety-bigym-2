"""Network modules for the WCSAC agent.

All three heads sit on top of a frozen-for-the-actor pixel encoder
(``MultiViewCNNEncoder``, reused verbatim from the vendored CQN-AS code so the
visual front-end matches every other agent in the repo). Each head owns a
small DrQ-style trunk (Linear -> LayerNorm -> Tanh) that projects the very wide
flattened CNN feature (``encoder.repr_dim``) down to ``feature_dim`` before the
MLP body, mirroring the projection ``C2FCritic`` does internally.

Shapes throughout:
    features  : [B, repr_dim]          (output of MultiViewCNNEncoder)
    low_dim   : [B, low_dim]
    action    : [B, action_dim]        (flattened action_sequence * act_dim)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cqn_as import utils

# tanh-squashed Gaussian log-std bounds (standard SAC values; matches RoboBase
# SACActor in robobase/method/sac_lix.py).
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _trunk(repr_dim: int, feature_dim: int) -> nn.Sequential:
    """DrQ-style projection from the wide CNN feature to ``feature_dim``."""
    return nn.Sequential(
        nn.Linear(repr_dim, feature_dim),
        nn.LayerNorm(feature_dim),
        nn.Tanh(),
    )


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class SquashedGaussianActor(nn.Module):
    """Stochastic policy: a = tanh(N(mu(s), sigma(s)))."""

    def __init__(self, repr_dim, low_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = _trunk(repr_dim, feature_dim)
        self.net = _mlp(feature_dim + low_dim, hidden_dim, 2 * action_dim)
        self.action_dim = action_dim
        self.apply(utils.weight_init)

    def _mu_log_std(self, features, low_dim):
        h = self.trunk(features)
        mu, log_std = self.net(torch.cat([h, low_dim], dim=-1)).chunk(2, dim=-1)
        # Bound log_std into [LOG_STD_MIN, LOG_STD_MAX] via a squashed map
        # (same parameterisation as RoboBase's SACActor).
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mu, log_std

    def forward(self, features, low_dim, sample: bool = True):
        """Return (action, log_prob).

        ``sample=True`` reparameterised-samples the squashed Gaussian (used for
        exploration and the actor loss). ``sample=False`` returns the
        deterministic mean action tanh(mu) (used at eval); ``log_prob`` is then
        the log-prob of that mean under the policy.
        """
        mu, log_std = self._mu_log_std(features, low_dim)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        pre_tanh = mu if not sample else normal.rsample()
        action = torch.tanh(pre_tanh)
        # log_prob with tanh change-of-variables correction. The stable form of
        # log(1 - tanh(x)^2) is 2*(log 2 - x - softplus(-2x)).
        log_prob = normal.log_prob(pre_tanh)
        log_prob -= 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def mean_action(self, features, low_dim):
        mu, _ = self._mu_log_std(features, low_dim)
        return torch.tanh(mu)


class TwinQCritic(nn.Module):
    """Twin reward Q-networks Q_r1, Q_r2 (clipped-double-Q)."""

    def __init__(self, repr_dim, low_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = _trunk(repr_dim, feature_dim)
        self.q1 = _mlp(feature_dim + low_dim + action_dim, hidden_dim, 1)
        self.q2 = _mlp(feature_dim + low_dim + action_dim, hidden_dim, 1)
        self.apply(utils.weight_init)

    def forward(self, features, low_dim, action):
        h = self.trunk(features)
        x = torch.cat([h, low_dim, action], dim=-1)
        return self.q1(x), self.q2(x)


class GaussianSafetyCritic(nn.Module):
    """Safety critic modelling the cost-return as N(mean, var).

    WCSAC assumes the discounted cost return D_c(s,a) is Gaussian; this head
    predicts its mean (``QC``) and variance (``VC``). Variance is produced via
    softplus to stay strictly positive.
    """

    def __init__(self, repr_dim, low_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = _trunk(repr_dim, feature_dim)
        self.body = _mlp(feature_dim + low_dim + action_dim, hidden_dim, 2)
        self.apply(utils.weight_init)

    def forward(self, features, low_dim, action):
        """Return (mean, var), each [B, 1]; var > 0."""
        h = self.trunk(features)
        out = self.body(torch.cat([h, low_dim, action], dim=-1))
        mean, var_param = out[..., :1], out[..., 1:]
        var = F.softplus(var_param) + 1e-6
        return mean, var
