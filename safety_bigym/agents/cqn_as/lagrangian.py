"""Dependency-light Lagrangian primitives for the Phase 3 (P3.1) cost glue.

Kept separate from :mod:`lagrangian_agent` (which pulls in ``tensordict`` via the
vendored agent) so the pure-logic pieces -- the PID multiplier controller and
the dual-Q bin selector -- are unit-testable with only ``torch`` installed.
"""

from __future__ import annotations

import torch


class LagrangianPID:
    """PID controller on rolling-mean cost driving the Lagrange multiplier.

    Per UPDATED_PROJECT_PLAN.md / PHASE3_1_HANDOFF.md::

        cost_violation = rolling_mean_cost - cost_budget
        lambda = clip(lambda + K_I * cv + K_P * cv + K_D * (cv - prev_cv),
                      0, lambda_max)

    Defaults: ``K_I=1e-3, K_P=1e-2, K_D=0, lambda_max=100, cost_budget=0.01``.
    lambda enters the policy ONLY at action-selection time (see
    :func:`dual_select`), never in any critic's regression target.
    """

    def __init__(
        self,
        *,
        k_i: float = 1e-3,
        k_p: float = 1e-2,
        k_d: float = 0.0,
        lambda_max: float = 100.0,
        cost_budget: float = 0.01,
        lambda_init: float = 0.0,
    ):
        self.k_i = float(k_i)
        self.k_p = float(k_p)
        self.k_d = float(k_d)
        self.lambda_max = float(lambda_max)
        self.cost_budget = float(cost_budget)
        self.lam = float(lambda_init)
        self._prev_violation = 0.0

    def update(self, rolling_mean_cost: float) -> float:
        """Step the controller with the current rolling-mean cost; return lambda."""
        cv = float(rolling_mean_cost) - self.cost_budget
        delta = cv - self._prev_violation
        new_lam = self.lam + self.k_i * cv + self.k_p * cv + self.k_d * delta
        self.lam = float(min(max(new_lam, 0.0), self.lambda_max))
        self._prev_violation = cv
        return self.lam

    def state_dict(self) -> dict:
        return {
            "lam": self.lam,
            "prev_violation": self._prev_violation,
            "k_i": self.k_i,
            "k_p": self.k_p,
            "k_d": self.k_d,
            "lambda_max": self.lambda_max,
            "cost_budget": self.cost_budget,
        }

    def load_state_dict(self, state: dict) -> None:
        self.lam = float(state.get("lam", self.lam))
        self._prev_violation = float(state.get("prev_violation", self._prev_violation))


def dual_select(qs_r: torch.Tensor, qs_c: torch.Tensor, lam: float) -> torch.Tensor:
    """Cost-aware argmax over the last (bin) dimension.

    ``combined = qs_r - lam * qs_c``; returns ``combined.max(-1)[1]``. With
    ``lam == 0`` this is exactly the plain ``qs_r`` argmax (mirrors the reward
    path's ``qs.max(-1)[1]`` at agent.py:428); with large ``lam`` it shifts
    toward low-``qs_c`` bins.
    """
    combined = qs_r - lam * qs_c
    return combined.max(-1)[1]
