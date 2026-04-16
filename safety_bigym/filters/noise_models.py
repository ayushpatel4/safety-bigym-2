"""Noise models for the Mock BodySLAM++ observation wrapper (Phase 1).

The real BodySLAM++ perception stack exhibits temporally-correlated error —
the regressor locks onto a slightly-wrong skeleton for several frames before
re-snapping. OrnsteinUhlenbeckNoise reproduces this with an AR(1) recurrence
whose steady-state std equals the configured sigma regardless of alpha.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class OrnsteinUhlenbeckNoise:
    """Vectorised discrete OU process with steady-state std = sigma.

    Recurrence: n_{t+1} = alpha * n_t + sqrt(1 - alpha^2) * sigma * eps_t,
    eps_t ~ N(0, I).

    The sqrt(1 - alpha^2) normalisation means the stationary distribution is
    N(0, sigma^2 I) regardless of alpha — so callers can sweep alpha without
    inadvertently changing the noise magnitude.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        alpha: float,
        sigma: float,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.shape = tuple(shape)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self._innovation_scale = float(np.sqrt(1.0 - self.alpha ** 2)) * self.sigma
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(self.shape, dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> None:
        """Zero the internal state; optionally reseed the RNG."""
        self.state = np.zeros(self.shape, dtype=np.float64)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def step(self) -> np.ndarray:
        """Advance one step and return the new noise sample."""
        eps = self._rng.standard_normal(self.shape)
        self.state = self.alpha * self.state + self._innovation_scale * eps
        return self.state.copy()
