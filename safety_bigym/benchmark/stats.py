"""Statistics helpers for the benchmark harness — bootstrap CIs, CVaR, percentiles.

PURE module: depends only on ``numpy``. No torch / mujoco / pandas imports here, so
the unit tests exercise these helpers in milliseconds without constructing an env or
loading a model.

The bootstrap is **seedable** so:
  * the CSV's ``*_ci_lo/hi`` columns reproduce exactly across re-runs of the harness
    (the CLI threads a fixed ``--stats-seed`` into every call), and
  * the unit test can assert exact equality against a manual numpy reimplementation
    that replays the identical RNG call sequence (hence we do NOT use
    ``scipy.stats.bootstrap`` — its internal RNG threading is harder to pin).
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

__all__ = ["bootstrap_ci", "cvar", "percentile"]


def _finite_1d(samples: Sequence[float]) -> np.ndarray:
    """Flatten to a 1-D float array and drop non-finite (nan/inf) entries."""
    arr = np.asarray(list(samples), dtype=float).ravel()
    return arr[np.isfinite(arr)]


def bootstrap_ci(
    samples: Sequence[float],
    agg: Callable[..., float] = np.mean,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap ``(1 - alpha)`` confidence interval.

    Returns ``(point, lo, hi)`` where ``point = agg(samples)`` and ``lo/hi`` are the
    ``alpha/2`` / ``1 - alpha/2`` percentiles of the resampled aggregates.

    Non-finite samples are dropped first. With fewer than 2 finite samples the CI is
    degenerate and ``(point, point, point)`` is returned (``point`` is ``nan`` when
    there are zero finite samples).

    Determinism: resampling uses ``np.random.default_rng(seed)`` and a single
    ``rng.integers(0, n, size=(n_resamples, n))`` index matrix, then ``agg(.., axis=1)``.
    A caller replaying that exact sequence reproduces ``lo``/``hi`` bit-for-bit.
    """
    arr = _finite_1d(samples)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(agg(arr))
    if n < 2 or n_resamples < 1:
        return (point, point, point)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    resampled = arr[idx]  # (n_resamples, n)
    try:
        boot = np.asarray(agg(resampled, axis=1), dtype=float)
    except TypeError:
        # agg doesn't accept an axis kwarg — fall back to a per-row loop.
        boot = np.array([float(agg(row)) for row in resampled], dtype=float)

    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return (point, lo, hi)


def cvar(samples: Sequence[float], q: float = 0.95, tail: str = "upper") -> float:
    """Conditional Value at Risk — the mean of the worst tail.

    ``tail="upper"``: mean of values ``>=`` the ``q``-quantile. Use for quantities where
    HIGH is dangerous (e.g. per-episode cost integral): ``cvar95`` = mean of the worst
    (highest) 5% of episodes.

    ``tail="lower"``: mean of values ``<=`` the ``(1 - q)``-quantile. Use for quantities
    where LOW is dangerous (e.g. minimum human-robot separation): ``cvar95`` = mean of
    the worst (lowest) 5% of episodes.

    Quantiles use ``np.quantile`` (linear interpolation). Non-finite samples are dropped;
    an empty array returns ``nan``. Deterministic (no RNG).
    """
    arr = _finite_1d(samples)
    if arr.size == 0:
        return float("nan")
    if tail == "upper":
        thresh = float(np.quantile(arr, q))
        sel = arr[arr >= thresh]
    elif tail == "lower":
        thresh = float(np.quantile(arr, 1.0 - q))
        sel = arr[arr <= thresh]
    else:
        raise ValueError(f"tail must be 'upper' or 'lower', got {tail!r}")
    if sel.size == 0:  # pragma: no cover — degenerate (all-equal) guard
        return thresh
    return float(sel.mean())


def percentile(samples: Sequence[float], p: float) -> float:
    """``p``-th percentile (0-100). Non-finite dropped; empty -> ``nan``.

    For the dangerous lower tail of separation use a small ``p`` (e.g. ``p=1`` is the
    1st percentile — the separation only 1% of episodes drop below).
    """
    arr = _finite_1d(samples)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))
