"""Phase 3 per-step cost signal ``c_t``.

Computes the continuous cost the Phase 3 cost critic regresses on:

.. code-block:: text

    c_ssm = max(0, 1 - ssm_margin / d_buffer)     # 0 far away, 1 at violation
    c_pfl = max(0, pfl_force_ratio - 0.8)          # 0 until 80% of threshold
    c_t   = max(c_ssm, c_pfl)                       # worst-case across ISO criteria

Defined verbatim in ``UPDATED_PROJECT_PLAN.md:343-346``. Pure function so it can
be reused by the env adapter (per-env-step attachment to the TimeStep), the
P3.0d smoke verification script (diagnostic logging), and downstream P3.1+
training-loop integration (Bellman target for ``Q_c``).

The ``pfl_force_ratio`` term is currently identically zero across every cell
under the open PFL contact-detection bug (see CLAUDE.md). The plumbing is
forward-compatible: when the bug lands ``c_t`` automatically picks up the PFL
contribution without any changes here.
"""

from __future__ import annotations

from typing import Mapping

D_BUFFER_DEFAULT: float = 0.3
"""SSM buffer distance in metres. Cost activates when min_separation < d_buffer."""

PFL_RATIO_THRESHOLD_DEFAULT: float = 0.8
"""PFL force ratio threshold for cost activation (80% of ISO limit)."""


def compute_cost(
    safety_info: Mapping[str, float],
    *,
    d_buffer: float = D_BUFFER_DEFAULT,
    pfl_threshold: float = PFL_RATIO_THRESHOLD_DEFAULT,
) -> float:
    """Return ``c_t ∈ [0, 1]`` for the given per-step ``info["safety"]`` dict.

    Missing or NaN-typed fields are treated as zero contribution (e.g. the
    first reset transition before SSM is populated). The output is clipped
    to [0, 1] so downstream Q_c regression sees a bounded target.
    """
    if not safety_info:
        return 0.0

    ssm_margin = safety_info.get("ssm_margin")
    if ssm_margin is None:
        c_ssm = 0.0
    else:
        c_ssm = max(0.0, 1.0 - float(ssm_margin) / float(d_buffer))

    pfl_force_ratio = safety_info.get("pfl_force_ratio")
    if pfl_force_ratio is None:
        c_pfl = 0.0
    else:
        c_pfl = max(0.0, float(pfl_force_ratio) - float(pfl_threshold))

    return float(min(1.0, max(c_ssm, c_pfl)))


COST_FORMS = ("continuous", "binary")
"""E3.1 cost-signal forms selectable via ``env.safety.cost_form``.

The third E3.1 cell — ``fixed`` — is NOT a cost form: it disables the
Lagrangian entirely and applies a reward penalty in the env
(``env.safety.add_violation_penalty`` / ``violation_penalty``), so it never
reaches :func:`select_cost`.
"""


def select_cost(
    safety_info: Mapping[str, float],
    *,
    cost_form: str = "continuous",
    d_buffer: float = D_BUFFER_DEFAULT,
    pfl_threshold: float = PFL_RATIO_THRESHOLD_DEFAULT,
) -> float:
    """Return the per-step cost ``c_t`` for the requested E3.1 cost form.

    - ``"continuous"`` (default, headline): the graded :func:`compute_cost`
      value in ``[0, 1]``.
    - ``"binary"``: ``1.0`` iff the worst-case ISO 15066 SSM flag
      ``info["safety"]["ssm_violation"]`` is set, else ``0.0`` — the ablation
      baseline that strips the gradient richness from the cost while keeping the
      same ``[0, 1]`` range (so the cost critic's C51 support is unaffected).
    """
    if cost_form == "continuous":
        return compute_cost(
            safety_info, d_buffer=d_buffer, pfl_threshold=pfl_threshold
        )
    if cost_form == "binary":
        if not safety_info:
            return 0.0
        return float(bool(safety_info.get("ssm_violation", False)))
    raise ValueError(
        f"cost_form must be one of {COST_FORMS!r}; got {cost_form!r}"
    )


__all__ = [
    "compute_cost",
    "select_cost",
    "COST_FORMS",
    "D_BUFFER_DEFAULT",
    "PFL_RATIO_THRESHOLD_DEFAULT",
]
