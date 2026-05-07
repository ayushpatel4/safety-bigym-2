"""Binary safety reward labelling.

The single chokepoint that turns ``info["safety"]`` into the
``(r_safe, terminal)`` pair the SVF dataset stores.

v1 ships SSM-only (``use_pfl=False``) because PFL contact detection is broken
(``pfl_force_ratio`` is identically zero — see CLAUDE.md). Flip ``use_pfl=True``
once the contact bug is fixed to retrofit PFL into the same labels without
re-collecting.
"""

from typing import Tuple


def label_transition(info: dict, *, use_pfl: bool = False) -> Tuple[float, bool]:
    """Return ``(r_safe, terminal)`` for one environment step.

    ``r_safe = 0.0`` on a violation, ``1.0`` otherwise. ``terminal`` mirrors the
    violation flag — callers OR it with the env's own ``terminated|truncated``
    when writing the dataset's ``done`` field.
    """
    safety = info["safety"]
    violation = bool(safety["ssm_violation"])
    if use_pfl:
        violation = violation or bool(safety.get("pfl_violation", False))
    return (0.0 if violation else 1.0), violation
