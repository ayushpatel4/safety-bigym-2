"""Binary safety reward labelling.

The single chokepoint that turns ``info["safety"]`` into the
``(r_safe, terminal)`` pair the SVF dataset stores.

v2 (2026-05-16) — label by **geometric proximity** rather than ISO 15066 SSM.
ISO 15066's required separation distance is industrial-cell calibrated: at
kitchen-scale robot velocities the stopping-distance formula demands ≥5m
clearance, so every transition gets labelled unsafe and the dataset becomes
degenerate for collaborative manipulation. We use a near-contact bar
instead: any human-joint / robot-link pair closer than
``proximity_threshold`` is unsafe. ``ssm_margin`` is still computed by
:class:`ISO15066Wrapper` and logged in ``info["safety"]`` for ISO
traceability and as a continuous cost signal for the Phase 3 Lagrangian.

PFL contact-detection is still broken (``pfl_force_ratio`` identically zero —
see CLAUDE.md); ``use_pfl=False`` until that's fixed. The PFL term, when
enabled, OR's with the proximity violation — hard contact is unsafe
regardless of the geometric threshold.
"""

from typing import Tuple


def label_transition(
    info: dict,
    *,
    use_pfl: bool = False,
    proximity_threshold: float = 0.10,
) -> Tuple[float, bool]:
    """Return ``(r_safe, terminal)`` for one environment step.

    ``r_safe = 0.0`` on a violation, ``1.0`` otherwise. ``terminal`` mirrors
    the violation flag — callers OR it with the env's own
    ``terminated|truncated`` when writing the dataset's ``done`` field.

    ``proximity_threshold`` is the geometric near-contact bar (metres).
    Default 0.10 m matches close-cooperation HRI literature and leaves room
    for ~5cm BodySLAM noise under the noisy bodyslam mode.
    """
    safety = info["safety"]
    violation = float(safety["min_separation"]) < proximity_threshold
    if use_pfl:
        violation = violation or bool(safety.get("pfl_violation", False))
    return (0.0 if violation else 1.0), violation
