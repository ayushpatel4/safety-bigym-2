"""Model-based Control-Barrier-Function "directional dodge" safety filter.

Unlike the learned SVF filter (:class:`safety_bigym.filters.runtime_wrapper.SafetyFilterWrapper`
+ a :class:`~safety_bigym.filters.fallback.Fallback`), which *vetoes* a proposed action
when a critic Q-value drops below a threshold and substitutes a freeze/retreat fallback for
the WHOLE action, this filter is **always-on and minimally invasive**:

* It needs **no critic / snapshot** — the barrier is a closed-form geometric quantity.
* It only ever touches the floating-**base** X,Y target (action indices 0,1). The Z target,
  the base yaw (RZ), and the entire arm/gripper sub-action (indices 2..15) are passed
  through untouched, so the policy keeps doing the task while the base "dodges".
* It is a *dodge, not a freeze*: when the human gets within ``d_target`` of the robot base
  it pushes the absolute base target a small step along the away-direction
  ``(robot_xy - human_xy)``; when the human is far enough away (``h >= 0``) the proposed
  action is returned byte-for-byte unchanged (minimal intervention).

CBF formulation
---------------
Barrier function (we want ``h >= 0`` to hold; ``h < 0`` is the unsafe set):

    h(x) = sep - d_target,   sep = ||robot_xy - human_xy||

When ``h < 0`` we apply an exponential-CBF-style corrective step toward the safe set,
proportional to the violation depth and capped::

    push = clip(gain * (d_target - sep), 0, max_push)               # base case
    push += beta * max(0, dot(human_vel_xy, away_unit))  (if use_velocity)  # approach term

and offset the base target along the unit away-direction::

    a_out[0:2] = a[0:2] + away * push      (then clipped to the raw action bounds)

The optional velocity term anticipates an *approaching* human and dodges harder; it is a
no-op when the human is receding. ``away_unit = (robot_xy - human_xy)/sep`` points from the
human to the robot, so a human *chasing* the robot has ``dot(human_vel_xy, away_unit) > 0``
(separation shrinking). (The project-plan sketch wrote ``dot(vel, -away)``; that sign is a
RECEDING human, so we use the physically-correct ``+away_unit`` here.)

Frame/index assumptions (verified against bigym; identical to :class:`RetreatFallback`):
  - ``action[0:2]``                          = absolute base X,Y world-position target.
  - ``obs['proprioception_floating_base'][0:2]`` = current base world X,Y (qpos).
  - ``obs['human_pos_estimate'][0:3]``       = human pelvis world X,Y,Z.
  - ``obs['human_pos_estimate'][3:5]``       = human pelvis world X,Y velocity (m/s).
All share the world frame. ``human_pos_estimate`` is present iff ``obs_mode != "off"``.

Fail-safe behaviour (the filter is never worse than a pass-through):
  - missing/short ``human_pos_estimate`` (obs-mode off) -> return ``a`` unchanged, warn once.
  - degenerate direction (``sep < 1e-3``)               -> return ``a`` unchanged.
  - any non-finite value                                -> return ``a`` unchanged.

EE-retract ("flinch") variant
-----------------------------
:class:`CBFRetractFilter` is a sibling mode that keeps the floating base planted and
instead retracts the ARM end-effector along the away-direction when the human gets
close to the *hand* (the quantity ``min_separation`` actually measures). The barrier
is ``h = sep_ee - d_target`` with ``sep_ee = ||ee_pos - human_pos||`` (closest human
body to the EE); the capped Cartesian retract ``u*push`` is mapped to arm joint
targets through the EE Jacobian (``dq = dls_pinv(J_ee) @ (u*push)``). It needs the
env's MuJoCo model+data (EE pose, closest human body, EE Jacobian), supplied by
:func:`compute_ee_retract_state`; the math half is MuJoCo-free and unit-tested with a
synthetic Jacobian. Hypothesis: retracting only the arm increases EE-human separation
at less task-success cost than backing the whole base out of the workspace.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CBFDodgeFilter",
    "CBFRetractFilter",
    "compute_ee_retract_state",
    "SpeedScaleFilter",
    "compute_speed_scale_state",
]


class CBFDodgeFilter:
    """Geometric CBF directional-dodge filter (always-on, base-XY only).

    Parameters
    ----------
    action_space:
        The RAW (de-normalised) action :class:`gym.spaces.Box`; ``apply`` operates and
        clips in this space.
    d_target:
        Barrier offset (metres). The filter keeps ``sep >= d_target``; default 0.45 m,
        a margin above the 0.30 m proximity-violation threshold.
    gain:
        Exponential-CBF correction gain on the violation depth ``(d_target - sep)``.
    max_push:
        Per-step cap on the base-target offset magnitude (metres).
    use_velocity:
        If ``True``, add ``beta * approach_speed`` to ``push`` when the human is closing.
    beta:
        Weight on the approach-velocity term (ignored when ``use_velocity`` is ``False``).
    base_xy_idx:
        Action indices of the base X,Y target (default ``(0, 1)``).
    human_key / base_key:
        Obs dict keys for the human estimate and the robot base proprioception.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        d_target: float = 0.45,
        gain: float = 1.0,
        max_push: float = 0.15,
        use_velocity: bool = True,
        beta: float = 0.1,
        base_xy_idx: tuple = (0, 1),
        human_key: str = "human_pos_estimate",
        base_key: str = "proprioception_floating_base",
    ):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"CBFDodgeFilter expects a Box action_space; got "
                f"{type(action_space).__name__}"
            )
        self._low = action_space.low.astype(np.float32)
        self._high = action_space.high.astype(np.float32)
        self._shape = action_space.shape
        self.d_target = float(d_target)
        self.gain = float(gain)
        self.max_push = float(max_push)
        self.use_velocity = bool(use_velocity)
        self.beta = float(beta)
        self._ix, self._iy = int(base_xy_idx[0]), int(base_xy_idx[1])
        self._human_key = human_key
        self._base_key = base_key
        self._warned_no_human = False

    def _passthrough(self, proposed: np.ndarray, *, sep: float = float("nan"),
                     reason: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
        out = np.asarray(proposed, dtype=np.float32).copy()
        info = {
            "intervened": False,
            "h": float(sep - self.d_target) if np.isfinite(sep) else float("nan"),
            "sep": float(sep),
            "push": 0.0,
            "d_target": self.d_target,
        }
        if reason:
            info["reason"] = reason
        return out, info

    def apply(
        self,
        obs: Mapping[str, Any],
        raw_proposed: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return ``(raw_corrected, info)`` in the RAW action space.

        ``info`` carries ``intervened`` (bool), the barrier ``h``, the separation
        ``sep``, the applied ``push`` magnitude, and ``d_target`` — mirroring the SVF
        veto bookkeeping so the runner can count interventions.
        """
        proposed = np.asarray(raw_proposed, dtype=np.float32)

        human = obs.get(self._human_key) if hasattr(obs, "get") else None
        base = obs.get(self._base_key) if hasattr(obs, "get") else None

        if human is None or base is None:
            if not self._warned_no_human:
                logger.warning(
                    "CBFDodgeFilter inactive: obs is missing %r and/or %r (obs-mode "
                    "is likely 'off'). Passing actions through unchanged.",
                    self._human_key, self._base_key,
                )
                self._warned_no_human = True
            return self._passthrough(proposed, reason="missing_obs")

        human = np.asarray(human, dtype=np.float32).ravel()
        base = np.asarray(base, dtype=np.float32).ravel()
        if human.size < 2 or base.size < 2:
            if not self._warned_no_human:
                logger.warning(
                    "CBFDodgeFilter inactive: %r/%r too short (sizes %d/%d). Passing "
                    "actions through unchanged.",
                    self._human_key, self._base_key, human.size, base.size,
                )
                self._warned_no_human = True
            return self._passthrough(proposed, reason="short_obs")

        human_xy = human[:2]
        base_xy = base[:2]
        away = base_xy - human_xy
        sep = float(np.linalg.norm(away))

        if not np.isfinite(sep):
            return self._passthrough(proposed, reason="nonfinite")

        h = sep - self.d_target
        # Outside the barrier (safe) -> minimal intervention: pass through unchanged.
        if h >= 0.0:
            return self._passthrough(proposed, sep=sep)
        # Degenerate direction: away vector undefined -> can't pick a dodge direction.
        if sep < 1e-3:
            return self._passthrough(proposed, sep=sep, reason="degenerate")

        away_unit = away / sep

        push = float(np.clip(self.gain * (self.d_target - sep), 0.0, self.max_push))

        if self.use_velocity and human.size >= 5:
            human_vel_xy = human[3:5]
            # away_unit points human -> robot. With the robot ~static over a control
            # step, d(sep)/dt = -dot(human_vel, away_unit); the human is APPROACHING
            # (separation shrinking) when dot(human_vel, away_unit) > 0 — i.e. its
            # velocity chases the robot. Add an anticipatory dodge proportional to that
            # closing speed. (NB: this is the sign that is physically correct for
            # sep = ||robot - human||; the project-plan sketch wrote dot(vel, -away),
            # which corresponds to a RECEDING human and is not what we want.)
            approach_speed = float(np.dot(human_vel_xy, away_unit))
            if np.isfinite(approach_speed) and approach_speed > 0.0:
                push += self.beta * approach_speed
        push = float(np.clip(push, 0.0, self.max_push))

        out = proposed.copy()
        out[self._ix] = proposed[self._ix] + away_unit[0] * push
        out[self._iy] = proposed[self._iy] + away_unit[1] * push
        out = np.clip(out, self._low, self._high).astype(np.float32)

        info = {
            "intervened": True,
            "h": float(h),
            "sep": float(sep),
            "push": float(push),
            "d_target": self.d_target,
        }
        return out, info


def _damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """Damped least-squares (Levenberg-Marquardt) pseudo-inverse of ``J`` (m x n).

    ``J^+ = J^T (J J^T + lambda^2 I_m)^-1`` — the right-pinv form, which is the
    minimum-norm solver for the under-determined Cartesian->joint map and never
    blows up near a kinematic singularity (the bare Moore-Penrose pinv does).
    ``m`` (the task dim) is 3 here, so the inverted matrix is a tiny 3x3.
    """
    J = np.asarray(J, dtype=np.float64)
    m = J.shape[0]
    lam2 = float(damping) ** 2
    JJt = J @ J.T + lam2 * np.eye(m)
    return J.T @ np.linalg.inv(JJt)


class CBFRetractFilter:
    """Geometric CBF *EE-retract ("flinch")* filter — arm-only, base untouched.

    Sibling of :class:`CBFDodgeFilter`. Where the dodge filter pushes the floating
    **base** X,Y away from the human (which works but *flees* the workspace, costing
    task success), this filter keeps the base planted and instead **retracts the arm
    end-effector** in Cartesian space along the away-direction when the human gets
    too close to the *hand* — the quantity ``min_separation`` actually measures.

    Barrier (we want ``h >= 0``):

        h = sep_ee - d_target,    sep_ee = ||ee_pos - human_pos||

    where ``human_pos`` is the closest tracked human body to the EE. When ``h < 0``
    we move the EE along the unit away-direction ``u = (ee_pos - human_pos)/sep_ee``
    by a capped step ``push = clip(gain*(d_target - sep_ee), 0, max_push)`` and map
    that Cartesian retract to ARM joint targets through the EE Jacobian::

        dq = dls_pinv(J_ee) @ (u * push)          # damped least-squares
        a_out[arm_idx] = arm_qpos + dq            # absolute joint targets

    Only the arm action indices change; base (X,Y,Z,RZ) and grippers pass through.
    Because the action mode is *absolute* joint position, ``arm_qpos + dq`` commands
    the arm to step ``dq`` from its current pose — i.e. the policy's arm command is
    overridden for the duration of the flinch (it resumes the moment ``h >= 0``).

    This is the *pure-math* half: the EE position, the closest human body position,
    the EE Jacobian (restricted to the arm DOF columns) and the current arm qpos are
    all supplied in the ``state`` dict by :func:`compute_ee_retract_state`, so this
    class needs **no MuJoCo** and is unit-testable with a synthetic Jacobian.

    Fail-safe (never worse than a pass-through): missing/ill-shaped state, a
    non-finite value, ``sep_ee >= d_target`` (safe), or a degenerate away-direction
    all return the proposed action unchanged.

    Parameters
    ----------
    action_space:
        The RAW (de-normalised) action :class:`gym.spaces.Box`; ``apply`` clips into it.
    d_target / gain / max_push:
        Barrier offset (m), CBF correction gain on the depth, and per-step Cartesian
        retract cap (m) — same roles as :class:`CBFDodgeFilter`.
    damping:
        Damped-least-squares lambda for the Jacobian pseudo-inverse (rad-ish units).
    state_key:
        Key under which the env-state dict is looked up when ``apply`` is handed the
        full obs mapping (the runner passes the state dict directly, so this is mostly
        for symmetry / debugging).
    """

    # state-dict fields produced by compute_ee_retract_state.
    _EE = "ee_pos"
    _HUMAN = "human_pos"
    _JARM = "J_arm"
    _QARM = "arm_qpos"
    _IDX = "arm_action_idx"

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        d_target: float = 0.45,
        gain: float = 1.0,
        max_push: float = 0.15,
        damping: float = 0.05,
        state_key: str = "ee_retract_state",
    ):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"CBFRetractFilter expects a Box action_space; got "
                f"{type(action_space).__name__}"
            )
        self._low = action_space.low.astype(np.float32)
        self._high = action_space.high.astype(np.float32)
        self._shape = action_space.shape
        self.d_target = float(d_target)
        self.gain = float(gain)
        self.max_push = float(max_push)
        self.damping = float(damping)
        self._state_key = state_key
        self._warned_no_state = False

    def _passthrough(self, proposed: np.ndarray, *, sep: float = float("nan"),
                     reason: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
        out = np.asarray(proposed, dtype=np.float32).copy()
        info = {
            "intervened": False,
            "h": float(sep - self.d_target) if np.isfinite(sep) else float("nan"),
            "sep": float(sep),
            "push": 0.0,
            "d_target": self.d_target,
            "mode": "ee",
        }
        if reason:
            info["reason"] = reason
        return out, info

    @staticmethod
    def _extract_state(obs_or_state: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Return the state dict — the runner hands it directly, but tolerate a
        nested ``obs['ee_retract_state']`` too."""
        if obs_or_state is None:
            return None
        if hasattr(obs_or_state, "get") and obs_or_state.get("ee_retract_state") is not None:
            return obs_or_state.get("ee_retract_state")
        return obs_or_state

    def apply(
        self,
        state: Mapping[str, Any],
        raw_proposed: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return ``(raw_corrected, info)`` in the RAW action space.

        ``state`` carries ``ee_pos`` (3,), ``human_pos`` (3,), ``J_arm`` (3,N),
        ``arm_qpos`` (N,) and ``arm_action_idx`` (N,) — see
        :func:`compute_ee_retract_state`. ``info`` mirrors :class:`CBFDodgeFilter`'s
        veto bookkeeping (``intervened`` / ``h`` / ``sep`` / ``push`` / ``d_target``)
        so the runner counts interventions identically.
        """
        proposed = np.asarray(raw_proposed, dtype=np.float32)
        st = self._extract_state(state)

        required = (self._EE, self._HUMAN, self._JARM, self._QARM, self._IDX)
        if st is None or any(
            (st.get(k) if hasattr(st, "get") else None) is None for k in required
        ):
            if not self._warned_no_state:
                logger.warning(
                    "CBFRetractFilter inactive: env-state is missing one of %r "
                    "(EE pose / closest human / Jacobian unavailable). Passing "
                    "actions through unchanged.", list(required),
                )
                self._warned_no_state = True
            return self._passthrough(proposed, reason="missing_state")

        ee = np.asarray(st[self._EE], dtype=np.float64).ravel()
        human = np.asarray(st[self._HUMAN], dtype=np.float64).ravel()
        J = np.asarray(st[self._JARM], dtype=np.float64)
        qarm = np.asarray(st[self._QARM], dtype=np.float64).ravel()
        idx = np.asarray(st[self._IDX], dtype=int).ravel()

        # Shape / consistency guards -> pass-through (never crash a rollout).
        n = idx.size
        if ee.size != 3 or human.size != 3 or J.ndim != 2 or J.shape[0] != 3 \
                or J.shape[1] != n or qarm.size != n or n == 0:
            return self._passthrough(proposed, reason="bad_shape")
        if np.any(idx < 0) or np.any(idx >= proposed.size):
            return self._passthrough(proposed, reason="bad_idx")
        if not (np.all(np.isfinite(ee)) and np.all(np.isfinite(human))
                and np.all(np.isfinite(J)) and np.all(np.isfinite(qarm))):
            return self._passthrough(proposed, reason="nonfinite")

        away = ee - human
        sep = float(np.linalg.norm(away))
        if not np.isfinite(sep):
            return self._passthrough(proposed, reason="nonfinite")

        h = sep - self.d_target
        if h >= 0.0:  # safe -> minimal intervention.
            return self._passthrough(proposed, sep=sep)
        if sep < 1e-6:  # degenerate: away-direction undefined.
            return self._passthrough(proposed, sep=sep, reason="degenerate")

        u = away / sep
        push = float(np.clip(self.gain * (self.d_target - sep), 0.0, self.max_push))
        if push <= 0.0:
            return self._passthrough(proposed, sep=sep)

        v = u * push  # desired Cartesian EE displacement (3,)
        dq = _damped_pinv(J, self.damping) @ v  # (N,)
        if not np.all(np.isfinite(dq)):
            return self._passthrough(proposed, sep=sep, reason="nonfinite_dq")

        out = proposed.copy()
        out[idx] = (qarm + dq).astype(np.float32)
        out = np.clip(out, self._low, self._high).astype(np.float32)

        info = {
            "intervened": True,
            "h": float(h),
            "sep": float(sep),
            "push": float(push),
            "d_target": self.d_target,
            "dq_norm": float(np.linalg.norm(dq)),
            "mode": "ee",
        }
        return out, info


class SpeedScaleFilter:
    """ISO-15066 SSM **speed-scaling** filter — slow the robot near the human.

    Where :class:`CBFDodgeFilter` / :class:`CBFRetractFilter` are *position* filters
    that fight geometric proximity (push the base away / retract the arm — both
    *flee* the workspace), this filter operates on ISO-15066's **velocity axis**: it
    leaves the *direction* of the policy's commanded motion untouched and only scales
    its *magnitude* down as the human gets close, in direct proportion to separation.
    The robot keeps doing the task on the same trajectory, just slower near a person,
    which is what ISO-15066 Speed-and-Separation-Monitoring actually asks for. It is
    expected to cut ``ep_ssm_violation_actual_rate`` (the velocity-adaptive ISO
    margin) and mean robot velocity at roughly unchanged geometric proximity and a
    modest success cost.

    Speed-scaling law
    -----------------
    Given the proposed RAW action ``a``, the robot's CURRENT joint/base positions
    ``q_cur`` (at the same action indices), and the closest human<->robot separation
    ``sep``::

        scale = clip((sep - d_stop) / (d_slow - d_stop), 0, 1)
        a_out[i] = q_cur[i] + scale * (a[i] - q_cur[i])     # per-dim motion scaling

    so ``sep >= d_slow -> scale = 1`` (full speed, byte-for-byte pass-through),
    ``sep <= d_stop -> scale = 0`` (hold the current position / zero motion), and in
    between a graded slowdown. Because the action mode is *absolute* for the arm,
    ``q_cur[i] + scale*(a[i]-q_cur[i])`` scales the per-step *motion* from the current
    pose; for the floating base — which BiGym always drives in *delta* (incremental)
    mode regardless of ``absolute=`` — the natural "current" of a delta command is
    zero motion, so :func:`compute_speed_scale_state` reports ``q_cur = 0`` for the
    base dims and the same formula collapses to ``a_out = scale * a`` (the delta is
    scaled directly). Action dims with no known current position (e.g. grippers,
    reported as NaN in ``q_cur``) pass through unchanged.

    Fail-safe (never worse than a pass-through; never crash a rollout): missing/ill-
    shaped state, any non-finite ``sep``/``q_cur``, or ``scale >= 1`` (human far
    enough away) all return the proposed action unchanged with ``intervened=False``.

    This is the *pure-math* half — ``q_cur`` and ``sep`` are supplied in the
    ``state`` dict by :func:`compute_speed_scale_state` from a live env, so this class
    needs **no MuJoCo** and is unit-testable with synthetic state.

    Parameters
    ----------
    action_space:
        The RAW (de-normalised) action :class:`gym.spaces.Box`; ``apply`` scales and
        clips in this space.
    d_slow:
        Separation (m) at/above which the robot runs at full speed (``scale = 1``).
        Default 0.5 m.
    d_stop:
        Separation (m) at/below which the robot holds position (``scale = 0``).
        Default 0.15 m. Must be ``< d_slow``.
    """

    # state-dict fields produced by compute_speed_scale_state.
    _QCUR = "q_cur"
    _SEP = "sep"

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        d_slow: float = 0.5,
        d_stop: float = 0.15,
        state_key: str = "speed_scale_state",
    ):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"SpeedScaleFilter expects a Box action_space; got "
                f"{type(action_space).__name__}"
            )
        if not (float(d_slow) > float(d_stop)):
            raise ValueError(
                f"SpeedScaleFilter requires d_slow > d_stop; got "
                f"d_slow={d_slow}, d_stop={d_stop}."
            )
        self._low = action_space.low.astype(np.float32)
        self._high = action_space.high.astype(np.float32)
        self._shape = action_space.shape
        self.d_slow = float(d_slow)
        self.d_stop = float(d_stop)
        self._state_key = state_key
        self._warned_no_state = False

    def _passthrough(self, proposed: np.ndarray, *, scale: float = 1.0,
                     sep: float = float("nan"), reason: str = "",
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
        out = np.asarray(proposed, dtype=np.float32).copy()
        info = {
            "intervened": False,
            "scale": float(scale),
            "sep": float(sep),
            "d_slow": self.d_slow,
            "d_stop": self.d_stop,
            "mode": "speedscale",
        }
        if reason:
            info["reason"] = reason
        return out, info

    @staticmethod
    def _extract_state(obs_or_state: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Return the state dict — the runner hands it directly, but tolerate a
        nested ``obs['speed_scale_state']`` too."""
        if obs_or_state is None:
            return None
        if hasattr(obs_or_state, "get") and obs_or_state.get("speed_scale_state") is not None:
            return obs_or_state.get("speed_scale_state")
        return obs_or_state

    def apply(
        self,
        state: Mapping[str, Any],
        raw_proposed: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return ``(raw_scaled, info)`` in the RAW action space.

        ``state`` carries ``q_cur`` (action-dim,) — the current joint/base position
        at each action index (NaN for dims with no counterpart, e.g. grippers) — and
        ``sep`` (float) — the closest human<->robot separation; see
        :func:`compute_speed_scale_state`. ``info`` mirrors the other filters'
        bookkeeping (``intervened`` / ``sep`` / ...) plus the applied ``scale`` so the
        runner counts interventions identically.
        """
        proposed = np.asarray(raw_proposed, dtype=np.float32)
        st = self._extract_state(state)

        required = (self._QCUR, self._SEP)
        if st is None or any(
            (st.get(k) if hasattr(st, "get") else None) is None for k in required
        ):
            if not self._warned_no_state:
                logger.warning(
                    "SpeedScaleFilter inactive: env-state is missing one of %r "
                    "(current joint positions / separation unavailable). Passing "
                    "actions through unchanged.", list(required),
                )
                self._warned_no_state = True
            return self._passthrough(proposed, reason="missing_state")

        q_cur = np.asarray(st[self._QCUR], dtype=np.float64).ravel()
        sep = float(st[self._SEP])

        # Shape guard -> pass-through (never mis-index / crash a rollout).
        if q_cur.size != proposed.size:
            return self._passthrough(proposed, reason="bad_shape")
        if not np.isfinite(sep):
            return self._passthrough(proposed, reason="nonfinite")

        denom = self.d_slow - self.d_stop  # > 0 (enforced in __init__)
        scale = float(np.clip((sep - self.d_stop) / denom, 0.0, 1.0))

        # Human far enough away -> full speed -> minimal intervention (pass-through).
        if scale >= 1.0:
            return self._passthrough(proposed, scale=1.0, sep=sep)

        # Scale the per-dim motion toward q_cur. Dims with no known current position
        # (NaN in q_cur) pass through unchanged so we never inject motion of our own.
        out = proposed.copy()
        finite = np.isfinite(q_cur)
        if np.any(finite):
            qf = q_cur[finite].astype(np.float32)
            out[finite] = (qf + scale * (proposed[finite] - qf)).astype(np.float32)
        out = np.clip(out, self._low, self._high).astype(np.float32)

        info = {
            "intervened": True,
            "scale": float(scale),
            "sep": float(sep),
            "d_slow": self.d_slow,
            "d_stop": self.d_stop,
            "mode": "speedscale",
        }
        return out, info


# ---------------------------------------------------------------------------
# Env-state extraction (MuJoCo) — the "hard part" half of the EE-retract filter.
# Only exercised on the GPU box (needs MuJoCo + a live SafetyBiGymEnv); the math
# above is tested independently with a synthetic Jacobian.
# ---------------------------------------------------------------------------

def _unwrap_safety_env(env):
    """Walk gym .env / .unwrapped links down to the raw SafetyBiGymEnv.

    The adapter's ``_env`` at filter time is
    ``ObsCacheWrapper(EpisodeSafetyMetrics(BodySLAMWrapper(SafetyBiGymEnv)))`` (all
    gym.Wrapper), so ``.unwrapped`` reaches the base env. Returns the first object
    in the chain exposing ``_get_robot_state`` (the SafetyBiGymEnv), else None.
    """
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "_get_robot_state") and hasattr(cur, "_human_ssm_state"):
            return cur
        nxt = getattr(cur, "env", None)
        if nxt is None:
            nxt = getattr(cur, "unwrapped", None)
            if nxt is cur:
                nxt = None
        cur = nxt
    # last resort: gym's own .unwrapped.
    base = getattr(env, "unwrapped", None)
    if base is not None and hasattr(base, "_get_robot_state"):
        return base
    return None


def _ee_body_id(env, model) -> int:
    """MuJoCo body id of the robot EE, matching the body whose origin
    :meth:`SafetyBiGymEnv._get_robot_state` reports as ``link_pos['ee']``.

    Uses the env's own ``_ROBOT_LINK_NAMES['ee']`` candidate list so the EE
    *position* and the EE *Jacobian* are taken at the same body frame.
    """
    import mujoco

    candidates = getattr(type(env), "_ROBOT_LINK_NAMES", {}).get("ee", [])
    for name in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    return -1


def _arm_joint_action_map(env, model) -> Tuple[List[int], List[int], List[int]]:
    """Map the robot's ARM action indices to MuJoCo (dof, qpos) addresses.

    The :class:`JointPositionActionMode` action layout is
    ``[floating-base dofs] + [limb_actuators...] + [grippers]``. Each limb actuator
    binds a single hinge joint; the grippers are *not* limb actuators, so iterating
    ``robot.limb_actuators`` naturally yields exactly the arm joints (10 for H1) and
    excludes the two gripper channels. The action index of limb-actuator ``i`` is
    ``base_dof_amount + i``.

    Returns ``(action_idx, dof_idx, qpos_idx)`` aligned lists. Joints that can't be
    resolved (no ``.joint`` / not found in the compiled model) are skipped, so the
    filter degrades gracefully rather than mis-indexing.
    """
    import mujoco

    robot = getattr(env, "_robot", None)
    if robot is None:
        return [], [], []
    fb = getattr(robot, "floating_base", None)
    base_n = int(getattr(fb, "dof_amount", 0)) if fb is not None else 0

    action_idx: List[int] = []
    dof_idx: List[int] = []
    qpos_idx: List[int] = []
    for i, actuator in enumerate(robot.limb_actuators):
        joint = getattr(actuator, "joint", None)
        if joint is None:
            continue
        jname = getattr(joint, "full_identifier", None) or getattr(joint, "name", None)
        if not jname:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        action_idx.append(base_n + i)
        dof_idx.append(int(model.jnt_dofadr[jid]))
        qpos_idx.append(int(model.jnt_qposadr[jid]))
    return action_idx, dof_idx, qpos_idx


def compute_ee_retract_state(env) -> Optional[Dict[str, Any]]:
    """Extract everything :class:`CBFRetractFilter` needs from a live env.

    Returns a dict with ``ee_pos`` (3,), ``human_pos`` (3,) — the closest tracked
    human body to the EE, ``J_arm`` (3,N) — the EE translational Jacobian restricted
    to the arm DOF columns, ``arm_qpos`` (N,), and ``arm_action_idx`` (N,). Returns
    ``None`` (filter then passes through) when the env handle, EE, human bodies, or
    arm mapping can't be resolved.

    All quantities are read at the *current* ``data`` (the obs the policy just acted
    on), so the retract is computed against the same state the agent saw.
    """
    import mujoco

    senv = _unwrap_safety_env(env)
    if senv is None:
        return None
    mojo = getattr(senv, "_mojo", None)
    if mojo is None:
        return None
    model, data = mojo.model, mojo.data

    # 1. EE position + body id (same body frame for pos and Jacobian).
    state = senv._get_robot_state()
    ee_pos = state.get("ee_pos")
    if ee_pos is None:
        ee_pos = (state.get("link_pos") or {}).get("ee")
    if ee_pos is None:
        return None
    ee_pos = np.asarray(ee_pos, dtype=float).reshape(3)
    ee_bid = _ee_body_id(senv, model)
    if ee_bid < 0:
        return None

    # 2. Closest tracked human body to the EE.
    human_positions, _names, _vel = senv._human_ssm_state()
    human_positions = np.atleast_2d(np.asarray(human_positions, dtype=float))
    if human_positions.size == 0 or human_positions.shape[-1] != 3:
        return None
    d = np.linalg.norm(human_positions - ee_pos[None, :], axis=1)
    human_pos = human_positions[int(np.argmin(d))]

    # 3. Arm action <-> (dof, qpos) map.
    action_idx, dof_idx, qpos_idx = _arm_joint_action_map(senv, model)
    if not action_idx:
        return None

    # 4. EE translational Jacobian (3 x nv), then select the arm DOF columns.
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, None, ee_bid)
    J_arm = jacp[:, dof_idx]
    arm_qpos = np.asarray(data.qpos, dtype=float)[qpos_idx]

    return {
        "ee_pos": ee_pos,
        "human_pos": np.asarray(human_pos, dtype=float).reshape(3),
        "J_arm": J_arm,
        "arm_qpos": arm_qpos,
        "arm_action_idx": np.asarray(action_idx, dtype=int),
    }


def _min_human_robot_separation(senv) -> Optional[float]:
    """Closest human-body <-> robot-link distance from a live SafetyBiGymEnv.

    Reuses the env's own SSM-state readers (:meth:`_robot_ssm_state` /
    :meth:`_human_ssm_state`) and returns the minimum pairwise distance — the exact
    quantity ``info["safety"]["min_separation"]`` reports, computed live against the
    obs the policy just acted on. Returns ``None`` if either set is unavailable
    (filter then passes through). Wrapped in a broad try so a reader hiccup never
    crashes a rollout.
    """
    try:
        robot_positions, _rn, _rv = senv._robot_ssm_state()
        human_positions, _hn, _hv = senv._human_ssm_state()
    except Exception:  # pragma: no cover — best-effort, never crash the rollout
        return None
    robot_arr = np.atleast_2d(np.asarray(robot_positions, dtype=float))
    human_arr = np.atleast_2d(np.asarray(human_positions, dtype=float))
    if robot_arr.size == 0 or human_arr.size == 0:
        return None
    if robot_arr.shape[-1] != 3 or human_arr.shape[-1] != 3:
        return None
    diff = human_arr[:, None, :] - robot_arr[None, :, :]
    d_min = float(np.linalg.norm(diff, axis=-1).min())
    return d_min if np.isfinite(d_min) else None


def compute_speed_scale_state(env) -> Optional[Dict[str, Any]]:
    """Extract everything :class:`SpeedScaleFilter` needs from a live env.

    Returns a dict with ``sep`` (float) — the closest human<->robot separation — and
    ``q_cur`` (action-dim,) — the robot's current position counterpart at every action
    index:

    * **Arm dims** (limb actuators, absolute-target mode): the current joint ``qpos``,
      so scaling moves the per-step motion ``q_cur + scale*(a - q_cur)``.
    * **Floating-base dims** (X, Y, Z, RZ — always *delta* mode in BiGym): ``0.0``. A
      delta command's "no motion" reference is zero, so the speed-scaling formula
      collapses to ``scale * a`` and directly scales the commanded base step.
    * **Everything else** (e.g. grippers): ``NaN`` -> the filter passes those dims
      through unchanged (we never scale a grip command toward an unknown reference).

    Returns ``None`` (filter then passes through) when the env handle, the action-dim,
    or the separation can't be resolved. All quantities are read at the *current*
    ``data`` (the obs the policy just acted on).
    """
    senv = _unwrap_safety_env(env)
    if senv is None:
        return None
    mojo = getattr(senv, "_mojo", None)
    if mojo is None:
        return None
    model, data = mojo.model, mojo.data

    # Action dimension: prefer the (possibly filter-wrapped) handle's action_space so
    # q_cur lines up with the raw action the filter scales; fall back to the base env.
    action_space = getattr(env, "action_space", None) or getattr(senv, "action_space", None)
    if action_space is None or not hasattr(action_space, "shape") or not action_space.shape:
        return None
    action_dim = int(np.prod(action_space.shape))
    if action_dim <= 0:
        return None

    # q_cur: NaN everywhere (pass-through), then fill base (=0, delta mode) + arm (qpos).
    q_cur = np.full(action_dim, np.nan, dtype=np.float64)

    robot = getattr(senv, "_robot", None)
    fb = getattr(robot, "floating_base", None) if robot is not None else None
    base_n = int(getattr(fb, "dof_amount", 0)) if fb is not None else 0
    if base_n > 0:
        base_n = min(base_n, action_dim)
        q_cur[:base_n] = 0.0  # delta-mode base: "no motion" reference is 0.

    action_idx, _dof_idx, qpos_idx = _arm_joint_action_map(senv, model)
    qpos = np.asarray(data.qpos, dtype=float)
    for a_i, q_i in zip(action_idx, qpos_idx):
        if 0 <= a_i < action_dim and 0 <= q_i < qpos.size:
            q_cur[a_i] = qpos[q_i]

    sep = _min_human_robot_separation(senv)
    if sep is None:
        return None

    return {
        "q_cur": q_cur,
        "sep": float(sep),
    }
