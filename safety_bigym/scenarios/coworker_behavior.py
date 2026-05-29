"""
Coworker Disruption Behaviour

Implements the per-step IK callback for `DisruptionType.COWORKER`.

The human stays parked in the robot's workspace and cycles through a
four-phase reach pattern with the active arm:

    [0,        t_extend)                           -> EXTEND  (blend rest -> target)
    [t_extend, t_extend + t_hold)                  -> HOLD    (held at target)
    [.., t_extend + t_hold + t_retract)            -> RETRACT (blend target -> rest)
    [.., T)                                         -> IDLE   (rest pose)

At the start of each period (and on the first call) the controller
samples which target to reach for: robot end-effector or task object,
weighted by ``DisruptionConfig.coworker_target_mix``. The non-reaching
arm, legs, and torso are held at a constant standing rest pose, so the
callback does not depend on AMASS clip availability during loiter.

The callback returns a full-sized ``qpos`` buffer; HumanController's
existing IK-blend path consumes it the same way it consumes
:class:`HumanIK` results for the other disruption types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import mujoco
import numpy as np

from safety_bigym.human.human_ik import HumanIK
from safety_bigym.human import g1_spec
from safety_bigym.scenarios.disruption_types import DisruptionConfig
from safety_bigym.scenarios.scenario_sampler import ScenarioParams


# Phase labels for the per-cycle state machine. Strings (not an Enum)
# so they are trivial to assert against in tests / log lines.
PHASE_EXTEND = "extend"
PHASE_HOLD = "hold"
PHASE_RETRACT = "retract"
PHASE_IDLE = "idle"


@dataclass
class CycleState:
    """Bookkeeping for the current reach cycle."""

    cycle_index: int = -1               # which cycle we last initialised
    target_kind: str = "ee"             # "ee" | "task_object"
    target_pos: Optional[np.ndarray] = None  # cached at cycle start


class CoworkerArmController:
    """
    Stateful per-episode procedural arm controller for ``COWORKER``.

    Parameters
    ----------
    model, data
        Live MuJoCo model/data of the env. Used to read joint indices
        and to drive the embedded ``HumanIK`` solver.
    scenario
        Active scenario; ``scenario.disruption_config`` supplies the
        cycle timing and target mix.
    rng
        Per-episode RNG so the reach-target schedule is reproducible
        given the env seed.
    rest_qpos
        Optional full-sized qpos buffer for the rest pose. If ``None``,
        defaults to all zeros (matches HumanController's
        ``_standing_pose`` fallback).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scenario: ScenarioParams,
        rng: np.random.Generator,
        rest_qpos: Optional[np.ndarray] = None,
        ik_solver: Optional[object] = None,
    ):
        self.model = model
        self.data = data
        self.scenario = scenario
        self.rng = rng

        cfg = scenario.disruption_config
        if cfg is None:
            raise ValueError(
                "CoworkerArmController requires scenario.disruption_config to be set"
            )
        self.cfg: DisruptionConfig = cfg

        # Accept any IK solver that exposes ``HumanIK``'s interface
        # (``solve``, ``chains``, ``_chain_cache``, ``_ik_data``). The SMPL-H
        # ``HumanIK`` is the default; ``G1HumanIK`` is duck-compatible.
        self.ik_solver = ik_solver if ik_solver is not None else HumanIK(model, data)

        if rest_qpos is None:
            # Build a natural arms-down standing pose by IK-solving each
            # arm to a point ~0.5 m below its shoulder. This ensures the
            # RETRACT phase moves the wrist *down* to the side, not
            # sideways across the front of the body where the robot's
            # workspace is.
            self._rest_qpos = self._build_arms_down_rest_pose()
        else:
            self._rest_qpos = np.asarray(rest_qpos, dtype=float).copy()
            if self._rest_qpos.shape[0] != self.model.nq:
                raise ValueError(
                    f"rest_qpos has nq={self._rest_qpos.shape[0]}, "
                    f"expected {self.model.nq}"
                )

        # The active arm's qpos indices, cached so we can blend a
        # fraction of the IK result over the rest pose.
        self._arm_qpos_indices: list[int] = list(
            self.ik_solver._chain_cache[self._active_arm()]["qpos_indices"]
        )

        # The "rest" angles for the arm DoFs we overwrite. Pulled from
        # the rest pose so the retract phase blends cleanly back to
        # arm-hanging-at-side.
        self._arm_rest = np.array(
            [self._rest_qpos[i] for i in self._arm_qpos_indices]
        )

        # Geometric reach gate: when the active shoulder is further from
        # the target than this, suppress the reach (return rest pose).
        # ~0.75 m covers a full-extension SMPL-H arm plus a small lean;
        # patrol-away positions sit well beyond this so the arm hangs
        # at the side while the human is far from the robot.
        self._max_reach_dist: float = 0.75

        # Cache the active arm's shoulder body id so the reach gate can
        # query world position cheaply each step. Name comes from the IK
        # solver's chain definition so SMPL-H and G1 (with different body
        # naming) work without dispatch here.
        shoulder_name = self.ik_solver.chains[self._active_arm()]["shoulder_body"]
        self._active_shoulder_bid = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name)
        )

        self._cycle = CycleState()
        # Last computed reach target (world coords) for visualisation /
        # tests. None until at least one extend/hold/retract step ran.
        self.last_reach_target: Optional[np.ndarray] = None
        # Most recent phase string, also for visualisation / tests.
        self.last_phase: str = PHASE_IDLE
        # Tracks whether the most recent compute_qpos call decided the
        # target was out of reach. Exposed for the demo / tests.
        self.last_out_of_reach: bool = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_callback(self) -> Callable[[dict], np.ndarray]:
        """Return a ``_ik_target_callback``-compatible closure.

        The closure captures ``self``; the actual time-keeping lives on
        ``HumanController.t`` and is read via the ``robot_state`` dict
        through ``robot_state['t']`` *if available*, falling back to
        ``data.time`` otherwise.
        """

        def _callback(robot_state: dict) -> np.ndarray:
            t = float(robot_state.get("t", self.data.time))
            return self.compute_qpos(t, robot_state)

        return _callback

    # ------------------------------------------------------------------
    # Core state machine
    # ------------------------------------------------------------------

    def compute_qpos(self, t: float, robot_state: dict) -> np.ndarray:
        """Compute the full qpos target for time ``t``.

        Public method (rather than just a closure) so tests can drive
        the controller without going through the HumanController.
        """
        period = max(float(self.cfg.coworker_reach_period), 1e-3)
        cycle_index = int(t // period)
        phase_t = t - cycle_index * period

        f_extend = float(self.cfg.coworker_reach_fraction)
        f_hold = float(self.cfg.coworker_hold_fraction)
        f_retract = float(self.cfg.coworker_retract_fraction)
        # idle fraction = remainder; computed implicitly below.

        t_extend_end = f_extend * period
        t_hold_end = (f_extend + f_hold) * period
        t_retract_end = (f_extend + f_hold + f_retract) * period

        # On entering a new cycle, sample the next reach target so the
        # human alternates EE/task-object across cycles.
        if cycle_index != self._cycle.cycle_index:
            self._sample_new_target(cycle_index, robot_state)

        # Resolve the cycle target each call so it tracks moving robots /
        # task objects during HOLD (otherwise the arm would freeze at a
        # stale point and the SSM margin would lie).
        target_pos = self._resolve_target(robot_state)
        if target_pos is not None:
            self._cycle.target_pos = target_pos
            self.last_reach_target = target_pos.copy()

        if phase_t < t_extend_end:
            phase = PHASE_EXTEND
            alpha = phase_t / max(t_extend_end, 1e-6)
        elif phase_t < t_hold_end:
            phase = PHASE_HOLD
            alpha = 1.0
        elif phase_t < t_retract_end:
            phase = PHASE_RETRACT
            denom = max(t_retract_end - t_hold_end, 1e-6)
            alpha = 1.0 - (phase_t - t_hold_end) / denom
        else:
            phase = PHASE_IDLE
            alpha = 0.0

        self.last_phase = phase

        qpos = self._rest_qpos.copy()

        # Skip the IK solve if we're fully idle or have no target. This
        # is the common case (idle fraction is non-trivial each cycle).
        if alpha <= 0.0 or self._cycle.target_pos is None:
            self.last_out_of_reach = False
            return qpos

        # Reach gate: if the active shoulder is too far from the target
        # for the arm to physically reach, suppress the reach. This
        # matters during COWORKER_PATROL "away" excursions — the human
        # walks off, the reach state machine still cycles, but the arm
        # stays at rest rather than waving toward an out-of-range target.
        if self._active_shoulder_bid >= 0:
            shoulder_pos = self.data.xpos[self._active_shoulder_bid]
            dist = float(np.linalg.norm(self._cycle.target_pos - shoulder_pos))
            if dist > self._max_reach_dist:
                self.last_out_of_reach = True
                return qpos
        self.last_out_of_reach = False

        # HumanIK.solve copies qpos into its working data but not
        # mocap_pos / mocap_quat. The SMPL-H Pelvis is a mocap body, so
        # without this sync the solver places Pelvis at the world origin
        # and the resulting arm angles reach toward the wrong world
        # position. Sync mocap state before each solve.
        ik_data = self.ik_solver._ik_data
        ik_data.mocap_pos[:] = self.data.mocap_pos[:]
        ik_data.mocap_quat[:] = self.data.mocap_quat[:]

        arm_angles = self.ik_solver.solve(
            self._active_arm(),
            self._cycle.target_pos,
            max_iterations=30,
            tolerance=0.02,
        )
        # Blend rest -> IK by alpha.
        blended = (1.0 - alpha) * self._arm_rest + alpha * arm_angles
        for qpos_idx, angle in zip(self._arm_qpos_indices, blended):
            qpos[qpos_idx] = angle

        return qpos

    # ------------------------------------------------------------------
    # Rest pose construction
    # ------------------------------------------------------------------

    def _build_arms_down_rest_pose(self) -> np.ndarray:
        """Return a qpos with both arms hanging straight down at the sides.

        Uses IK on each arm to a point 0.5 m below the corresponding
        shoulder. Falls back to zeros for any joint where the IK fails
        or the shoulder isn't found.

        This is the pose RETRACT and IDLE blend back to, so it dictates
        the *direction* the wrist travels when the human pulls back. With
        arms down at the sides, the wrist sweeps from the reach target
        downward — away from anywhere the robot is working — rather than
        sideways through the robot's workspace (which the all-zero
        T-pose would produce).
        """
        rest = np.zeros(self.model.nq)

        # Snapshot the current mocap state into the IK solver's working
        # data; otherwise IK works in a frame where the Pelvis is at the
        # world origin (see compute_qpos for the same fix).
        ik_data = self.ik_solver._ik_data
        ik_data.mocap_pos[:] = self.data.mocap_pos[:]
        ik_data.mocap_quat[:] = self.data.mocap_quat[:]
        # Seed IK starting state with arms at the down side of T-pose so
        # the solver doesn't have to walk through the front-of-body
        # subspace to find the hang-down solution. We can't trust live
        # qpos because the initial AMASS frame may have arms half-raised.
        ik_data.qpos[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, ik_data)

        # Solve each arm in turn. The target is well below the shoulder
        # (deeper than the physical arm length) so DLS pulls the wrist
        # to its maximum-down configuration — wrist ends up near
        # hip/thigh level on the same side, which is what "arm hanging
        # at the side" looks like.
        for chain in ("right_arm", "left_arm"):
            shoulder_name = self.ik_solver.chains[chain]["shoulder_body"]
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name
            )
            if sid < 0:
                continue
            shoulder_pos = self.data.xpos[sid].copy()
            target = shoulder_pos + np.array([0.0, 0.0, -0.7])
            try:
                angles = self.ik_solver.solve(
                    chain, target, max_iterations=80, tolerance=0.05,
                    damping=0.05, step_size=0.7,
                )
            except Exception:
                continue
            qpos_indices = self.ik_solver._chain_cache[chain]["qpos_indices"]
            for idx, ang in zip(qpos_indices, angles):
                rest[idx] = ang
            # Apply this arm's result back into the IK working qpos so
            # the next arm's solve starts from a clean half-pose rather
            # than mixing in the first arm's progress.
            for idx, ang in zip(qpos_indices, angles):
                ik_data.qpos[idx] = ang

        return rest

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_arm(self) -> str:
        arm = self.cfg.coworker_active_arm
        if arm not in ("left_arm", "right_arm"):
            return "right_arm"
        return arm

    def _sample_new_target(self, cycle_index: int, robot_state: dict) -> None:
        """Choose ``ee`` vs ``task_object`` for the upcoming cycle."""
        mix = self.cfg.coworker_target_mix
        p_ee = float(mix[0])
        p_task = float(mix[1]) if len(mix) > 1 else max(0.0, 1.0 - p_ee)
        total = p_ee + p_task
        if total <= 0.0:
            kind = "ee"
        else:
            r = self.rng.uniform(0.0, total)
            kind = "ee" if r < p_ee else "task_object"

        # If the chosen kind isn't available (e.g. task_object_pos
        # missing for reach tasks with no manipulable), fall back to EE.
        if kind == "task_object" and robot_state.get("task_object_pos") is None:
            kind = "ee"

        self._cycle.cycle_index = cycle_index
        self._cycle.target_kind = kind
        self._cycle.target_pos = None  # resolved on the same step below

    @staticmethod
    def _lookup_ee(robot_state: dict) -> Optional[np.ndarray]:
        """Best-effort EE world position. Falls back to link_pos['ee']
        because some BiGym robots don't implement ``get_ee_position()``
        and only populate the per-link map."""
        base = robot_state.get("ee_pos")
        if base is None:
            link_pos = robot_state.get("link_pos") or {}
            base = link_pos.get("ee")
        return None if base is None else np.asarray(base, dtype=float)

    def _resolve_target(self, robot_state: dict) -> Optional[np.ndarray]:
        """Look up the live world-position for the cached target kind."""
        if self._cycle.target_kind == "task_object":
            base = robot_state.get("task_object_pos")
            if base is None:
                base = self._lookup_ee(robot_state)
        else:
            base = self._lookup_ee(robot_state)
        if base is None:
            return None
        return np.asarray(base, dtype=float)
