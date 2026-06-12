#!/usr/bin/env python
"""Staged mechanism figures: Lagrangian-vs-baseline, SVF veto anatomy, speed-scale.

ILLUSTRATIVE RECONSTRUCTIONS. The trained policy snapshots live on the GPU
box, so these figures stage each mechanism with a SCRIPTED robot in the real
environment (real physics, real G1 coworker, real safety monitor — every
HUD number is measured from the simulation). The scripted behaviours are
tuned to reproduce what each mechanism measurably did in the benchmark
(docs/filter_fallback_findings.md, docs/results_discussion_draft.md):

- fig_lagrangian_vs_baseline: the unconstrained baseline holds its task
  posture through the coworker's reach windows (proximity 0.296); the
  fixed-lambda=0.1 Lagrangian yields the base away when the coworker is
  near and resumes when they depart (0.228, -23%, unshortened episodes).
- fig_svf_veto_anatomy: the SVF filter classifies the proposed action
  (Q(s,a) vs R=2.25; the Q trace here is a schematic of the v3 critic) and
  substitutes a fallback on veto. From the same veto state we branch:
  no-filter (original action), zero-velocity (freeze -> dwells, no
  proximity reduction), retreat step 0.10 (flees -> separation at the cost
  of the task and 6x velocity).
- fig_speedscale_mechanism: the ISO-SSM speed-scaling filter is REAL math
  (scale = clip((sep-0.15)/(0.40-0.15), 0, 1)) applied to the same
  scripted task motion: full speed when clear, graded slow-down near the
  coworker (ssm-actual 0.146 -> 0.048, -67% in the benchmark).

Each figure carries an explicit "illustrative reconstruction" footer; the
captions in the report cite the real benchmark aggregates.

Usage (from safety_bigym/):
    ./venv/bin/python scripts/render_mechanism_figures.py --figure all
    ./venv/bin/python scripts/render_mechanism_figures.py --probe-poses
    ./venv/bin/python scripts/render_mechanism_figures.py --figure veto --recompose
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("render_mechanism_figures")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the env construction, auto-camera, and visibility probe from the
# patrol-disruption figure script (same directory).
_spec = importlib.util.spec_from_file_location(
    "rdf", Path(__file__).parent / "render_disruption_figure.py")
rdf = importlib.util.module_from_spec(_spec)
sys.modules["rdf"] = rdf
_spec.loader.exec_module(rdf)

PROX = rdf.PROX_THRESH       # 0.30 m
NEAR = rdf.NEAR_THRESH       # 0.50 m
HZ = rdf.CONTROL_HZ          # 20

# Action layout (JointPositionActionMode, floating_dofs=[X,Y,Z,RZ], absolute):
# dims 0-3 = pelvis x/y/z/rz PER-STEP DELTAS (bounds +-0.25); dims 4-15 =
# absolute joint targets [l_shoulder_pitch/roll/yaw, l_elbow, r_shoulder_
# pitch/roll/yaw, r_elbow, l_wrist, r_wrist, l_grip, r_grip].
DIM_BASE_X, DIM_BASE_Y, DIM_BASE_Z, DIM_BASE_RZ = 0, 1, 2, 3
ARM_DIMS = list(range(4, 14))  # absolute-target joint dims (no grippers)

# SVF veto operating point (snapshots.py): R = 2.25.
VETO_R = 2.25
# Speed-scale operating point (results_discussion_draft.md): the optimal cell.
SS_D_SLOW, SS_D_STOP = 0.40, 0.15
# Retreat fallback step in raw base-delta units (filter_fallback_findings 1).
RETREAT_STEP = 0.10

DISCLAIMER = ("illustrative reconstruction - scripted robot, real environment/"
              "coworker; HUD values measured from simulation")


# --------------------------------------------------------------------------
# Scripted robot behaviours
# --------------------------------------------------------------------------

# Hand-tuned "working at the counter" arm cycle (validated via --probe-poses):
# the right arm sweeps between a raised carry pose and a reach toward the
# counter top, the left arm stays at its side.
POSE_REST = np.zeros(12, dtype=np.float32)
POSE_WORK = np.zeros(12, dtype=np.float32)
POSE_WORK[[4, 5, 7]] = [-0.90, -0.20, 1.20]    # right arm extended to counter
POSE_WORK[[0, 3]] = [-0.20, 0.35]              # slight left-arm counterpose
POSE_WORK_B = POSE_WORK.copy()
POSE_WORK_B[[4, 7]] = [-0.45, 0.70]            # half-lowered second waypoint


class ScriptedRobot:
    """Deterministic task behaviour with an optional proximity-yield term.

    mode="baseline": hold the task spot, run the arm work cycle regardless
    of the coworker (the unconstrained policy's signature: it works through
    the reach windows).
    mode="lagrangian": same task cycle, but when the coworker is inside
    d_engage the base smoothly yields away (capped offset) and the arm
    tucks; it returns to the anchor when the coworker departs (the graceful
    avoidance signature of the fixed-lambda policy).
    """

    def __init__(self, mode: str, env, cycle_period: float = 4.0,
                 shuttle_amp: float = 0.0, shuttle_period: float = 6.0):
        self.mode = mode
        self.env = env
        u = env
        while hasattr(u, "env"):
            u = u.env
        self._raw = u
        self._anchor = None      # set on first action (base x, y)
        self._qpos_adr = self._arm_qpos_addresses()
        self.d_engage = 0.85
        self.max_yield = 0.50
        self.gain = 0.10          # fraction of remaining offset per step
        # Work-cycle speed + an oblivious base shuttle between two work
        # spots. The defaults give a placid worker; the speed-scale figure
        # uses (cycle_period~2, shuttle_amp~0.22) so the no-filter baseline
        # carries realistic velocity near the coworker — the regime where
        # the ISO-SSM velocity-adaptive check actually fires.
        self.cycle_period = float(cycle_period)
        self.shuttle_amp = float(shuttle_amp)
        self.shuttle_period = float(shuttle_period)
        self._shuttle_dir = None

    def _arm_qpos_addresses(self) -> List[int]:
        m = self._raw._mojo.model
        adr = []
        names = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                 "left_elbow", "right_shoulder_pitch", "right_shoulder_roll",
                 "right_shoulder_yaw", "right_elbow", "left_wrist", "right_wrist"]
        for n in names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"h1/{n}")
            adr.append(int(m.jnt_qposadr[jid]) if jid >= 0 else -1)
        return adr

    def arm_qpos(self) -> np.ndarray:
        d = self._raw._mojo.data
        return np.array([d.qpos[a] if a >= 0 else 0.0 for a in self._qpos_adr],
                        dtype=np.float32)

    def base_xy(self) -> np.ndarray:
        m = self._raw._mojo.model
        d = self._raw._mojo.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "h1/pelvis")
        return d.xpos[bid][:2].copy()

    def human_xy(self) -> np.ndarray:
        d = self._raw._mojo.data
        return d.xpos[self._raw._human_pelvis_id][:2].copy()

    def action(self, t: float, sep: float) -> np.ndarray:
        a = np.zeros(16, dtype=np.float32)
        base = self.base_xy()
        if self._anchor is None:
            self._anchor = base.copy()
            if self.shuttle_amp > 0.0:
                # Sweep PERPENDICULAR to the walk-in axis: lateral workspace
                # motion beside the coworker (realistic velocity at small
                # separation) without driving the base into them.
                d = self.human_xy() - self._anchor
                n = np.linalg.norm(d)
                d = d / n if n > 1e-6 else np.array([1.0, 0.0])
                self._shuttle_dir = np.array([-d[1], d[0]])

        # --- arm work cycle (rest -> work -> work_b -> rest) ---
        ph = (t % self.cycle_period) / self.cycle_period
        if ph < 0.3:
            k = ph / 0.3
            target = (1 - k) * POSE_REST + k * POSE_WORK
        elif ph < 0.55:
            k = (ph - 0.3) / 0.25
            target = (1 - k) * POSE_WORK + k * POSE_WORK_B
        elif ph < 0.8:
            k = (ph - 0.55) / 0.25
            target = (1 - k) * POSE_WORK_B + k * POSE_WORK
        else:
            k = (ph - 0.8) / 0.2
            target = (1 - k) * POSE_WORK + k * POSE_REST

        tuck = 1.0
        goal = self._anchor.copy()
        if self._shuttle_dir is not None:
            # Oblivious shuttle between two work spots along the walk-in
            # axis (visible laterally from the episode camera).
            half = 0.5 * self.shuttle_period
            sgn = 1.0 if (t % self.shuttle_period) < half else -1.0
            goal = goal + self._shuttle_dir * (self.shuttle_amp * sgn)
        if self.mode == "lagrangian" and np.isfinite(sep) and sep < self.d_engage:
            # Yield: offset the base goal away from the coworker, scaled by
            # how deep they are inside the engage radius; tuck the arm.
            human = self.human_xy()
            away = base - human
            n = np.linalg.norm(away)
            away = away / n if n > 1e-6 else np.array([1.0, 0.0])
            depth = min(1.0, (self.d_engage - sep) / self.d_engage)
            goal = self._anchor + away * (self.max_yield * depth)
            tuck = float(np.clip((sep - 0.35) / 0.35, 0.15, 1.0))

        a[4:14] = tuck * target[:10] + (1 - tuck) * 0.0
        # base: first-order pursuit of goal. The shuttle uses a brisker
        # pursuit (<= 0.05/step = 1.0 m/s) so the no-filter run carries
        # SSM-relevant velocity; the default is a placid 0.8 m/s cap.
        gain, cap = ((0.16, 0.05) if self._shuttle_dir is not None
                     else (self.gain, 0.04))
        delta = np.clip(gain * (goal - base), -cap, cap)
        a[DIM_BASE_X], a[DIM_BASE_Y] = float(delta[0]), float(delta[1])
        return a


def q_hat(sep: float, approach: float) -> float:
    """Schematic SVF safety-Q: high when the coworker is clear, dipping
    below R when the proposed action keeps moving with the coworker close.
    Shape calibrated to the v3 critic's sweep (passthrough mean_q 3.2,
    interventions concentrated in the reach windows at R=2.25). Unknown
    separation (start of episode, before the first monitor step) reads as
    clear."""
    if not np.isfinite(sep):
        return 3.3
    q = 1.0 + 2.3 * float(np.clip((sep - 0.30) / 0.55, 0.0, 1.0))
    q -= 0.45 * float(np.clip(approach / 0.03, 0.0, 1.0))
    return q


# --------------------------------------------------------------------------
# Staged rollout
# --------------------------------------------------------------------------

def run_staged(
    env,
    seed: int,
    mode: str,                 # baseline | lagrangian
    filter_mode: str,          # none | veto_zero | veto_retreat | speedscale
    sim_seconds: float,
    capture_every: int,
    frames_dir: Optional[Path],
    cam_azimuth: Optional[float],
    width: int = 960,
    height: int = 720,
    behavior_kwargs: Optional[dict] = None,
) -> dict:
    import imageio.v2 as imageio

    env.reset(seed=seed)
    robot = ScriptedRobot(mode, env, **(behavior_kwargs or {}))
    raw = robot._raw
    hc = raw.human_controller

    renderer = cam = None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        renderer = rdf._make_renderer(raw, width, height)
        cam = rdf._auto_camera(raw, rdf._scenario_meta(raw))
        if cam_azimuth is not None:
            cam.azimuth = float(cam_azimuth)

    rows = []
    frame_steps: List[int] = []
    sep = float("inf")
    n_steps = int(sim_seconds * HZ)
    prev_base = robot.base_xy()

    for i in range(n_steps):
        proposed = robot.action(hc.t, sep)

        # approach = commanded base motion component toward the coworker
        human = robot.human_xy()
        base = robot.base_xy()
        to_h = human - base
        nh = np.linalg.norm(to_h)
        to_h = to_h / nh if nh > 1e-6 else np.zeros(2)
        approach = float(proposed[DIM_BASE_X] * to_h[0]
                         + proposed[DIM_BASE_Y] * to_h[1])

        q = q_hat(sep, approach)
        scale = 1.0
        intervened = False
        executed = proposed

        if filter_mode in ("veto_zero", "veto_retreat") and q < VETO_R:
            intervened = True
            if filter_mode == "veto_zero":
                executed = np.zeros(16, dtype=np.float32)   # ZeroVelocityFallback
            else:
                executed = np.zeros(16, dtype=np.float32)   # RetreatFallback: base away
                executed[DIM_BASE_X] = -RETREAT_STEP * to_h[0]
                executed[DIM_BASE_Y] = -RETREAT_STEP * to_h[1]
        elif filter_mode == "speedscale":
            # REAL speed-scale law on the commanded per-step motion. Unknown
            # separation (first step) reads as clear -> full speed.
            scale = float(np.clip((sep - SS_D_STOP) / (SS_D_SLOW - SS_D_STOP),
                                  0.0, 1.0)) if np.isfinite(sep) else 1.0
            if scale < 1.0:
                intervened = True
                executed = proposed.copy()
                executed[:4] *= scale
                cur = robot.arm_qpos()
                executed[4:14] = cur + scale * (proposed[4:14] - cur)

        _, _, terminated, truncated, info = env.step(executed)
        safety = info.get("safety", {}) or {}
        sep = float(safety.get("min_separation", np.nan))
        rows.append(dict(
            t=float(hc.t), sep=sep,
            prox=bool(safety.get("proximity_violation", False)),
            ssm_actual=bool(safety.get("ssm_violation_actual", False)),
            robot_vel=float(safety.get("robot_vel", np.nan)),
            q=float(q), scale=float(scale), intervened=bool(intervened),
            base_speed=float(np.linalg.norm(robot.base_xy() - prev_base) * HZ),
        ))
        prev_base = robot.base_xy()

        if renderer is not None and i % capture_every == 0:
            renderer.update_scene(raw._mojo.data, camera=cam)
            imageio.imwrite(frames_dir / f"frame_{len(frame_steps):05d}.png",
                            renderer.render())
            frame_steps.append(i)
        if terminated or truncated:
            break

    if renderer is not None:
        renderer.close()
    keys = rows[0].keys()
    series = {k: [r[k] for r in rows] for k in keys}
    series["frame_step"] = frame_steps
    series["mode"] = mode
    series["filter_mode"] = filter_mode
    series["seed"] = seed
    return series


# --------------------------------------------------------------------------
# Composition helpers (HUD style shared with the patrol figure)
# --------------------------------------------------------------------------

def _chip(ax, x, y, text, color="white", bg=(0, 0, 0, 0.66), ha="left",
          va="top", size=10.5, weight="normal"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
            fontweight=weight, color=color,
            bbox=dict(facecolor=bg if isinstance(bg, str) else bg[:3],
                      alpha=bg[3] if not isinstance(bg, str) else 0.92,
                      pad=3.5, edgecolor="none"))


def panel(ax, img, letter, label, t, sep, viol, extra=(), status_override=None):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#dc2828" if viol else "#444444")
        sp.set_linewidth(3.5 if viol else 0.8)
    _chip(ax, 0.025, 0.965, f"({letter})  {label}", size=11.5, weight="bold")
    _chip(ax, 0.975, 0.965, f"t = {t:.1f} s", ha="right", size=10)
    status, color = status_override or rdf._status(sep)
    sep_color = rdf._status(sep)[1]
    sep_txt = "contact" if not np.isfinite(sep) else f"{sep:.2f} m"
    ax.text(0.025, 0.038, f"min sep  {sep_txt}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, fontweight="bold",
            color=sep_color, bbox=dict(facecolor="black", alpha=0.66, pad=3.5,
                                       edgecolor="none"))
    ax.text(0.975, 0.038, status, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=10.5, fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.92, pad=3.5, edgecolor="none"))
    # extra chips: list of (text, fg, bg, (x, y, ha, va))
    for text, fg, bg, pos in extra:
        x, y, ha, va = pos
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=10, fontweight="bold", color=fg,
                bbox=dict(facecolor=bg, alpha=0.9, pad=3.5, edgecolor="none"))


def _load_frame(dump: Path, series: dict, step: int):
    import imageio.v2 as imageio
    fs = np.asarray(series["frame_step"])
    k = int(np.argmin(np.abs(fs - step)))
    return imageio.imread(dump / f"frame_{k:05d}.png")


def _nearest(series: dict, t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(series["t"]) - t)))


def _viol_ribbon(ax, t, mask, y0, h, color, label=None):
    ax.fill_between(t, y0, y0 + h, where=np.asarray(mask, dtype=bool),
                    transform=ax.get_xaxis_transform(), color=color,
                    alpha=0.85, linewidth=0, label=label, step="mid")


def _footer(fig, staged_text):
    fig.text(0.988, 0.005,
             f"{staged_text}  ·  {DISCLAIMER}",
             ha="right", va="bottom", fontsize=8.5, color="#444444",
             style="italic")


def _savefig(fig, out_base: Path) -> List[Path]:
    import matplotlib.pyplot as plt
    out_base.parent.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("png", "pdf"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300)
        outs.append(p)
    plt.close(fig)
    return outs


# --------------------------------------------------------------------------
# Figure A: Lagrangian vs baseline
# --------------------------------------------------------------------------

def compose_lagrangian(dumps: Dict[str, Path], series: Dict[str, dict],
                       fig_base: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    sb, sl = series["baseline"], series["lagrangian"]
    tb = np.asarray(sb["t"]); tl = np.asarray(sl["t"])
    sep_b = np.asarray(sb["sep"]); sep_l = np.asarray(sl["sep"])

    # Column instants: before approach / first reach violation / deepest
    # violation in a LATER reach cycle / mid of the away excursion.
    prox_b = np.asarray(sb["prox"])
    viol_idx = np.flatnonzero(prox_b)
    i_first = int(viol_idx[0]) if len(viol_idx) else int(np.nanargmin(sep_b))
    i_pre = max(0, i_first - int(2.5 * HZ))
    gap = int(3.0 * HZ)
    n0 = min(len(sep_b), len(sep_l))
    # Max-contrast instant: baseline in violation while the Lagrangian holds
    # the largest margin, at least `gap` after the first-violation column.
    contrast = sep_l[:n0] - sep_b[:n0]
    contrast[~prox_b[:n0]] = -np.inf
    contrast[:i_first + gap] = -np.inf
    if np.isfinite(contrast).any() and np.max(contrast) > -np.inf:
        i_deep = int(np.argmax(contrast))
    else:
        later_viol = sep_b.copy()
        later_viol[:i_first + gap] = np.inf
        i_deep = int(np.nanargmin(later_viol)) if np.isfinite(later_viol).any() \
            else min(i_first + gap, len(tb) - 1)
    # post-depart: a clearly-away moment (max separation after the deep dip)
    n = min(len(tb), len(tl))
    tail = np.minimum(sep_b[:n], sep_l[:n]).copy()
    tail[:min(i_deep + int(1.5 * HZ), n - 1)] = -np.inf
    i_post = int(np.argmax(tail))
    cols = sorted(set([i_pre, i_first, i_deep, i_post]))[:4]
    while len(cols) < 4:
        cols.append(min(cols[-1] + int(1.5 * HZ), n - 1))

    fig = plt.figure(figsize=(16.4, 9.6))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.6],
                           hspace=0.07, wspace=0.025,
                           left=0.034, right=0.988, top=0.965, bottom=0.075)
    row_meta = [("baseline", sb, "BASELINE\n(unconstrained)", "#8a2525"),
                ("lagrangian", sl, "LAGRANGIAN\nfixed λ = 0.1", "#1f3b73")]
    letters = "abcdefgh"
    col_labels = [
        ["coworker clear", "coworker reaches in",
         "works through reach", "coworker departed"],
        ["coworker clear", "yields away",
         "keeps margin", "returns to task"],
    ]
    for r, (tag, s, title, color) in enumerate(row_meta):
        for c, idx in enumerate(cols):
            idx = min(idx, len(s["t"]) - 1)
            ax = fig.add_subplot(gs[r, c])
            img = _load_frame(dumps[tag], s, idx)
            panel(ax, img, letters[r * 4 + c], col_labels[r][c], s["t"][idx],
                  s["sep"][idx], bool(s["prox"][idx]))
            if c == 0:
                ax.text(-0.045, 0.5, title, transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=10.5,
                        fontweight="bold", color=color)

    axt = fig.add_subplot(gs[2, :])
    axt.plot(tb, sep_b, color="#b03a3a", lw=1.7, label="baseline min separation")
    axt.plot(tl, sep_l, color="#1f3b73", lw=1.7, label="Lagrangian min separation")
    axt.axhline(PROX, color="#dc2828", lw=1.1,
                label=f"proximity threshold ({PROX:.1f} m)")
    _viol_ribbon(axt, tb, sb["prox"], 0.00, 0.05, "#b03a3a",
                 label="baseline violation")
    _viol_ribbon(axt, tl, sl["prox"], 0.06, 0.05, "#3a5fb0",
                 label="Lagrangian violation")
    ymax = max(np.nanmax(sep_b), np.nanmax(sep_l)) * 1.1
    for k, idx in enumerate(cols):
        ti = tb[min(idx, len(tb) - 1)]
        axt.axvline(ti, color="#222222", lw=0.7, alpha=0.5)
        axt.text(ti, ymax * 0.99, f"({letters[k]}/{letters[k+4]})", ha="center",
                 va="top", fontsize=9, fontweight="bold", color="#222222")
    axt.set_ylim(0, ymax)
    axt.set_xlim(tb[0], min(tb[-1], tl[-1]))
    axt.set_xlabel("episode time (s)", fontsize=11)
    axt.set_ylabel("human–robot distance (m)", fontsize=10.5, labelpad=2)
    axt.tick_params(labelsize=9.5)
    axt.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)
    pb, pl = float(np.mean(sb["prox"])), float(np.mean(sl["prox"]))
    _footer(fig, f"this episode: {100*pb:.0f}% vs {100*pl:.0f}% of steps in "
                 f"proximity  ·  benchmark (180 ep): 0.296 vs 0.228 (−23%)")
    return _savefig(fig, fig_base)


# --------------------------------------------------------------------------
# Figure B: SVF veto anatomy (freeze vs flee)
# --------------------------------------------------------------------------

def compose_veto(dumps: Dict[str, Path], series: Dict[str, dict],
                 fig_base: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    sv = series["veto_zero"]
    sn, sr = series["nofilter"], series["veto_retreat"]
    t = np.asarray(sv["t"])
    q = np.asarray(sv["q"])
    iv = np.asarray(sv["intervened"], dtype=bool)

    # PASS frame: max-Q moment after warm-up; VETO frame: first intervention
    # after warm-up; branch frames: veto + delta.
    warm = int(1.0 * HZ)
    live = np.flatnonzero(iv & (np.arange(len(iv)) >= warm))
    i_veto = int(live[0]) if len(live) else int(np.argmin(q[warm:])) + warm
    i_pass = warm + int(np.argmax(q[warm:max(i_veto - int(0.5 * HZ), warm + 1)]))
    i_branch = min(i_veto + int(2.0 * HZ), len(t) - 1,
                   len(sn["t"]) - 1, len(sr["t"]) - 1)

    fig = plt.figure(figsize=(12.6, 11.2))
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 0.58],
                           hspace=0.10, wspace=0.03,
                           left=0.04, right=0.985, top=0.94, bottom=0.07)

    def qchip(s, i, ax):
        qq = s["q"][i]
        ok = qq >= VETO_R
        _chip(ax, 0.5, 0.855,
              f"Q̂(s,a) = {qq:.2f}  {'≥' if ok else '<'}  R = {VETO_R}"
              f"  →  {'PASS' if ok else 'VETO'}",
              ha="center", size=11, weight="bold",
              bg="#28b450" if ok else "#dc2828")

    # Row 1: classification (2 wide panels)
    ax = fig.add_subplot(gs[0, 0:3])
    panel(ax, _load_frame(dumps["veto_zero"], sv, i_pass), "a",
          "filter check: PASS", t[i_pass], sv["sep"][i_pass],
          bool(sv["prox"][i_pass]))
    qchip(sv, i_pass, ax)
    ax = fig.add_subplot(gs[0, 3:6])
    panel(ax, _load_frame(dumps["veto_zero"], sv, i_veto), "b",
          "filter check: VETO", t[i_veto], sv["sep"][i_veto],
          bool(sv["prox"][i_veto]))
    qchip(sv, i_veto, ax)

    # Row 2: three branches from the veto state (+2.0 s)
    branches = [
        ("nofilter", sn, "c", "original action", "#8a2525"),
        ("veto_zero", sv, "d", "zero-velocity — dwells", "#7a6a1e"),
        ("veto_retreat", sr, "e", "retreat — flees", "#1f6b3a"),
    ]
    for k, (tag, s, letter, lab, _c) in enumerate(branches):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        i = min(i_branch, len(s["t"]) - 1)
        panel(ax, _load_frame(dumps[tag], s, i), letter, lab, s["t"][i],
              s["sep"][i], bool(s["prox"][i]))
        vel = s["robot_vel"][i]
        _chip(ax, 0.5, 0.038, f"robot vel {vel:.2f} m/s", ha="center",
              va="bottom", size=10,
              bg="#dc2828" if vel > 1.0 else (0, 0, 0, 0.66))
        if tag == "veto_retreat":
            _chip(ax, 0.5, 0.14, "task abandoned", ha="center", va="bottom",
                  size=10, weight="bold", bg="#dc2828")

    # Timeline: schematic Q vs R with veto ticks + separation twin
    axt = fig.add_subplot(gs[2, :])
    axt.plot(t, q, color="#5b2d8a", lw=1.6, label="Q̂(s, a) (schematic)")
    axt.axhline(VETO_R, color="#dc2828", lw=1.1, ls="--",
                label=f"veto threshold R = {VETO_R}")
    _viol_ribbon(axt, t, iv, 0.0, 0.06, "#5b2d8a", label="filter intervenes")
    for ti, lab in ((t[i_pass], "(a)"), (t[i_veto], "(b)"), (t[i_branch], "(c–e)")):
        axt.axvline(ti, color="#222222", lw=0.7, alpha=0.5)
        axt.text(ti, 0.98, lab, transform=axt.get_xaxis_transform(),
                 ha="center", va="top", fontsize=9, fontweight="bold")
    ax2 = axt.twinx()
    ax2.plot(t, sv["sep"], color="#888888", lw=1.1, ls=":")
    ax2.set_ylabel("min separation (m)", fontsize=9.5, color="#666666")
    ax2.tick_params(labelsize=8.5, colors="#666666")
    axt.set_xlabel("episode time (s)", fontsize=11)
    axt.set_ylabel("safety value Q̂", fontsize=10.5)
    axt.tick_params(labelsize=9.5)
    axt.set_xlim(t[0], t[-1])
    axt.legend(loc="upper right", fontsize=8.5, ncol=3, framealpha=0.9)
    _footer(fig, "benchmark (60 ep): zero-velocity → proximity unchanged (0.303), "
                 "dwells · retreat → 0.095 (−68%) but success 0.85→0.18, vel ×6")
    return _savefig(fig, fig_base)


# --------------------------------------------------------------------------
# Figure C: speed-scale mechanism
# --------------------------------------------------------------------------

def compose_speedscale(dumps: Dict[str, Path], series: Dict[str, dict],
                       fig_base: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    sn, ss = series["nofilter"], series["speedscale"]
    t = np.asarray(ss["t"])
    sc = np.asarray(ss["scale"])
    sep = np.asarray(ss["sep"])

    # Columns in NARRATIVE order (clear -> slowing -> deep slow-down), not
    # chronological: each column's label must match its content.
    warm = int(1.0 * HZ)
    n0 = min(len(t), len(sn["t"]))
    idx = np.arange(warm, n0)

    def _smooth(x, w=int(0.4 * HZ)):
        return np.convolve(np.asarray(x, dtype=np.float64),
                           np.ones(w) / w, mode="same")

    vel_n = _smooth(sn["robot_vel"])[:n0]
    vel_s = _smooth(ss["robot_vel"])[:n0]
    # Deep slow-down: lowest scale among instants where the scaled run is
    # genuinely slower than the no-filter run on the RAW instantaneous
    # velocity (the HUD value) and not in the contact-shove regime where
    # the coworker pushes the held arm (sep below d_stop).
    rv_n = np.asarray(sn["robot_vel"])[:n0]
    rv_s = np.asarray(ss["robot_vel"])[:n0]
    sep_s = np.asarray(ss["sep"])[:n0]
    ssm_n = np.asarray(sn["ssm_actual"], dtype=bool)[:n0]
    ssm_s = np.asarray(ss["ssm_actual"], dtype=bool)[:n0]
    # NEAR column: the axis-contrast moment — the no-filter run in
    # SSM-actual violation while the scaled run (slower) is not, picked at
    # the largest instantaneous velocity gap outside the contact regime.
    cand = idx[ssm_n[idx] & ~ssm_s[idx] & (sep_s[idx] > 0.12)
               & (sc[idx] < 0.6)]
    if len(cand):
        i_hold = int(cand[np.argmax(rv_n[cand] - rv_s[cand])])
    else:
        deep = idx[(sc[idx] < 0.35) & (rv_s[idx] < 0.6 * rv_n[idx])
                   & (sep_s[idx] > 0.12)]
        i_hold = int(deep[np.argmin(sc[deep])]) if len(deep) \
            else int(idx[np.argmin(sc[idx])])
    part = idx[(sc[idx] > 0.30) & (sc[idx] < 0.85)]
    i_slow = int(part[np.argmin(np.abs(sc[part] - 0.55))]) if len(part) \
        else max(warm, i_hold - int(1.5 * HZ))
    full = idx[sc[idx] >= 0.999]
    early = full[full < i_slow]
    pool = early if len(early) else full
    i_clear = int(pool[np.argmax(sep[pool])]) if len(pool) else warm
    cols = [i_clear, i_slow, i_hold]

    fig = plt.figure(figsize=(12.6, 9.8))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.62],
                           hspace=0.08, wspace=0.03,
                           left=0.04, right=0.985, top=0.955, bottom=0.07)
    letters = "abcdef"
    # Borders and status chips on THIS figure are keyed to the ISO-SSM
    # velocity-adaptive violation — the axis the speed-scaling filter
    # targets (geometric proximity is unchanged by design).
    for r, (tag, s, title, color) in enumerate(
            [("nofilter", sn, "NO FILTER", "#8a2525"),
             ("speedscale", ss, "ISO-SSM SPEED-SCALING", "#1f6b3a")]):
        for c, idx_c in enumerate(cols):
            i = min(idx_c, len(s["t"]) - 1)
            ax = fig.add_subplot(gs[r, c])
            ssm = bool(s["ssm_actual"][i])
            panel(ax, _load_frame(dumps[tag], s, i), letters[r * 3 + c],
                  ["coworker clear", "coworker approaches", "coworker at NEAR"][c],
                  s["t"][i], s["sep"][i], ssm,
                  status_override=(("SSM VIOLATION", "#dc2828") if ssm
                                   else ("SSM OK", "#28b450")))
            vel = s["robot_vel"][i]
            _chip(ax, 0.5, 0.13, f"robot vel {vel:.2f} m/s", ha="center",
                  va="bottom", size=10, weight="bold",
                  bg="#dc2828" if ssm else (0, 0, 0, 0.66))
            if r == 1:
                _chip(ax, 0.5, 0.225, f"scale = {s['scale'][i]:.2f}",
                      ha="center", va="bottom", size=10.5, weight="bold",
                      bg="#1f6b3a")
            if c == 0:
                ax.text(-0.045, 0.5, title, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color=color)

    axt = fig.add_subplot(gs[2, :])
    tn = np.asarray(sn["t"])
    axt.plot(tn[:n0], vel_n[:n0], color="#b03a3a", lw=1.4,
             label="robot speed, no filter")
    axt.plot(t[:n0], vel_s[:n0], color="#1f6b3a", lw=1.4,
             label="robot speed, speed-scaled")
    _viol_ribbon(axt, tn, sn["ssm_actual"], 0.00, 0.05, "#b03a3a",
                 label="SSM-actual violation (no filter)")
    _viol_ribbon(axt, t, ss["ssm_actual"], 0.06, 0.05, "#1f6b3a",
                 label="SSM-actual violation (scaled)")
    axt.set_xlabel("episode time (s)", fontsize=11)
    axt.set_ylabel("robot link speed (m/s)", fontsize=10.5)
    axt.set_xlim(t[0], min(t[-1], tn[-1]))
    axt.tick_params(labelsize=9.5)
    ax2 = axt.twinx()
    ax2.plot(t, sc, color="#555555", lw=1.2, ls="--", label="speed scale")
    ax2.plot(t, sep, color="#999999", lw=1.0, ls=":", label="min separation (m)")
    ax2.set_ylabel("scale  /  separation (m)", fontsize=9.5, color="#555555")
    ax2.tick_params(labelsize=8.5, colors="#555555")
    h1, l1 = axt.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axt.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=2,
               framealpha=0.9)
    for k, idx in enumerate(cols):
        ti = t[min(idx, len(t) - 1)]
        axt.axvline(ti, color="#222222", lw=0.7, alpha=0.5)
        axt.text(ti, 0.98, f"({letters[k]}/{letters[k+3]})",
                 transform=axt.get_xaxis_transform(), ha="center", va="top",
                 fontsize=9, fontweight="bold")
    ssm_pct_n = 100.0 * float(ssm_n.mean())
    ssm_pct_s = 100.0 * float(ssm_s.mean())
    _footer(fig, f"borders/status: ISO-SSM velocity-adaptive violation (this "
                 f"filter's target axis; proximity unchanged by design) · "
                 f"this episode: {ssm_pct_n:.0f}% → {ssm_pct_s:.0f}% of steps "
                 f"in SSM violation · scale = clip((sep−0.15)/(0.40−0.15), 0, 1)"
                 f" — real filter law · benchmark (60 ep): ssm-actual "
                 f"0.146→0.048 (−67%)")
    return _savefig(fig, fig_base)


# --------------------------------------------------------------------------
# Pose probe
# --------------------------------------------------------------------------

def probe_poses(env, seed: int, out_dir: Path, width=960, height=720):
    """Render candidate work poses so the arm cycle can be hand-tuned."""
    import imageio.v2 as imageio
    env.reset(seed=seed)
    u = env
    while hasattr(u, "env"):
        u = u.env
    renderer = rdf._make_renderer(u, width, height)
    cam = rdf._auto_camera(u, rdf._scenario_meta(u))
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = {
        "rest": np.zeros(12, dtype=np.float32),
        "work": POSE_WORK,
        "work_b": POSE_WORK_B,
    }
    for name, pose in poses.items():
        a = np.zeros(16, dtype=np.float32)
        a[4:16] = np.asarray(pose, dtype=np.float32)[:12]
        for _ in range(int(2.0 * HZ)):   # settle 2 s
            env.step(a)
        for az_off, tag in ((0.0, ""), (180.0, "_flip")):
            cam2 = mujoco.MjvCamera()
            cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam2.lookat[:] = cam.lookat
            cam2.distance, cam2.elevation = cam.distance * 0.8, cam.elevation
            cam2.azimuth = cam.azimuth + az_off
            renderer.update_scene(u._mojo.data, camera=cam2)
            imageio.imwrite(out_dir / f"pose_{name}{tag}.png", renderer.render())
        log.info("  pose %s rendered", name)
    renderer.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

FIGURES = {
    "lagrangian": dict(
        runs=[("baseline", "baseline", "none"),
              ("lagrangian", "lagrangian", "none")],
        compose=compose_lagrangian,
        fig_name="fig_lagrangian_vs_baseline",
        sim_seconds=40.0),
    "veto": dict(
        runs=[("nofilter", "baseline", "none"),
              ("veto_zero", "baseline", "veto_zero"),
              ("veto_retreat", "baseline", "veto_retreat")],
        compose=compose_veto,
        fig_name="fig_svf_veto_anatomy",
        sim_seconds=26.0),
    "speedscale": dict(
        runs=[("nofilter", "baseline", "none"),
              ("speedscale", "baseline", "speedscale")],
        compose=compose_speedscale,
        fig_name="fig_speedscale_mechanism",
        sim_seconds=40.0,
        # Faster work cycle + oblivious shuttle so the no-filter baseline
        # carries realistic velocity near the coworker (the regime the
        # ISO-SSM velocity check penalises and the filter suppresses).
        behavior=dict(cycle_period=2.0, shuttle_amp=0.22)),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", default="all",
                   choices=["all", *FIGURES.keys()])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task", default="saucepan", choices=list(rdf.TASK_MAP))
    p.add_argument("--human-model", default="g1", choices=["g1", "smplh"])
    p.add_argument("--capture-every", type=int, default=2)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "mechanism_figures")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    p.add_argument("--pin-closest", type=float, default=0.72)
    p.add_argument("--pin-p-ee", type=float, default=0.70)
    p.add_argument("--pin-near-loiter", type=float, default=None)
    p.add_argument("--pin-away-distance", type=float, default=None)
    p.add_argument("--pin-excursions", type=int, default=1)
    p.add_argument("--cam-azimuth", type=float, default=None)
    p.add_argument("--probe-poses", action="store_true")
    p.add_argument("--recompose", action="store_true")
    args = p.parse_args()

    todo = list(FIGURES) if args.figure == "all" else [args.figure]
    out_root = args.out_dir / f"seed_{args.seed}"

    if args.recompose:
        for name in todo:
            spec = FIGURES[name]
            dumps, series = {}, {}
            for tag, _, _ in spec["runs"]:
                d = out_root / name / tag
                series[tag] = json.loads((d / "series.json").read_text())
                dumps[tag] = d
            outs = spec["compose"](dumps, series, args.fig_dir / spec["fig_name"])
            for o in outs:
                log.info("wrote %s", o)
        return 0

    env = rdf.build_env(args.task, args.human_model, args)
    try:
        if args.probe_poses:
            probe_poses(env, args.seed, out_root / "pose_probe",
                        args.width, args.height)
            return 0

        cam_az = args.cam_azimuth
        if cam_az is None:
            log.info("probing camera side (seed %d)...", args.seed)
            cam_az = rdf._pick_visible_azimuth(
                env, args.seed, 30.0, args.width, args.height)

        for name in todo:
            spec = FIGURES[name]
            dumps, series = {}, {}
            for tag, mode, fmode in spec["runs"]:
                d = out_root / name / tag
                log.info("[%s] staging run %s (mode=%s filter=%s)...",
                         name, tag, mode, fmode)
                s = run_staged(env, args.seed, mode, fmode,
                               spec["sim_seconds"], args.capture_every,
                               d, cam_az, args.width, args.height,
                               behavior_kwargs=spec.get("behavior"))
                (d / "series.json").write_text(json.dumps(s))
                series[tag] = s
                dumps[tag] = d
                log.info("    %d steps, prox %.0f%%, ssm-actual %.0f%%, "
                         "min sep %.2f, vel mean %.2f",
                         len(s["t"]), 100 * np.mean(s["prox"]),
                         100 * np.mean(s["ssm_actual"]),
                         np.nanmin(s["sep"]), np.nanmean(s["robot_vel"]))
            outs = spec["compose"](dumps, series, args.fig_dir / spec["fig_name"])
            for o in outs:
                log.info("wrote %s", o)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
