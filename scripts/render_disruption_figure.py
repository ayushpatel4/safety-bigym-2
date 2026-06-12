#!/usr/bin/env python
"""Render a report figure of the COWORKER_PATROL disruption as a frame grid.

Rolls out one episode of the COWORKER disruption with the trajectory forced
to COWORKER_PATROL (walk in -> reach cycles at NEAR -> walk away -> loiter
AWAY -> return -> reach again), records frames from a custom wide camera
plus the per-step ``info["safety"]`` series, then composes a 2x4 panel grid
with HUD-style annotations (time, phase, min separation, SAFE/NEAR/CLOSE
status, red border while in proximity violation) over a min-separation
timeline with the panel instants marked.

The robot holds its home pose (zero absolute joint targets) so every
violation in the figure is attributable to the human's behaviour, not the
policy. Coworker knobs default to the canonical stage-2 training band
(cfgs/disruption/coworker_train.yaml, 2026-05-28 values).

Usage (from safety_bigym/, AMASS only needed for --human-model smplh)::

    ./venv/bin/python scripts/render_disruption_figure.py            # scout + render + compose
    ./venv/bin/python scripts/render_disruption_figure.py --seed 7   # skip scouting
    ./venv/bin/python scripts/render_disruption_figure.py --probe-camera --seed 7
    ./venv/bin/python scripts/render_disruption_figure.py --recompose --seed 7

On a headless Linux box prepend ``MUJOCO_GL=egl PYOPENGL_PLATFORM=egl``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mujoco
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("render_disruption_figure")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROX_THRESH = 0.30  # SSMConfig.proximity_threshold (thesis-primary axis)
NEAR_THRESH = 0.50

TASK_MAP = {
    "reach": "bigym.envs.reach_target:ReachTargetSingle",
    "saucepan": "bigym.envs.pick_and_place:SaucepanToHob",
    "dishwasher_close": "bigym.envs.dishwasher:DishwasherClose",
    "drawers_open_all": "bigym.envs.cupboards:DrawersAllOpen",
}

# Canonical stage-2 COWORKER training band (cfgs/disruption/coworker_train.yaml).
COWORKER_TRAIN_KNOBS = dict(
    coworker_closest_approach_range=(0.60, 0.95),
    coworker_reach_period_range=(1.3, 2.2),
    coworker_target_mix_p_ee_range=(0.45, 0.72),
    coworker_near_loiter_range=(9.0, 14.0),
    coworker_walk_speed_range=(1.0, 1.4),
)

CONTROL_HZ = 20  # matches saucepan_to_hob training control frequency


# --------------------------------------------------------------------------
# Env construction
# --------------------------------------------------------------------------

def _load_task_cls(task_key: str) -> type:
    module_path, cls_name = TASK_MAP[task_key].rsplit(":", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def build_env(task_key: str, human_model: str, args):
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from safety_bigym import HumanConfig, SafetyConfig, make_safety_env
    from safety_bigym.scenarios import DisruptionType, ParameterSpace, ScenarioSampler

    motion_dir = None
    clip_paths: list[str] = []
    if human_model == "smplh":
        motion_dir = os.environ.get("AMASS_DATA_DIR")
        if not motion_dir:
            raise RuntimeError(
                "AMASS_DATA_DIR is not set (required for --human-model smplh):\n"
                "  export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU"
            )
        clip_paths = ["74/74_01_poses.npz"]

    space = ParameterSpace(
        clip_paths=clip_paths,
        disruption_weights={DisruptionType.COWORKER: 1.0},
        coworker_trajectory_weights={"COWORKER_PATROL": 1.0},
        **COWORKER_TRAIN_KNOBS,
    )
    sampler = ScenarioSampler(parameter_space=space, motion_dir=motion_dir)
    _patch_sampler(sampler, args)

    env = make_safety_env(
        task_cls=_load_task_cls(task_key),
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        safety_config=SafetyConfig(log_violations=False, terminate_on_violation=False),
        human_config=HumanConfig(
            human_model=human_model,
            motion_clip_dir=motion_dir,
            motion_clip_paths=clip_paths,
        ),
        scenario_sampler=sampler,
        inject_human=True,
        control_frequency=CONTROL_HZ,
    )
    return env


def _patch_sampler(sampler, args) -> None:
    """Post-sample pins so the figure episode is art-directable from the CLI.

    Defaults leave the scenario exactly as sampled from the training band;
    only ``patrol_excursions`` is floored at 1 (it always is) so the
    depart+return pattern exists within the recording window.
    """
    base = sampler.sample_scenario

    def _patched(seed):
        s = base(seed)
        s.trajectory_type = "COWORKER_PATROL"
        if args.pin_closest is not None:
            s.closest_approach = float(args.pin_closest)
        if args.pin_p_ee is not None:
            p = float(args.pin_p_ee)
            s.disruption_config.coworker_target_mix = (p, 1.0 - p)
        if args.pin_near_loiter is not None:
            s.patrol_near_loiter = float(args.pin_near_loiter)
        if args.pin_away_distance is not None:
            s.patrol_away_distance = float(args.pin_away_distance)
        if args.pin_excursions is not None:
            s.patrol_excursions = int(args.pin_excursions)
        return s

    sampler.sample_scenario = _patched  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Rollout + recording
# --------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    seed: int
    scenario: dict
    t: np.ndarray            # (N,) human-controller sim time per env step
    min_sep: np.ndarray      # (N,) closest human-joint <-> robot-link distance
    prox: np.ndarray         # (N,) bool, min_sep < 0.3
    ssm_actual: np.ndarray   # (N,) bool, observed-velocity ISO violation
    traj_phase: np.ndarray   # (N,) str: approach / loiter / depart / walk
    arm_phase: np.ndarray    # (N,) str: idle / extend / hold / retract
    pelvis_dist: np.ndarray  # (N,) human pelvis XY distance to robot anchor
    frame_step: np.ndarray   # (F,) env-step index of each captured frame

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "t", "min_sep", "prox", "ssm_actual", "pelvis_dist", "frame_step")}
        out = {k: np.asarray(v).tolist() for k, v in d.items()}
        out["traj_phase"] = list(self.traj_phase)
        out["arm_phase"] = list(self.arm_phase)
        out["seed"] = self.seed
        out["scenario"] = self.scenario
        return out

    @staticmethod
    def from_json(d: dict) -> "EpisodeRecord":
        return EpisodeRecord(
            seed=d["seed"], scenario=d["scenario"],
            t=np.asarray(d["t"], dtype=np.float64),
            min_sep=np.asarray(d["min_sep"], dtype=np.float64),
            prox=np.asarray(d["prox"], dtype=bool),
            ssm_actual=np.asarray(d["ssm_actual"], dtype=bool),
            traj_phase=np.asarray(d["traj_phase"], dtype=object),
            arm_phase=np.asarray(d["arm_phase"], dtype=object),
            pelvis_dist=np.asarray(d["pelvis_dist"], dtype=np.float64),
            frame_step=np.asarray(d["frame_step"], dtype=np.int64),
        )


def _scenario_meta(env) -> dict:
    s = env._current_scenario
    cfgd = s.disruption_config
    return dict(
        trajectory_type=s.trajectory_type,
        closest_approach=float(s.closest_approach),
        spawn_distance=float(s.spawn_distance),
        walk_speed=float(s.walk_speed),
        patrol_near_loiter=float(s.patrol_near_loiter),
        patrol_away_loiter=float(s.patrol_away_loiter),
        patrol_away_distance=float(s.patrol_away_distance),
        patrol_excursions=int(s.patrol_excursions),
        reach_period=float(getattr(cfgd, "coworker_reach_period", float("nan"))),
        target_mix=list(getattr(cfgd, "coworker_target_mix", (float("nan"),) * 2)),
        active_arm=str(getattr(cfgd, "coworker_active_arm", "?")),
    )


def _robot_anchor_xy(env) -> np.ndarray:
    hc = env.human_controller
    planner = getattr(hc, "_trajectory_planner", None)
    if planner is not None and getattr(planner, "config", None) is not None:
        return np.asarray(planner.config.robot_pos[:2], dtype=np.float64)
    return np.zeros(2)


def _auto_camera(env, scenario: dict) -> mujoco.MjvCamera:
    """Frame robot + full patrol route from a side-on elevated viewpoint.

    The camera sits perpendicular to the robot->AWAY bearing so the
    depart/return excursion sweeps laterally across the frame.
    """
    hc = env.human_controller
    planner = hc._trajectory_planner
    robot_xy = _robot_anchor_xy(env)
    pts = np.array([wp.position for wp in planner.waypoints] + [robot_xy])
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    radius = float(np.linalg.norm(pts - center, axis=1).max())

    # Bearing from robot to the farthest waypoint (the AWAY pole).
    far = pts[np.argmax(np.linalg.norm(pts - robot_xy, axis=1))]
    bearing = np.degrees(np.arctan2(far[1] - robot_xy[1], far[0] - robot_xy[0]))

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [center[0], center[1], 0.7]
    cam.distance = max(3.2, 1.32 * radius + 1.55)
    cam.elevation = -20.0

    # Default to one of the two side-on choices that keep the depart /
    # return sweep lateral. _pick_visible_azimuth refines the side.
    cam.azimuth = bearing + 90.0
    return cam


def _skin_pixel_count(px: np.ndarray) -> int:
    """Count warm skin-tone pixels — the coworker's capsules (rgba
    0.8/0.6/0.5) are the only warm-hued object in the scene, so this is a
    cheap occlusion test for the human."""
    r = px[..., 0].astype(np.int16)
    g = px[..., 1].astype(np.int16)
    b = px[..., 2].astype(np.int16)
    return int(((r > 110) & (r > g) & (g > b) & (r - b > 30)).sum())


def _pick_visible_azimuth(env, seed: int, sim_seconds: float,
                          width: int, height: int) -> float:
    """Run a throwaway partial episode and render candidate azimuths at
    two key instants — mid-NEAR (the reach panels) and mid-AWAY (the
    patrol-excursion panel) — picking the azimuth that maximises the
    *worst-case* human visibility across both. A single-instant probe can
    choose a view where the AWAY pole hides exactly behind the robot."""
    env.reset(seed=seed)
    scenario = _scenario_meta(env)
    hc = env.human_controller
    planner = hc._trajectory_planner
    robot_xy = _robot_anchor_xy(env)
    mid = 0.5 * (scenario["closest_approach"]
                 + scenario["patrol_away_distance"])

    # Probe instants from the planner's own waypoint schedule.
    t_near = t_away = None
    for wp in planner.waypoints:
        d = float(np.linalg.norm(np.asarray(wp.position) - robot_xy))
        if wp.phase == "loiter" and d < mid and t_near is None:
            t_near = wp.time + 1.5
        if wp.phase == "loiter" and d >= mid and t_away is None:
            t_away = wp.time + 0.6 * scenario["patrol_away_loiter"]
    instants = [t for t in (t_near, t_away)
                if t is not None and t < sim_seconds]
    if not instants:
        instants = [0.5 * sim_seconds]

    base = _auto_camera(env, scenario)
    side = base.azimuth
    candidates = [side + d for d in (0.0, 25.0, -25.0)] \
        + [side + 180.0 + d for d in (0.0, 25.0, -25.0)]

    renderer = _make_renderer(env, width, height)
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    counts = {az: [] for az in candidates}
    for t_target in sorted(instants):
        while hc.t < t_target:
            env.step(zeros)
        for az in candidates:
            base.azimuth = az
            renderer.update_scene(env._mojo.data, camera=base)
            counts[az].append(_skin_pixel_count(renderer.render()))
    renderer.close()

    scores = {az: min(c) for az, c in counts.items()}
    best = max(scores, key=scores.get)
    log.info("  visibility probe (instants %s):",
             [f"{t:.1f}s" for t in sorted(instants)])
    for az in candidates:
        log.info("    az=%6.1f  counts=%s  worst=%d%s", az, counts[az],
                 scores[az], "   <- picked" if az == best else "")
    return float(best)


def _make_renderer(env, width: int, height: int) -> mujoco.Renderer:
    model = env._mojo.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    return mujoco.Renderer(model, height=height, width=width)


def rollout(
    env,
    seed: int,
    sim_seconds: float,
    capture_every: int = 0,
    frames_dir: Optional[Path] = None,
    width: int = 960,
    height: int = 720,
    cam_overrides: Optional[dict] = None,
) -> Tuple[EpisodeRecord, Optional[mujoco.MjvCamera]]:
    """Run one zero-action episode; optionally render every Nth step."""
    import imageio.v2 as imageio

    env.reset(seed=seed)
    scenario = _scenario_meta(env)
    robot_xy = _robot_anchor_xy(env)
    hc = env.human_controller
    planner = hc._trajectory_planner

    renderer = None
    cam = None
    if capture_every > 0:
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)
        renderer = _make_renderer(env, width, height)
        cam = _auto_camera(env, scenario)
        for key, val in (cam_overrides or {}).items():
            if val is None:
                continue
            if key == "lookat":
                cam.lookat[:] = val
            else:
                setattr(cam, key, float(val))
        log.info(
            "  camera: lookat=(%.2f, %.2f, %.2f) dist=%.2f az=%.1f el=%.1f",
            *cam.lookat, cam.distance, cam.azimuth, cam.elevation,
        )

    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    rows: list[tuple] = []
    frame_steps: list[int] = []
    n_steps = int(sim_seconds * CONTROL_HZ)
    pelvis_id = env._human_pelvis_id

    for i in range(n_steps):
        _, _, terminated, truncated, info = env.step(zeros)
        t = float(hc.t)
        safety = info.get("safety", {}) or {}
        sep = float(safety.get("min_separation", np.nan))
        prox = bool(safety.get("proximity_violation", False))
        ssm_act = bool(safety.get("ssm_violation_actual", False))
        _, _, _, traj_phase = planner.get_pose(t)
        cb = env._coworker_controller
        arm_phase = cb.last_phase if cb is not None else "idle"
        pelvis_xy = env._mojo.data.xpos[pelvis_id][:2]
        pdist = float(np.linalg.norm(pelvis_xy - robot_xy))
        rows.append((t, sep, prox, ssm_act, traj_phase, arm_phase, pdist))

        if renderer is not None and i % capture_every == 0:
            renderer.update_scene(env._mojo.data, camera=cam)
            px = renderer.render()
            if frames_dir is not None:
                imageio.imwrite(frames_dir / f"frame_{len(frame_steps):05d}.png", px)
            frame_steps.append(i)

        if terminated or truncated:
            log.warning("episode ended early at t=%.1fs (term=%s trunc=%s)",
                        t, terminated, truncated)
            break

    t_arr, sep_arr, prox_arr, ssm_arr, tp_arr, ap_arr, pd_arr = map(
        np.asarray, zip(*rows))
    rec = EpisodeRecord(
        seed=seed, scenario=scenario,
        t=t_arr.astype(np.float64), min_sep=sep_arr.astype(np.float64),
        prox=prox_arr.astype(bool), ssm_actual=ssm_arr.astype(bool),
        traj_phase=tp_arr, arm_phase=ap_arr,
        pelvis_dist=pd_arr.astype(np.float64),
        frame_step=np.asarray(frame_steps, dtype=np.int64),
    )
    if renderer is not None:
        renderer.close()
    return rec, cam


# --------------------------------------------------------------------------
# Phase segmentation + panel selection
# --------------------------------------------------------------------------

@dataclass
class Segment:
    kind: str  # approach / near / away / depart / walk
    i0: int
    i1: int    # inclusive


def segment_episode(rec: EpisodeRecord) -> List[Segment]:
    """Merge per-step trajectory phases into segments; split loiter into
    NEAR vs AWAY by pelvis distance against the scenario midpoint."""
    mid = 0.5 * (rec.scenario["closest_approach"]
                 + rec.scenario["patrol_away_distance"])
    segs: list[Segment] = []
    i0 = 0
    for i in range(1, len(rec.traj_phase) + 1):
        if i == len(rec.traj_phase) or rec.traj_phase[i] != rec.traj_phase[i0]:
            kind = str(rec.traj_phase[i0])
            if kind == "loiter":
                kind = "near" if np.median(rec.pelvis_dist[i0:i]) < mid else "away"
            segs.append(Segment(kind, i0, i - 1))
            i0 = i
    # Drop blink segments (<0.25 s) that come from waypoint boundaries.
    min_len = int(0.25 * CONTROL_HZ)
    return [s for s in segs if s.i1 - s.i0 + 1 >= min_len]


def find_pattern(segs: List[Segment]) -> dict:
    """Locate approach#1, near#1, depart#1, away#1, return, near#2."""
    out: dict[str, Segment] = {}
    state = 0
    for s in segs:
        if state == 0 and s.kind == "approach":
            out["approach1"] = s; state = 1
        elif state == 1 and s.kind == "near":
            out["near1"] = s; state = 2
        elif state == 2 and s.kind == "depart":
            out["depart1"] = s; state = 3
        elif state == 3 and s.kind == "away":
            out["away1"] = s; state = 4
        elif state == 4 and s.kind == "approach":
            out["return1"] = s; state = 5
        elif state == 5 and s.kind == "near":
            out["near2"] = s; state = 6
        elif state == 6 and s.kind == "depart":
            out["depart2"] = s; state = 7
    return out


def score_episode(rec: EpisodeRecord) -> Tuple[float, dict]:
    segs = segment_episode(rec)
    pat = find_pattern(segs)
    complete = all(k in pat for k in
                   ("approach1", "near1", "depart1", "away1", "return1", "near2"))

    def _viol_in(seg) -> bool:
        # Count only reach-driven violations (arm extended/holding) so the
        # picked episode demonstrates the reaches causing the proximity
        # breaches, not just walk-by transits.
        if seg is None:
            return False
        idx = np.arange(seg.i0, seg.i1 + 1)
        active = np.isin(rec.arm_phase[idx], ("extend", "hold"))
        return bool((rec.prox[idx] & active).any())

    v1 = _viol_in(pat.get("near1"))
    v2 = _viol_in(pat.get("near2"))
    depth = max(0.0, PROX_THRESH - float(np.nanmin(rec.min_sep)))
    away_span = float(rec.pelvis_dist.max())
    # Penalise episodes that sit in violation continuously — the figure
    # should show violations *caused by reaches*, not a saturated state.
    saturation = max(0.0, float(rec.prox.mean()) - 0.5)
    score = (100.0 * complete + 50.0 * v1 + 50.0 * v2
             + 100.0 * depth + 5.0 * away_span - 120.0 * saturation)
    detail = dict(complete=complete, viol_near1=v1, viol_near2=v2,
                  min_sep=float(np.nanmin(rec.min_sep)), away_max=away_span,
                  prox_rate=float(rec.prox.mean()))
    return score, detail


def _nearest_frame(rec: EpisodeRecord, step: int) -> int:
    return int(np.argmin(np.abs(rec.frame_step - step)))


def _deepest_reach(rec: EpisodeRecord, seg: Segment,
                   avoid_steps: List[int], min_gap_s: float = 1.2) -> Optional[int]:
    """Step inside seg with the smallest separation while the arm is
    extended/holding, at least min_gap_s away from already-picked steps."""
    idx = np.arange(seg.i0, seg.i1 + 1)
    arm_ok = np.isin(rec.arm_phase[idx], ("extend", "hold"))
    cand = idx[arm_ok] if arm_ok.any() else idx
    gap = int(min_gap_s * CONTROL_HZ)
    cand = np.array([i for i in cand
                     if all(abs(i - a) >= gap for a in avoid_steps)])
    if len(cand) == 0:
        return None
    return int(cand[np.argmin(rec.min_sep[cand])])


def select_panels(rec: EpisodeRecord, n_panels: int = 8) -> List[dict]:
    """Pick representative steps for the grid. Returns list of dicts with
    step, frame, letter, label."""
    segs = segment_episode(rec)
    pat = find_pattern(segs)
    picks: list[tuple[str, int]] = []  # (label, step)

    def mid(seg: Segment) -> int:
        return (seg.i0 + seg.i1) // 2

    if "approach1" in pat:
        picks.append(("walks in", mid(pat["approach1"])))
    if "near1" in pat:
        s1 = _deepest_reach(rec, pat["near1"], [p[1] for p in picks])
        if s1 is not None:
            picks.append(("reaches toward robot", s1))
        if n_panels >= 8:
            # Contrast frame: arm withdrawn between reach cycles (ideally
            # back out of proximity), showing the violations are caused by
            # the reaches themselves. Drawn from either NEAR visit.
            near_segs = [pat[k] for k in ("near1", "near2") if k in pat]
            idx = np.concatenate([np.arange(s.i0, s.i1 + 1)
                                  for s in near_segs])
            idle = idx[(rec.arm_phase[idx] == "idle")
                       & np.isfinite(rec.min_sep[idx])]
            gap = int(1.2 * CONTROL_HZ)
            idle = np.array([i for i in idle
                             if all(abs(i - a) >= gap for _, a in picks)])
            if len(idle):
                picks.append(("between reaches",
                              int(idle[np.argmax(rec.min_sep[idle])])))
            else:
                s2 = _deepest_reach(rec, pat["near1"], [p[1] for p in picks])
                if s2 is not None:
                    picks.append(("repeated reach cycles", s2))
    if "depart1" in pat:
        picks.append(("walks away", mid(pat["depart1"])))
    if "away1" in pat:
        seg = pat["away1"]
        idx = np.arange(seg.i0, seg.i1 + 1)
        picks.append(("loiters away", int(idx[np.argmax(rec.pelvis_dist[idx])])))
    if "return1" in pat:
        picks.append(("returns", mid(pat["return1"])))
    if "near2" in pat:
        s3 = _deepest_reach(rec, pat["near2"], [p[1] for p in picks])
        if s3 is not None:
            picks.append(("reaches again", s3))
        if len(picks) < n_panels:
            s4 = _deepest_reach(rec, pat["near2"], [p[1] for p in picks])
            if s4 is not None:
                picks.append(("reach cycles continue", s4))
    if "depart2" in pat and len(picks) < n_panels:
        picks.append(("departs again", mid(pat["depart2"])))

    picks = sorted(picks[:n_panels], key=lambda p: p[1])
    letters = "abcdefghij"
    return [
        dict(letter=letters[k], label=lbl, step=int(st),
             frame=_nearest_frame(rec, st))
        for k, (lbl, st) in enumerate(picks)
    ]


# --------------------------------------------------------------------------
# Figure composition
# --------------------------------------------------------------------------

def _status(sep: float) -> Tuple[str, str]:
    if not np.isfinite(sep) or sep < PROX_THRESH:
        return "VIOLATION", "#dc2828"
    if sep < NEAR_THRESH:
        return "NEAR", "#e6aa1e"
    return "SAFE", "#28b450"


def compose_figure(rec: EpisodeRecord, panels: List[dict], frames_dir: Path,
                   out_base: Path, human_model: str) -> List[Path]:
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    n = len(panels)
    ncols = 4 if n > 4 else max(n, 1)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(4.1 * ncols, 3.1 * nrows + 2.6))
    gs = gridspec.GridSpec(nrows + 1, ncols, figure=fig,
                           height_ratios=[1.0] * nrows + [0.62],
                           hspace=0.06, wspace=0.025,
                           left=0.034, right=0.988, top=0.995, bottom=0.075)

    for k, p in enumerate(panels):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        img = imageio.imread(frames_dir / f"frame_{p['frame']:05d}.png")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        i = p["step"]
        sep = float(rec.min_sep[i])
        status, color = _status(sep)
        viol = bool(rec.prox[i]) or not np.isfinite(sep)

        for sp in ax.spines.values():
            sp.set_color("#dc2828" if viol else "#444444")
            sp.set_linewidth(3.5 if viol else 0.8)

        ax.text(0.025, 0.965, f"({p['letter']})  {p['label']}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold", color="white",
                bbox=dict(facecolor="black", alpha=0.66, pad=4,
                          edgecolor="none"))
        ax.text(0.975, 0.965, f"t = {rec.t[i]:.1f} s",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5, color="white",
                bbox=dict(facecolor="black", alpha=0.66, pad=3.5,
                          edgecolor="none"))
        sep_txt = "contact" if not np.isfinite(sep) else f"{sep:.2f} m"
        ax.text(0.025, 0.038, f"min sep  {sep_txt}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, fontweight="bold", color=color,
                bbox=dict(facecolor="black", alpha=0.66, pad=4,
                          edgecolor="none"))
        ax.text(0.975, 0.038, status,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(facecolor=color, alpha=0.92, pad=4,
                          edgecolor="none"))

    # --- Timeline ---
    axt = fig.add_subplot(gs[nrows, :])
    t = rec.t
    axt.plot(t, rec.min_sep, color="#1f3b73", lw=1.7,
             label="min human–robot separation")
    axt.plot(t, rec.pelvis_dist, color="#888888", lw=1.2, ls="--",
             label="pelvis distance to robot")
    axt.axhline(PROX_THRESH, color="#dc2828", lw=1.1,
                label=f"proximity threshold ({PROX_THRESH:.1f} m)")
    axt.axhline(NEAR_THRESH, color="#e6aa1e", lw=0.9, ls=":")
    viol = rec.prox | ~np.isfinite(rec.min_sep)
    axt.fill_between(t, 0, 1, where=viol, transform=axt.get_xaxis_transform(),
                     color="#dc2828", alpha=0.16, linewidth=0,
                     label="proximity violation")

    # Shade the away excursion (depart start -> return end).
    pat = find_pattern(segment_episode(rec))
    if "depart1" in pat and "return1" in pat:
        t0, t1 = t[pat["depart1"].i0], t[pat["return1"].i1]
        axt.axvspan(t0, t1, color="#5588cc", alpha=0.10, linewidth=0)
        axt.text(0.5 * (t0 + t1), 0.94, "away excursion",
                 transform=axt.get_xaxis_transform(), ha="center", va="top",
                 fontsize=9.5, color="#33567f", style="italic")

    ymax = max(np.nanmax(rec.pelvis_dist), np.nanmax(rec.min_sep)) * 1.12
    for p in panels:
        ti = t[p["step"]]
        axt.axvline(ti, color="#222222", lw=0.7, alpha=0.55)
        axt.text(ti, ymax * 0.99, f"({p['letter']})", ha="center", va="top",
                 fontsize=9.5, fontweight="bold", color="#222222")
    axt.set_ylim(0, ymax)
    axt.set_xlim(t[0], t[-1])
    axt.set_xlabel("episode time (s)", fontsize=11)
    axt.set_ylabel("human–robot distance (m)", fontsize=10.5, labelpad=2)
    axt.tick_params(labelsize=9.5)
    axt.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)
    prox_pct = 100.0 * float(viol.mean())
    fig.text(0.988, 0.006,
             f"{prox_pct:.0f}% of steps inside {PROX_THRESH:.1f} m  ·  "
             f"min separation {np.nanmin(rec.min_sep):.2f} m  ·  "
             f"{human_model.upper()} coworker, stationary robot",
             ha="right", va="bottom", fontsize=9, color="#333333")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        outs.append(path)
    plt.close(fig)
    return outs


# --------------------------------------------------------------------------
# Camera probe
# --------------------------------------------------------------------------

def probe_camera(env, seed: int, args, out_dir: Path) -> None:
    """Render one mid-NEAR frame from several candidate camera angles."""
    import imageio.v2 as imageio

    env.reset(seed=seed)
    scenario = _scenario_meta(env)
    hc = env.human_controller
    # Step until mid of first NEAR loiter (approach time + half dwell).
    t_target = (scenario["spawn_distance"] - scenario["closest_approach"]) \
        / max(scenario["walk_speed"], 0.1) + 0.5 * scenario["patrol_near_loiter"]
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    while hc.t < t_target:
        env.step(zeros)

    renderer = _make_renderer(env, args.width, args.height)
    base = _auto_camera(env, scenario)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "auto": dict(),
        "flip": dict(azimuth=base.azimuth + 180.0),
        "minus45": dict(azimuth=base.azimuth - 45.0),
        "plus45": dict(azimuth=base.azimuth + 45.0),
        "higher": dict(elevation=-32.0),
        "closer": dict(distance=base.distance * 0.78),
        "wider": dict(distance=base.distance * 1.25),
    }
    for name, over in variants.items():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = base.lookat
        cam.distance, cam.azimuth, cam.elevation = (
            base.distance, base.azimuth, base.elevation)
        for k, v in over.items():
            setattr(cam, k, v)
        renderer.update_scene(env._mojo.data, camera=cam)
        imageio.imwrite(out_dir / f"probe_{name}.png", renderer.render())
        log.info("  probe %-8s az=%.1f el=%.1f dist=%.2f", name,
                 cam.azimuth, cam.elevation, cam.distance)
    renderer.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="saucepan", choices=list(TASK_MAP))
    p.add_argument("--human-model", default="g1", choices=["g1", "smplh"])
    p.add_argument("--seed", type=int, default=None,
                   help="Episode seed; if unset, scouts --scout seeds and picks.")
    p.add_argument("--scout", type=int, default=8,
                   help="Number of seeds to scout (no rendering).")
    p.add_argument("--scout-base-seed", type=int, default=0)
    p.add_argument("--sim-seconds", type=float, default=48.0)
    p.add_argument("--capture-every", type=int, default=2,
                   help="Render every Nth env step (20 Hz steps; 2 -> 10 fps).")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--panels", type=int, default=8)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "disruption_figure")
    p.add_argument("--fig-base", type=Path,
                   default=REPO_ROOT / "docs" / "figures"
                   / "fig_coworker_patrol_disruption")
    # Art-direction pins (default: sample from the training band).
    p.add_argument("--pin-closest", type=float, default=None)
    p.add_argument("--pin-p-ee", type=float, default=None)
    p.add_argument("--pin-near-loiter", type=float, default=None)
    p.add_argument("--pin-away-distance", type=float, default=None)
    p.add_argument("--pin-excursions", type=int, default=None)
    # Camera overrides applied on top of auto-framing.
    p.add_argument("--cam-azimuth", type=float, default=None)
    p.add_argument("--cam-elevation", type=float, default=None)
    p.add_argument("--cam-distance", type=float, default=None)
    p.add_argument("--probe-camera", action="store_true")
    p.add_argument("--recompose", action="store_true",
                   help="Recompose the figure from an existing dump (needs --seed).")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recompose:
        if args.seed is None:
            raise SystemExit("--recompose requires --seed of an existing dump")
        dump = out_dir / f"seed_{args.seed}"
        rec = EpisodeRecord.from_json(
            json.loads((dump / "series.json").read_text()))
        panels = select_panels(rec, args.panels)
        outs = compose_figure(rec, panels, dump / "frames", args.fig_base,
                              args.human_model)
        for o in outs:
            log.info("wrote %s", o)
        return 0

    env = build_env(args.task, args.human_model, args)
    try:
        if args.probe_camera:
            probe_camera(env, args.seed if args.seed is not None else 0,
                         args, out_dir / "camera_probe")
            return 0

        seed = args.seed
        if seed is None:
            log.info("scouting %d seeds (no rendering)...", args.scout)
            best, best_score = None, -1.0
            for s in range(args.scout_base_seed,
                           args.scout_base_seed + args.scout):
                rec, _ = rollout(env, s, args.sim_seconds, capture_every=0)
                score, detail = score_episode(rec)
                log.info("  seed %-3d score %-7.1f %s", s, score, detail)
                if score > best_score:
                    best, best_score = s, score
            seed = best
            log.info("picked seed %d (score %.1f)", seed, best_score)

        dump = out_dir / f"seed_{seed}"
        frames_dir = dump / "frames"
        cam_azimuth = args.cam_azimuth
        if cam_azimuth is None:
            log.info("probing camera side for occlusion (seed %d)...", seed)
            cam_azimuth = _pick_visible_azimuth(
                env, seed, args.sim_seconds, args.width, args.height)
        cam_overrides = dict(azimuth=cam_azimuth,
                             elevation=args.cam_elevation,
                             distance=args.cam_distance)
        log.info("rendering seed %d (%.0f s sim, every %d steps)...",
                 seed, args.sim_seconds, args.capture_every)
        rec, _ = rollout(env, seed, args.sim_seconds,
                         capture_every=args.capture_every,
                         frames_dir=frames_dir,
                         width=args.width, height=args.height,
                         cam_overrides=cam_overrides)
        dump.mkdir(parents=True, exist_ok=True)
        (dump / "series.json").write_text(json.dumps(rec.to_json()))
        log.info("scenario: %s", rec.scenario)

        # Byproduct mp4 of the captured frames.
        try:
            import imageio.v2 as imageio
            frames = [imageio.imread(frames_dir / f"frame_{k:05d}.png")
                      for k in range(len(rec.frame_step))]
            fps = max(1, round(CONTROL_HZ / args.capture_every))
            imageio.mimsave(str(dump / "episode.mp4"), frames, fps=fps,
                            macro_block_size=1)
        except Exception as e:  # pragma: no cover
            log.warning("mp4 write failed: %s", e)

        panels = select_panels(rec, args.panels)
        if len(panels) < 6:
            log.warning("only %d panels selected — pattern incomplete; "
                        "consider another seed", len(panels))
        for pnl in panels:
            i = pnl["step"]
            log.info("  panel (%s) %-24s t=%5.1fs sep=%.2f arm=%s",
                     pnl["letter"], pnl["label"], rec.t[i],
                     rec.min_sep[i], rec.arm_phase[i])
        outs = compose_figure(rec, panels, frames_dir, args.fig_base,
                              args.human_model)
        for o in outs:
            log.info("wrote %s", o)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
