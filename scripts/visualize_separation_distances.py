#!/usr/bin/env python
"""Visual scale reference: render the G1 coworker at chosen human-robot separations.

Builds a SafetyBiGym G1 scene, slides the coworker (a mocap body) along the approach axis
toward the H1 robot, recomputing the closest-joint ``min_separation`` at each position
(no physics stepping — the robot stays frozen in its reset pose), and renders the frame
whose separation is closest to each requested target. Outputs a labelled montage so you
can *see* what e.g. 0.34 / 0.50 / 0.54 / 0.94 m looks like — separations across the
columns, camera angles down the rows.

The default targets are the ISO 15066 S_p values from the proximity-vs-SSM explainer
(robot stopped / ~1 / ~2 m/s) plus the 0.5 m geometric proximity threshold. Column
headers are coloured by the proximity rule: red if `min_separation < 0.5 m` (proximity
violation), green otherwise. Without --views each separation is rendered from 4 preset
camera angles (default / orbit_left / orbit_right / top_down); pass --views to choose
absolute azimuth,elevation degree pairs.

    python scripts/visualize_separation_distances.py
    python scripts/visualize_separation_distances.py --targets 0.3 0.5 1.0 1.5 \
        --task saucepan_to_hob --out results/separation_scale.png
    python scripts/visualize_separation_distances.py --range 0.1 0.7 0.1 \
        --views 90,-10 180,-20 90,-89 --out results/sep_angles.png

Run with the plain venv Python (``venv/bin/python ...``), **not** ``mjpython`` — this script
renders offscreen, and ``mjpython``'s main-thread interactive viewer conflicts with offscreen
GLFW on macOS (``NSWindow should only be instantiated on the main thread``). Headless Linux:
export ``MUJOCO_GL=egl`` first.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("visualize_separation_distances")

PROXIMITY_THRESHOLD = 0.3  # SSMConfig.proximity_threshold default (0.3 as of 2026-05-30; was 0.5)


def _unwrap(env):
    e = env
    while hasattr(e, "env") and not hasattr(e, "_human_pelvis_mocapid"):
        e = e.env
    return e.unwrapped if hasattr(e, "unwrapped") and hasattr(e.unwrapped, "_human_pelvis_mocapid") else e


def _measure_min_separation(core) -> float:
    """Recompute closest-joint min_separation for the current sim state."""
    import mujoco

    mujoco.mj_forward(core._mojo.model, core._mojo.data)
    if getattr(core, "_step_contacts", None) is None:
        core._step_contacts = []
    core._aggregate_safety_info()
    return float(core._step_safety_info.min_separation)


def _scan(core, *, n: int = 70, d_lo: float = 0.2, d_hi: float = 2.6):
    """Slide the coworker along the robot→human axis; record ``(d, min_sep)``.

    No rendering happens here — framing is deferred to :func:`_render_view` so
    each picked separation can be shot from several camera angles without
    re-running the scan once per angle. Returns ``(samples, ctx)`` where ``ctx``
    is the pose context consumed by :func:`_pose` to re-place the coworker.
    """
    mid = core._human_pelvis_mocapid
    if mid is None:
        raise RuntimeError("No human mocap pelvis found — is inject_human enabled?")

    data = core._mojo.data
    robot_xy = core._robot_ssm_state()[0].mean(axis=0)[:2]
    human_pos0 = np.asarray(data.mocap_pos[mid], dtype=float).copy()
    z = float(human_pos0[2])

    direction = human_pos0[:2] - robot_xy
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])

    ctx = (mid, robot_xy, unit, z)
    samples: List[Tuple[float, float]] = []
    for d in np.linspace(d_hi, d_lo, n):
        sep = _pose(core, ctx, float(d))
        samples.append((float(d), sep))
    return samples, ctx


def _pose(core, ctx, d: float) -> float:
    """Place the coworker pelvis at distance ``d`` along the axis; return min_sep."""
    mid, robot_xy, unit, z = ctx
    xy = robot_xy + unit * d
    core._mojo.data.mocap_pos[mid] = [xy[0], xy[1], z]
    return _measure_min_separation(core)


# Default camera angles, expressed as azimuth/elevation *offsets* (deg) from
# MuJoCo's scene-centred free-camera default so they stay sensible regardless of
# how the scene happens to be oriented.
_DEFAULT_VIEWS = [
    ("default", 0.0, 0.0),
    ("orbit_left", 55.0, 0.0),
    ("orbit_right", -55.0, 0.0),
    ("top_down", 0.0, -40.0),
]


def _clamp_elevation(e: float) -> float:
    return max(-89.0, min(89.0, e))


def _resolve_views(core, abs_views):
    """Return ``([(label, azimuth, elevation, distance, lookat)], viewer)``.

    Distance/lookat are inherited from MuJoCo's default (scene-centred) framing.
    ``abs_views`` (from ``--views``) are absolute azimuth,elevation pairs; when
    ``None`` the :data:`_DEFAULT_VIEWS` offsets are applied to the base camera.
    """
    core.render()  # force viewer creation + default cam config
    viewer = core.mujoco_renderer._viewers["rgb_array"]
    cam = viewer.cam
    base_az, base_el = float(cam.azimuth), float(cam.elevation)
    base_dist = float(cam.distance)
    base_lookat = np.array(cam.lookat, dtype=float).copy()

    views = []
    if abs_views:
        for az, el in abs_views:
            views.append((f"az{az:.0f}_el{el:.0f}", float(az),
                          _clamp_elevation(float(el)), base_dist, base_lookat))
    else:
        for label, daz, dele in _DEFAULT_VIEWS:
            views.append((label, base_az + daz,
                          _clamp_elevation(base_el + dele), base_dist, base_lookat))
    return views, viewer


def _render_view(core, viewer, az, el, dist, lookat):
    """Set the free camera, render one offscreen frame. The render path only
    resets ``cam.type``/``fixedcamid`` (free camera, no fixed cam named 'track'),
    so these azimuth/elevation/distance/lookat overrides persist per call."""
    viewer.cam.azimuth = float(az)
    viewer.cam.elevation = float(el)
    viewer.cam.distance = float(dist)
    viewer.cam.lookat[:] = lookat
    frame = core.render()
    return np.asarray(frame) if frame is not None else None


def _nearest(samples, target: float):
    return min(samples, key=lambda s: abs(s[1] - target))


def _montage(picks, views, out: Path, *, threshold: float) -> None:
    """Grid montage: camera angle per row, separation target per column.

    ``picks`` is ``[(d, sep, {view_label: frame})]``; ``views`` is the resolved
    ``[(label, az, el, dist, lookat)]`` list (row order)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(views)
    n_cols = len(picks)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.3 * n_cols, 3.5 * n_rows), squeeze=False
    )
    for r, view in enumerate(views):
        vlabel = view[0]
        for c, (d, sep, frames) in enumerate(picks):
            ax = axes[r][c]
            frame = frames.get(vlabel)
            if frame is None:
                ax.axis("off")
                continue
            ax.imshow(frame)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:  # column header: the separation + proximity verdict
                violated = sep < threshold
                colour = "tab:red" if violated else "tab:green"
                tag = "BELOW thr" if violated else "above thr"
                ax.set_title(f"{sep:.2f} m  ({tag})", color=colour, fontsize=11)
            if c == 0:  # row label: the camera angle
                ax.set_ylabel(vlabel, fontsize=11)
    fig.suptitle("Human–robot separation (columns) × camera angle (rows) — closest-joint "
                 f"min-separation; red header = below the {threshold:.2f} m proximity threshold",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", type=float, nargs="+", default=[0.34, 0.50, 0.54, 0.94],
                   help="Specific separation distances (m) to render (ignored if --range given).")
    p.add_argument("--range", type=float, nargs=3, metavar=("LO", "HI", "STEP"), default=None,
                   help="Sweep a range of separations, e.g. --range 0.1 1.2 0.1, laid out in a grid.")
    p.add_argument("--threshold", type=float, default=PROXIMITY_THRESHOLD,
                   help="Candidate proximity threshold (m) used to colour frames red/green. "
                        "Vary it to calibrate where 'too close' should be.")
    p.add_argument("--views", nargs="+", default=None, metavar="AZ,EL",
                   help="Camera angles as absolute azimuth,elevation degree pairs, e.g. "
                        "--views 90,-10 180,-20 90,-89 — one montage row per angle. "
                        "Default: 4 scene-relative preset angles "
                        "(default / orbit_left / orbit_right / top_down).")
    p.add_argument("--task", default="saucepan_to_hob")
    p.add_argument("--disruption", default="coworker_train")
    p.add_argument("--human-model", choices=("g1", "smplh"), default="g1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=70, help="Positions to scan along the axis.")
    p.add_argument("--out", type=Path, default=Path("results/separation_scale.png"))
    p.add_argument("--save-frames", action="store_true", help="Also save each picked frame as a PNG.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    # This script renders OFFSCREEN; mjpython's main-thread viewer crashes it on macOS
    # (NSWindow main-thread assertion). Fail fast with guidance.
    if "mjpython" in os.path.basename(sys.executable).lower() or "mjpython" in sys.argv[0].lower():
        raise SystemExit(
            "Run this offscreen-render script with the plain venv Python, not mjpython:\n"
            "  venv/bin/python scripts/visualize_separation_distances.py ...\n"
            "(mjpython's interactive viewer conflicts with offscreen rendering on macOS.)"
        )

    from safety_bigym.benchmark.env_build import build_g1_gym_env

    # Build the list of separation targets: a swept range, or explicit values.
    if args.range is not None:
        lo, hi, step = args.range
        targets = list(np.round(np.arange(lo, hi + step / 2.0, step), 3))
    else:
        targets = list(args.targets)
    if not targets:
        raise SystemExit("No targets — pass --targets or --range.")
    logger.info("Candidate proximity threshold for colouring: %.2f m", args.threshold)

    # Parse explicit camera angles (absolute azimuth,elevation), if given.
    abs_views = None
    if args.views:
        abs_views = []
        for tok in args.views:
            try:
                az, el = (float(x) for x in tok.split(","))
            except Exception:
                raise SystemExit(f"--views entries must be 'az,el' degree pairs, got {tok!r}")
            abs_views.append((az, el))

    env = build_g1_gym_env(args.task, args.disruption, "off", human_model=args.human_model)
    try:
        env.reset(seed=args.seed)
        core = _unwrap(env)
        # Scan far enough to cover the largest requested separation.
        d_hi = max(2.6, max(targets) + 1.0)
        n = max(int(args.samples), int((d_hi - 0.2) / 0.03))
        samples, ctx = _scan(core, n=n, d_hi=d_hi)
        if not samples:
            raise SystemExit("No positions scanned — is inject_human enabled?")
        sep_range = (min(s[1] for s in samples), max(s[1] for s in samples))
        logger.info("Scanned %d positions; achievable min_separation range %.2f–%.2f m",
                    len(samples), sep_range[0], sep_range[1])

        views, viewer = _resolve_views(core, abs_views)
        logger.info("Rendering %d camera angle(s): %s",
                    len(views), ", ".join(v[0] for v in views))

        # Re-pose to each target separation and shoot it from every angle.
        picks = []
        for t in targets:
            d, sep = _nearest(samples, t)
            _pose(core, ctx, d)
            frames = {label: _render_view(core, viewer, az, el, dist, lookat)
                      for (label, az, el, dist, lookat) in views}
            if all(f is None for f in frames.values()):
                raise SystemExit("No frames rendered — check the GL backend (set MUJOCO_GL=egl on headless).")
            picks.append((d, sep, frames))
            note = "" if abs(sep - t) < 0.06 else "  (closest achievable)"
            logger.info("target %.2f m -> pelvis d=%.2f m, actual min-sep=%.2f m%s", t, d, sep, note)

        _montage(picks, views, args.out, threshold=args.threshold)
        logger.info("Wrote montage -> %s", args.out)
        if args.save_frames:
            import matplotlib.pyplot as plt
            for (d, sep, frames) in picks:
                for vlabel, frame in frames.items():
                    if frame is None:
                        continue
                    fp = args.out.with_name(f"{args.out.stem}_{sep:.2f}m_{vlabel}.png")
                    plt.imsave(fp, frame)
                    logger.info("  frame -> %s", fp)
        print(args.out)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
