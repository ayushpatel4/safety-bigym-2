#!/usr/bin/env python
"""Regenerate task-suite figure panels via expert-demo replay (no snapshot).

The v16 task_suite.png panels for dishwasher_close (human cropped out of
frame) and drawers_open_all (human clipping through the cabinet) need
replacing, but the unconstrained baseline snapshots only exist on the GPU
box. Mock strategy (user-approved): replay cached BiGym expert demos at
native 500 Hz in the live safety env (G1 coworker, COWORKER_PATROL,
canonical stage-2 training band) so the robot performs the real task while
the coworker behaves exactly as in training. Per-step ground-truth
separation is recorded in the same pass, so the annotated "separation
here" stays honest. Frames + full sim states are captured so any instant
can be re-rendered from any camera afterwards.

Workflow (from safety_bigym/, no AMASS needed for G1)::

    ./venv/bin/python scripts/remake_task_suite_panels.py --task dishwasher_close --list-demos
    ./venv/bin/python scripts/remake_task_suite_panels.py --task dishwasher_close --probe-camera
    ./venv/bin/python scripts/remake_task_suite_panels.py --task dishwasher_close \
        --demos 0,1,2 --scenario-seeds 0,1
    ./venv/bin/python scripts/remake_task_suite_panels.py --task dishwasher_close --contact-sheet
    ./venv/bin/python scripts/remake_task_suite_panels.py --task dishwasher_close \
        --rerender ep_d0_s1:1200 --cam-azimuth 140 --cam-distance 2.8
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import mujoco
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("remake_task_suite_panels")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROX_THRESH = 0.30
NEAR_THRESH = 0.50
CONTROL_HZ = 500  # native demo frequency: 1 env step per recorded action

DEMO_CACHE = Path.home() / ".bigym" / "demonstrations" / "0.9.0"
ACTION_MODE_DIR = (
    "JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute"
)

TASK_MAP = {
    "saucepan_to_hob": ("bigym.envs.pick_and_place:SaucepanToHob", "SaucepanToHob"),
    "dishwasher_close": ("bigym.envs.dishwasher:DishwasherClose", "DishwasherClose"),
    "drawers_open_all": ("bigym.envs.cupboards:DrawersAllOpen", "DrawersAllOpen"),
}

# Canonical stage-2 COWORKER training band (cfgs/disruption/coworker_train.yaml).
COWORKER_TRAIN_KNOBS = dict(
    coworker_closest_approach_range=(0.60, 0.95),
    coworker_reach_period_range=(1.3, 2.2),
    coworker_target_mix_p_ee_range=(0.45, 0.72),
    coworker_near_loiter_range=(9.0, 14.0),
    coworker_walk_speed_range=(1.0, 1.4),
)

# Hand-tuned per-task default cameras (refined via --probe-camera).
# Dishwasher: unit at (1.2, 0), open door tip ~(0.63, 0), robot pelvis
# spawns at (0, -0.8) — frame the volume between them.
DEFAULT_CAMERA = {
    "dishwasher_close": dict(lookat=(0.6, -0.3, 0.5), distance=3.2,
                             azimuth=160.0, elevation=-24.0),
    "drawers_open_all": dict(lookat=(0.6, -0.3, 0.5), distance=3.2,
                             azimuth=160.0, elevation=-24.0),
    "saucepan_to_hob": dict(lookat=(0.0, 0.0, 0.55), distance=3.0,
                            azimuth=160.0, elevation=-24.0),
}


class _ScenarioSeedHolder:
    """Mutable override for the scenario seed used at env.reset.

    env.reset(seed=demo.seed) must use the demo's seed for the task
    randomisation, but the coworker scenario should vary across replays of
    the same demo. The patched sampler reads this holder instead of the
    reset seed.
    """

    value: Optional[int] = None


# --------------------------------------------------------------------------
# Env + demos
# --------------------------------------------------------------------------

def _load_task_cls(task_key: str):
    module_path, cls_name = TASK_MAP[task_key][0].rsplit(":", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def build_env(task_key: str, holder: _ScenarioSeedHolder):
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from safety_bigym import HumanConfig, SafetyConfig, make_safety_env
    from safety_bigym.scenarios import DisruptionType, ParameterSpace, ScenarioSampler

    space = ParameterSpace(
        clip_paths=[],
        disruption_weights={DisruptionType.COWORKER: 1.0},
        coworker_trajectory_weights={"COWORKER_PATROL": 1.0},
        **COWORKER_TRAIN_KNOBS,
    )
    sampler = ScenarioSampler(parameter_space=space, motion_dir=None)

    base = sampler.sample_scenario

    def _patched(seed):
        s = base(holder.value if holder.value is not None else seed)
        s.trajectory_type = "COWORKER_PATROL"
        return s

    sampler.sample_scenario = _patched  # type: ignore[assignment]

    env = make_safety_env(
        task_cls=_load_task_cls(task_key),
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        safety_config=SafetyConfig(log_violations=False, terminate_on_violation=False),
        human_config=HumanConfig(human_model="g1"),
        scenario_sampler=sampler,
        inject_human=True,
        control_frequency=CONTROL_HZ,
    )
    return env


def load_demos(task_key: str, indices: List[int]):
    """Load lightweight demos (native 500 Hz actions + seed) from cache."""
    from demonstrations.demo import Demo

    demo_dir = DEMO_CACHE / TASK_MAP[task_key][1] / ACTION_MODE_DIR / "lightweight"
    files = sorted(demo_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no cached lightweight demos under {demo_dir}")
    out = []
    for i in indices:
        if i >= len(files):
            raise SystemExit(f"demo index {i} out of range ({len(files)} cached)")
        demo = Demo.from_safetensors(files[i])
        actions = np.stack([ts.executed_action for ts in demo.timesteps])
        out.append(dict(index=i, file=files[i].name, seed=demo.seed,
                        actions=actions))
    return out


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def _make_renderer(env, width: int, height: int) -> mujoco.Renderer:
    model = env._mojo.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    return mujoco.Renderer(model, height=height, width=width)


def _camera(task_key: str, overrides: Optional[dict] = None) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    spec = dict(DEFAULT_CAMERA[task_key])
    spec.update({k: v for k, v in (overrides or {}).items() if v is not None})
    cam.lookat[:] = spec["lookat"]
    cam.distance = spec["distance"]
    cam.azimuth = spec["azimuth"]
    cam.elevation = spec["elevation"]
    return cam


def _skin_stats(px: np.ndarray):
    """Skin-tone pixel count + bbox for the G1 capsule coworker."""
    r = px[..., 0].astype(np.int16)
    g = px[..., 1].astype(np.int16)
    b = px[..., 2].astype(np.int16)
    mask = (r > 110) & (r > g) & (g > b) & (r - b > 30)
    n = int(mask.sum())
    if n == 0:
        return 0, None
    ys, xs = np.nonzero(mask)
    return n, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _save_state(env) -> dict:
    d = env._mojo.data
    return dict(
        qpos=np.array(d.qpos), qvel=np.array(d.qvel),
        mocap_pos=np.array(d.mocap_pos), mocap_quat=np.array(d.mocap_quat),
        time=float(d.time),
    )


def _restore_state(env, st: dict) -> None:
    d = env._mojo.data
    d.qpos[:] = st["qpos"]
    d.qvel[:] = st["qvel"]
    d.mocap_pos[:] = st["mocap_pos"]
    d.mocap_quat[:] = st["mocap_quat"]
    d.time = st["time"]
    mujoco.mj_forward(env._mojo.model, d)


# --------------------------------------------------------------------------
# Replay + capture
# --------------------------------------------------------------------------

def replay_episode(env, holder, demo: dict, scen_seed: int, out_dir: Path,
                   capture_every: int, pad_seconds: float,
                   width: int, height: int, task_key: str) -> dict:
    import imageio.v2 as imageio

    holder.value = scen_seed
    env.reset(seed=demo["seed"])
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer = _make_renderer(env, width, height)
    cam = _camera(task_key)

    series = dict(min_sep=[], prox=[], t=[])
    captures = []  # dicts: step, t, min_sep, skin_px, bbox, padded
    states = []
    terminated = truncated = False
    success_step = None

    actions = demo["actions"]
    n_pad = int(pad_seconds * CONTROL_HZ)
    total = len(actions) + n_pad

    for i in range(total):
        padded = i >= len(actions)
        act = actions[min(i, len(actions) - 1)]
        _, _, terminated, truncated, info = env.step(act)
        safety = info.get("safety", {}) or {}
        sep = float(safety.get("min_separation", np.nan))
        prox = bool(safety.get("proximity_violation", False))
        series["min_sep"].append(sep)
        series["prox"].append(prox)
        series["t"].append(float(env._mojo.data.time))

        if terminated and success_step is None:
            success_step = i

        if i % capture_every == 0:
            renderer.update_scene(env._mojo.data, camera=cam)
            px = renderer.render()
            skin_n, bbox = _skin_stats(px)
            k = len(captures)
            imageio.imwrite(frames_dir / f"cap_{k:04d}.jpg", px, quality=88)
            pelvis = env._mojo.data.xpos[env._human_pelvis_id]
            captures.append(dict(
                k=k, step=i, t=float(env._mojo.data.time), min_sep=sep,
                prox=prox, skin_px=skin_n, bbox=bbox, padded=padded,
                pelvis=[round(float(pelvis[0]), 3), round(float(pelvis[1]), 3)],
            ))
            states.append(_save_state(env))

        if terminated or truncated:
            # BiGym success-terminates; keep stepping only through padding
            # (hold pose) so the human keeps moving for more candidates.
            if not padded and i < len(actions) - 1:
                continue  # terminate flag stays up; env.step still advances
    renderer.close()

    np.savez_compressed(
        out_dir / "states.npz",
        **{f"s{j}_{k}": v for j, st in enumerate(states) for k, v in st.items()},
        n_states=len(states),
    )
    meta = dict(
        demo_index=demo["index"], demo_file=demo["file"], demo_seed=demo["seed"],
        scenario_seed=scen_seed, steps=len(series["min_sep"]),
        demo_len=len(actions), success_step=success_step,
        terminated=bool(terminated), truncated=bool(truncated),
        near_frac=float(np.mean(np.asarray(series["min_sep"]) < NEAR_THRESH)),
        prox_frac=float(np.mean(np.asarray(series["min_sep"]) < PROX_THRESH)),
        min_sep=float(np.nanmin(series["min_sep"])),
        captures=captures,
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "series.json").write_text(json.dumps(series))
    return meta


# --------------------------------------------------------------------------
# Contact sheet + rerender
# --------------------------------------------------------------------------

def contact_sheet(task_dir: Path, width: int, top_k: int = 24,
                  sep_lo: float = 0.15, sep_hi: float = 0.65,
                  min_skin: int = 800, edge_margin: int = 12,
                  exclude_box: Optional[tuple] = None) -> Path:
    import imageio.v2 as imageio

    cands = []
    for ep_dir in sorted(task_dir.glob("ep_*")):
        meta = json.loads((ep_dir / "meta.json").read_text())
        for c in meta["captures"]:
            if not (sep_lo <= (c["min_sep"] or 9) <= sep_hi):
                continue
            if c["skin_px"] < min_skin or c["bbox"] is None:
                continue
            x0, y0, x1, y1 = c["bbox"]
            if x0 < edge_margin or x1 > width - edge_margin:
                continue
            if exclude_box is not None and c.get("pelvis") is not None:
                bx0, bx1, by0, by1 = exclude_box
                px_, py_ = c["pelvis"]
                if bx0 <= px_ <= bx1 and by0 <= py_ <= by1:
                    continue  # human inside furniture footprint
            cands.append((ep_dir.name, c))
    # prefer un-padded, close-but-visible instants with a big human
    cands.sort(key=lambda ec: (ec[1]["padded"], ec[1]["min_sep"],
                               -ec[1]["skin_px"]))
    cands = cands[:top_k]
    if not cands:
        raise SystemExit("no candidates pass the filters — relax them")

    from PIL import Image, ImageDraw
    cols = 4
    rows = int(np.ceil(len(cands) / cols))
    thumb_w, thumb_h = 480, 360
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (ep, c) in enumerate(cands):
        img = Image.open(task_dir / ep / "frames" / f"cap_{c['k']:04d}.jpg")
        img = img.resize((thumb_w, thumb_h))
        x = (j % cols) * thumb_w
        y = (j // cols) * (thumb_h + 22)
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + thumb_h + 4),
                  f"{ep}:{c['k']}  step={c['step']} sep={c['min_sep']:.2f} "
                  f"skin={c['skin_px']}{' PAD' if c['padded'] else ''}",
                  fill="black")
    out = task_dir / "contact_sheet.png"
    sheet.save(out)
    log.info("wrote %s (%d candidates)", out, len(cands))
    return out


def load_state(task_dir: Path, ep: str, k: int) -> dict:
    z = np.load(task_dir / ep / "states.npz")
    return {key: z[f"s{k}_{key}"] for key in
            ("qpos", "qvel", "mocap_pos", "mocap_quat", "time")}


def rerender(env, task_key: str, task_dir: Path, ep: str, k: int,
             cam_overrides: dict, width: int, height: int,
             out: Optional[Path] = None) -> List[Path]:
    """Re-render a captured instant; azimuth may be a list for a sweep."""
    import imageio.v2 as imageio

    st = load_state(task_dir, ep, k)
    _restore_state(env, st)
    renderer = _make_renderer(env, width, height)
    azimuths = cam_overrides.pop("azimuth", None)
    if not isinstance(azimuths, (list, tuple)):
        azimuths = [azimuths]
    outs = []
    for az in azimuths:
        cam = _camera(task_key, dict(cam_overrides, azimuth=az))
        renderer.update_scene(env._mojo.data, camera=cam)
        px = renderer.render()
        path = out if (out and len(azimuths) == 1) else (
            (out or task_dir / f"rerender_{ep}_k{k}.png").with_name(
                (out or task_dir / f"rerender_{ep}_k{k}.png").stem
                + (f"_az{int(round(cam.azimuth)):03d}" if len(azimuths) > 1 else "")
                + ".png"))
        imageio.imwrite(path, px)
        log.info("wrote %s (az=%.1f el=%.1f dist=%.2f lookat=%s)", path,
                 cam.azimuth, cam.elevation, cam.distance, list(cam.lookat))
        outs.append(path)
    renderer.close()
    return outs


def probe_camera(env, task_key: str, out_dir: Path, width: int, height: int,
                 scen_seed: int = 0, sim_seconds: float = 6.0) -> None:
    """Render the post-walk-in scene from several azimuths for hand-tuning."""
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    env.reset(seed=0)
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(int(sim_seconds * CONTROL_HZ)):
        env.step(zeros)
    # Print scene anchors to help choose lookat.
    model = env._mojo.model
    for name in range(model.nbody):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bn and any(s in bn.lower() for s in
                      ("dishwasher", "drawer", "cabinet", "counter", "pelvis")):
            log.info("body %-40s xpos=%s", bn,
                     np.round(env._mojo.data.xpos[name], 3))
    renderer = _make_renderer(env, width, height)
    for az in range(0, 360, 30):
        cam = _camera(task_key, dict(azimuth=float(az)))
        renderer.update_scene(env._mojo.data, camera=cam)
        imageio.imwrite(out_dir / f"probe_az{az:03d}.png", renderer.render())
    renderer.close()
    log.info("wrote probes to %s", out_dir)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["dishwasher_close", "drawers_open_all"])
    p.add_argument("--demos", default="0,1,2",
                   help="Comma-separated demo indices to replay.")
    p.add_argument("--scenario-seeds", default="0,1",
                   help="Comma-separated coworker scenario seeds per demo.")
    p.add_argument("--capture-every", type=int, default=100,
                   help="Capture state+frame every Nth 500 Hz step (100 = 5 fps).")
    p.add_argument("--pad-seconds", type=float, default=5.0,
                   help="Hold the final demo action this long after replay.")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out-root", type=Path,
                   default=REPO_ROOT / "results" / "figs" / "task_suite_remake")
    p.add_argument("--list-demos", action="store_true")
    p.add_argument("--probe-camera", action="store_true")
    p.add_argument("--contact-sheet", action="store_true")
    p.add_argument("--exclude-box", default=None,
                   help="xmin,xmax,ymin,ymax pelvis exclusion for --contact-sheet "
                        "(drop frames where the human stands inside furniture).")
    p.add_argument("--rerender", default=None,
                   help="ep_dir:k to re-render from saved state (e.g. ep_d0_s1:12).")
    p.add_argument("--rerender-out", type=Path, default=None)
    p.add_argument("--cam-azimuth", default=None,
                   help="Azimuth, or comma list for a sweep in --rerender.")
    p.add_argument("--cam-elevation", type=float, default=None)
    p.add_argument("--cam-distance", type=float, default=None)
    p.add_argument("--cam-lookat", default=None,
                   help="x,y,z lookat override.")
    args = p.parse_args()

    task_dir = args.out_root / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    if args.list_demos:
        demo_dir = DEMO_CACHE / TASK_MAP[args.task][1] / ACTION_MODE_DIR / "lightweight"
        files = sorted(demo_dir.glob("*.safetensors"))
        from demonstrations.demo import Demo
        for i, f in enumerate(files[:20]):
            d = Demo.from_safetensors(f)
            log.info("demo %2d  %s  seed=%-12s steps=%d (%.1fs)", i, f.name,
                     d.seed, len(d.timesteps), len(d.timesteps) / 500.0)
        log.info("(%d total)", len(files))
        return 0

    if args.contact_sheet:
        box = (tuple(float(x) for x in args.exclude_box.split(","))
               if args.exclude_box else None)
        contact_sheet(task_dir, args.width, exclude_box=box)
        return 0

    azimuth = None
    if args.cam_azimuth is not None:
        vals = [float(x) for x in str(args.cam_azimuth).split(",")]
        azimuth = vals if len(vals) > 1 else vals[0]
    cam_overrides = dict(
        azimuth=azimuth, elevation=args.cam_elevation,
        distance=args.cam_distance,
        lookat=tuple(float(x) for x in args.cam_lookat.split(","))
        if args.cam_lookat else None,
    )

    holder = _ScenarioSeedHolder()
    env = build_env(args.task, holder)
    try:
        if args.probe_camera:
            probe_camera(env, args.task, task_dir / "camera_probe",
                         max(args.width, 640), max(args.height, 480))
            return 0

        if args.rerender:
            ep, k = args.rerender.rsplit(":", 1)
            rerender(env, args.task, task_dir, ep, int(k), cam_overrides,
                     960, 720, args.rerender_out)
            return 0

        demo_indices = [int(x) for x in args.demos.split(",")]
        scen_seeds = [int(x) for x in args.scenario_seeds.split(",")]
        demos = load_demos(args.task, demo_indices)
        for demo in demos:
            for ss in scen_seeds:
                ep_dir = task_dir / f"ep_d{demo['index']}_s{ss}"
                if (ep_dir / "meta.json").exists():
                    log.info("skip existing %s", ep_dir.name)
                    continue
                log.info("replaying demo %d (seed=%s, %d steps) scen_seed=%d ...",
                         demo["index"], demo["seed"], len(demo["actions"]), ss)
                meta = replay_episode(env, holder, demo, ss, ep_dir,
                                      args.capture_every, args.pad_seconds,
                                      args.width, args.height, args.task)
                log.info("  -> success_step=%s min_sep=%.2f near=%.0f%% prox=%.0f%%",
                         meta["success_step"], meta["min_sep"],
                         100 * meta["near_frac"], 100 * meta["prox_frac"])
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
