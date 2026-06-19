#!/usr/bin/env python
"""Render scripted-reconstruction DEMO VIDEOS (with HUD) for the presentation.

These are the *video* analogues of report figures E.2 / E.3 / E.4: real
environment, real G1 coworker, real ISO-15066 monitor, SCRIPTED robot (the
trained-policy snapshots live on the GPU box). Every HUD number is measured
from the simulation. This reuses `run_staged` from
scripts/render_mechanism_figures.py so the behaviours match the report.

Needs a display / GL context (macOS window server) -> run OUTSIDE any sandbox:
    ./venv/bin/python presentation/render_demo_videos.py --figure all

Outputs -> presentation/assets/clips/:
  clip_avoid_baseline.mp4, clip_avoid_constrained.mp4, clip_avoid_compare.mp4
  clip_veto_freeze.mp4, clip_veto_flee.mp4, clip_veto_compare.mp4
  clip_speedscale_on.mp4, clip_speedscale_compare.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ISO thresholds / control rate, hardcoded so the HUD + --recompose path need no
# MuJoCo import (== rdf.PROX_THRESH / NEAR_THRESH / CONTROL_HZ).
PROX_THRESH, NEAR_THRESH, HZ = 0.30, 0.50, 20

# The render path (run_staged, build_env, offscreen renderer) pulls in MuJoCo,
# which can't initialise in a headless/sandboxed shell. Import it lazily so
# --recompose (frames already on disk) works anywhere.
rmf = rdf = None


def _load_render():
    global rmf, rdf
    if rmf is None:
        spec = importlib.util.spec_from_file_location(
            "rmf", REPO / "scripts" / "render_mechanism_figures.py")
        rmf = importlib.util.module_from_spec(spec)
        sys.modules["rmf"] = rmf
        spec.loader.exec_module(rmf)
        rdf = rmf.rdf
    return rmf, rdf


OUT = REPO / "presentation" / "assets" / "clips"
FRAMES_ROOT = REPO / "results" / "demo_video_frames"
_FTTF = REPO / "venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf"
FB_PATH, FR_PATH = str(_FTTF / "DejaVuSans-Bold.ttf"), str(_FTTF / "DejaVuSans.ttf")

W, H = 960, 720
# ISO status palette taken verbatim from render_disruption_figure._status so the
# clips match the report figures (E.2-E.4).
C_SAFE, C_NEAR, C_VIOL = "#28b450", "#e6aa1e", "#dc2828"
PURPLE, INK, BLUE = "#5b21b6", "#0f172a", "#1e3a8a"
RED, GREEN = C_VIOL, C_SAFE


def _fb(sz):
    return ImageFont.truetype(FB_PATH, sz)


def _fr(sz):
    return ImageFont.truetype(FR_PATH, sz)


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _status(sep):
    if not np.isfinite(sep) or sep < PROX_THRESH:
        return "VIOLATION", C_VIOL
    if sep < NEAR_THRESH:
        return "NEAR", C_NEAR
    return "SAFE", C_SAFE


def hud(frame_png, row, kind, label, lab_bg=None):
    """Report-style HUD (matches figures E.2-E.4): translucent black chips,
    ISO status palette, bottom-left coloured `min sep`, bottom-right status
    chip, a `Q(s,a) >=/< R -> PASS/VETO` chip for the veto, a `scale =` chip
    for speed-scaling, and a red border on violation. `frame_png` may be a
    path or a PIL image."""
    img = (frame_png if isinstance(frame_png, Image.Image)
           else Image.open(frame_png)).convert("RGB")
    sep, t = row["sep"], row["t"]
    if kind == "speedscale":                       # this filter owns the SSM axis
        viol = bool(row["ssm_actual"])
        st_txt, st_hex = ("SSM VIOLATION", C_VIOL) if viol else ("SSM OK", C_SAFE)
        sep_hex = _status(sep)[1]
    else:
        viol = (not np.isfinite(sep)) or sep < PROX_THRESH
        st_txt, st_hex = _status(sep)
        sep_hex = st_hex

    boxes = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(boxes)
    texts = []
    BLACK = (0, 0, 0, 170)                          # the report's ~0.66-alpha backing

    def chip(x, y, text, fg, box_rgba, anchor, font, pad=8):
        l, tt, r, b = od.textbbox((x, y), text, font=font, anchor=anchor)
        od.rounded_rectangle([l - pad, tt - pad, r + pad, b + pad], radius=6,
                             fill=box_rgba)
        texts.append(((x, y), text, font, fg, anchor))

    if label:
        chip(20, 18, label, (255, 255, 255), BLACK, "la", _fb(26))
    chip(W - 20, 18, f"t = {t:.1f} s", (255, 255, 255), BLACK, "ra", _fr(21))
    sep_txt = "contact" if not np.isfinite(sep) else f"{sep:.2f} m"
    chip(20, H - 20, f"min sep  {sep_txt}", _hex(sep_hex), BLACK, "ld", _fb(25))
    chip(W - 20, H - 20, st_txt, (255, 255, 255), _hex(st_hex) + (235,), "rd", _fb(25))

    if kind == "veto" and row.get("q") is not None:
        interv = bool(row["intervened"])
        sign, verdict, col = (("<", "VETO", C_VIOL) if interv
                              else ("\u2265", "PASS", C_SAFE))
        chip(W // 2, 18, f"Q(s,a) = {row['q']:.2f}   {sign}   R = 2.25   \u2192   {verdict}",
             _hex(col), BLACK, "ma", _fb(21))
        chip(W // 2, H - 20, f"robot vel {row['robot_vel']:.2f} m/s",
             (255, 255, 255), BLACK, "md", _fr(19))
    elif kind == "speedscale":
        chip(20, H - 66, f"scale = {row['scale']:.2f}", (255, 255, 255),
             _hex(PURPLE) + (235,), "ld", _fr(22))
        chip(W // 2, H - 20, f"robot vel {row['robot_vel']:.2f} m/s",
             (255, 255, 255), BLACK, "md", _fr(19))

    img = Image.alpha_composite(img.convert("RGBA"), boxes).convert("RGB")
    d = ImageDraw.Draw(img)
    for xy, text, font, fill, anchor in texts:
        d.text(xy, text, font=font, fill=fill, anchor=anchor)
    d.rectangle([3, 3, W - 4, H - 4],
                outline=C_VIOL if viol else (120, 120, 120),
                width=7 if viol else 2)
    return np.asarray(img)


def write_mp4(frames_dir, series, kind, label, lab_bg, dst, fps):
    steps = series["frame_step"]
    with imageio.get_writer(str(dst), fps=fps, macro_block_size=1,
                            codec="libx264", quality=8) as w:
        for k, step in enumerate(steps):
            fp = frames_dir / f"frame_{k:05d}.png"
            if not fp.exists():
                break
            row = {key: series[key][step] for key in
                   ("t", "sep", "prox", "ssm_actual", "robot_vel", "scale",
                    "intervened", "q")}
            w.append_data(hud(fp, row, kind, label, lab_bg))
    print("  wrote", dst.name)


def compare_mp4(left, right, lkind, rkind, ll, rl, lbg, rbg,
                ls, rs, dst, fps, ldir, rdir):
    """Side-by-side, frame-aligned (same seed/scenario)."""
    n = min(len(ls["frame_step"]), len(rs["frame_step"]))
    bw = 56
    with imageio.get_writer(str(dst), fps=fps, macro_block_size=1,
                            codec="libx264", quality=8) as w:
        for k in range(n):
            lf = ldir / f"frame_{k:05d}.png"
            rf = rdir / f"frame_{k:05d}.png"
            if not (lf.exists() and rf.exists()):
                break
            lr = {key: ls[key][ls["frame_step"][k]] for key in
                  ("t", "sep", "prox", "ssm_actual", "robot_vel", "scale", "intervened", "q")}
            rr = {key: rs[key][rs["frame_step"][k]] for key in
                  ("t", "sep", "prox", "ssm_actual", "robot_vel", "scale", "intervened", "q")}
            li = hud(lf, lr, lkind, "", lbg)
            ri = hud(rf, rr, rkind, "", rbg)
            canvas = Image.new("RGB", (2 * W, H + bw), INK)
            canvas.paste(Image.fromarray(li), (0, bw))
            canvas.paste(Image.fromarray(ri), (W, bw))
            dr = ImageDraw.Draw(canvas)
            f = _fb(28)
            dr.rectangle([0, 0, W, bw], fill=lbg)
            dr.rectangle([W, 0, 2 * W, bw], fill=rbg)
            dr.text((W // 2, bw // 2), ll, font=f, fill="white", anchor="mm")
            dr.text((W + W // 2, bw // 2), rl, font=f, fill="white", anchor="mm")
            dr.line([(W, 0), (W, H + bw)], fill="white", width=3)
            w.append_data(np.asarray(canvas))
    print("  wrote", dst.name, f"({n} frames)")


_TRAIN_BAND = dict(
    coworker_closest_approach_range=(0.55, 0.85),
    coworker_reach_period_range=(0.9, 1.6),
    coworker_target_mix_p_ee_range=(0.55, 0.85),
    coworker_near_loiter_range=(12.0, 18.0),
    coworker_walk_speed_range=(1.0, 1.5),
    coworker_trajectory_weights={"COWORKER_PATROL": 8.0,
                                 "APPROACH_LOITER_DEPART": 1.0, "STATIONARY": 1.0},
)


def render_disruption(seed, every, secs, cam_az, spawn="in_place"):
    """Passive-robot coworker-disruption clip. The robot holds its home pose so
    every violation is the coworker's doing. -> clip_benchmark_disruption.mp4.
    spawn="in_place" parks & reaches continuously; spawn="patrol" walks in,
    reaches, departs (away excursion), and returns (the E.1 rhythm)."""
    _load_render()
    from bigym.action_modes import JointPositionActionMode
    from bigym.envs.pick_and_place import SaucepanToHob
    from safety_bigym import HumanConfig, SafetyConfig, make_safety_env
    from safety_bigym.scenarios import DisruptionType, ParameterSpace, ScenarioSampler

    sampler = ScenarioSampler(parameter_space=ParameterSpace(
        clip_paths=[], disruption_weights={DisruptionType.COWORKER: 1.0},
        **_TRAIN_BAND), motion_dir=None)
    _s = sampler.sample_scenario

    def _ov(sd):
        sc = _s(sd)
        if spawn == "patrol":                       # walk in -> reach -> away -> return
            sc.trajectory_type = "COWORKER_PATROL"
            sc.patrol_excursions = 1
            sc.patrol_near_loiter = 10.0
            sc.patrol_away_distance = 2.3
        else:                                       # in_place: park & reach continuously
            sc.trajectory_type = "STATIONARY"
        sc.disruption_config.coworker_target_mix = (1.0, 0.0)   # reach EE
        return sc
    sampler.sample_scenario = _ov
    env = make_safety_env(
        task_cls=SaucepanToHob,
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        safety_config=SafetyConfig(log_violations=False, terminate_on_violation=False),
        human_config=HumanConfig(human_model="g1"),
        scenario_sampler=sampler, inject_human=True)
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    env.reset(seed=seed)
    renderer = rdf._make_renderer(raw, W, H)
    cam = rdf._auto_camera(raw, rdf._scenario_meta(raw))
    if cam_az is not None:
        cam.azimuth = float(cam_az)
    dst = OUT / "clip_benchmark_disruption.mp4"
    n = int(secs * HZ)
    sep = float("inf")
    seps = []
    print(f"staging disruption ({spawn} + reach ee, passive robot, {secs}s)...")
    with imageio.get_writer(str(dst), fps=max(1, round(HZ / every)),
                            macro_block_size=1, codec="libx264", quality=8) as w:
        for i in range(n):
            _, _, term, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))
            sep = float(info.get("safety", {}).get("min_separation", np.nan))
            seps.append(sep)
            if i % every == 0:
                renderer.update_scene(raw._mojo.data, camera=cam)
                img = Image.fromarray(renderer.render()).convert("RGB")
                row = dict(t=i / HZ, sep=sep, ssm_actual=False,
                           robot_vel=0.0, scale=1.0, intervened=False)
                w.append_data(hud(img, row, "disruption",
                                  "safety_bigym - coworker disruption (passive robot)",
                                  BLUE))
            if spawn != "patrol" and (term or trunc):
                env.reset(seed=seed)
    renderer.close()
    env.close()
    a = np.array(seps, dtype=float)
    print(f"  wrote {dst.name}  | min {np.nanmin(a):.2f} max {np.nanmax(a):.2f} "
          f"close<0.3 {100*np.mean(a < 0.3):.0f}%  away>1.5 {100*np.mean(a > 1.5):.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="all",
                    choices=["all", "avoid", "veto", "speedscale", "disruption"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--every", type=int, default=1)   # 1 -> 20 fps (smooth)
    ap.add_argument("--seconds", type=float, default=None)  # override for a quick test
    ap.add_argument("--cam-azimuth", type=float, default=None)
    ap.add_argument("--spawn", default="patrol", choices=["in_place", "patrol"],
                    help="disruption backup style (patrol = walk-in/reach/away/return).")
    ap.add_argument("--recompose", action="store_true",
                    help="re-overlay the HUD onto already-rendered frames "
                         "(no MuJoCo/GL); reads results/demo_video_frames/.")
    args = ap.parse_args()
    fps = max(1, round(HZ / args.every))
    OUT.mkdir(parents=True, exist_ok=True)

    env = cam = None
    if not args.recompose:
        _load_render()
        pin = argparse.Namespace(pin_closest=0.72, pin_p_ee=0.70,
                                 pin_near_loiter=None, pin_away_distance=None,
                                 pin_excursions=1)
        env = rdf.build_env("saucepan", "g1", pin)
        cam = args.cam_azimuth
        if cam is None:
            print("probing camera azimuth...")
            cam = rdf._pick_visible_azimuth(env, args.seed, 30.0, W, H)
        print("camera azimuth", round(cam, 1))

    def stage(tag, mode, fmode, secs, behavior=None):
        d = FRAMES_ROOT / f"s{args.seed}" / tag
        if args.recompose:                          # reuse saved frames + series
            s = json.loads((d / "series.json").read_text())
            print(f"recompose {tag} ({len(s['t'])} steps from disk)...")
            return d, s
        print(f"staging {tag} (mode={mode} filter={fmode}, {secs}s)...")
        s = rmf.run_staged(env, args.seed, mode, fmode, secs, args.every,
                           d, cam, W, H, behavior_kwargs=behavior)
        (d / "series.json").write_text(json.dumps(s))
        print(f"  {len(s['t'])} steps, prox {100*np.mean(s['prox']):.0f}%, "
              f"ssm {100*np.mean(s['ssm_actual']):.0f}%, "
              f"minsep {np.nanmin(s['sep']):.2f}, vel {np.nanmean(s['robot_vel']):.2f}")
        return d, s

    try:
        todo = (["avoid", "veto", "speedscale", "disruption"]
                if args.figure == "all" else [args.figure])

        if "disruption" in todo:
            if args.recompose:
                print("skip disruption (no saved frames; re-render with GL: "
                      "--figure disruption)")
            else:
                secs = args.seconds or (46.0 if args.spawn == "patrol" else 24.0)
                render_disruption(args.seed, args.every, secs, cam, spawn=args.spawn)

        if "avoid" in todo:
            secs = args.seconds or 40.0
            bd, bs = stage("avoid_baseline", "baseline", "none", secs)
            cd, cs = stage("avoid_constrained", "lagrangian", "none", secs)
            write_mp4(bd, bs, "baseline", "Baseline (unconstrained)", RED,
                      OUT / "clip_avoid_baseline.mp4", fps)
            write_mp4(cd, cs, "lagrangian", "Constrained policy  (lambda=0.1)",
                      GREEN, OUT / "clip_avoid_constrained.mp4", fps)
            compare_mp4(bd, cd, "baseline", "lagrangian",
                        "BASELINE - works through", "CONSTRAINED - yields & waits",
                        RED, GREEN, bs, cs, OUT / "clip_avoid_compare.mp4", fps, bd, cd)

        if "veto" in todo:
            secs = args.seconds or 26.0
            fd, fs = stage("veto_freeze", "baseline", "veto_zero", secs)
            ld, ls = stage("veto_flee", "baseline", "veto_retreat", secs)
            write_mp4(fd, fs, "veto", "Learned veto -> FREEZE (zero velocity)",
                      RED, OUT / "clip_veto_freeze.mp4", fps)
            write_mp4(ld, ls, "veto", "Learned veto -> FLEE (retreat)",
                      RED, OUT / "clip_veto_flee.mp4", fps)
            compare_mp4(fd, ld, "veto", "veto",
                        "FREEZE - dwells in danger", "FLEE - abandons task",
                        "#8b1a1a", "#b45309", fs, ls,
                        OUT / "clip_veto_compare.mp4", fps, fd, ld)

        if "speedscale" in todo:
            secs = args.seconds or 40.0
            beh = dict(cycle_period=2.0, shuttle_amp=0.22)
            nd, ns = stage("ss_nofilter", "baseline", "none", secs, beh)
            sd, ss = stage("ss_on", "baseline", "speedscale", secs, beh)
            write_mp4(sd, ss, "speedscale", "ISO-SSM speed-scaling ON", PURPLE,
                      OUT / "clip_speedscale_on.mp4", fps)
            compare_mp4(nd, sd, "speedscale", "speedscale",
                        "NO FILTER", "ISO-SSM SPEED-SCALING",
                        "#8b1a1a", PURPLE, ns, ss,
                        OUT / "clip_speedscale_compare.mp4", fps, nd, sd)
    finally:
        if env is not None:
            env.close()
    print("done ->", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
