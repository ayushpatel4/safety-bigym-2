#!/usr/bin/env python3
"""Render REAL trained-policy rollouts with the report-style HUD (the *policy*
analogue of report figures E.2 / E.3 / E.4 — and of the scripted presentation
clips). Unlike the scripted clips, here the robot is the actual CQN-AS policy
performing the task; the HUD numbers are read live off each StepRecord.

Runs on the GPU box (where the trained snapshots live). Headless:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
    AMASS_DATA_DIR=/path/to/CMU/CMU \
    python scripts/render_policy_hud.py --mode rq1 \
        --baseline <stage2.pt> --lagrangian <fixlam_basin.pt> \
        --out-dir results/pres_hud/rq1

Three preset modes (one per results slide):
  rq1        : baseline (no filter)  vs  constrained Lagrangian (no filter)  [saucepan]
  veto       : baseline + SVF veto FREEZE  vs  baseline + SVF veto FLEE      [saucepan]
  speedscale : baseline (no filter)  vs  baseline + ISO-SSM speed-scaling    [dishwasher]

Each mode rolls BOTH arms over the SAME seeds on a reused runner (exactly as the
benchmark does), writes a per-episode HUD mp4 per arm, auto-picks the episode
that best shows the effect, and stitches a frame-aligned side-by-side. The picked
clip's on-screen numbers are the true numbers for that exact rollout.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("render_policy_hud")

PROX_THRESH, NEAR_THRESH = 0.30, 0.50
CTRL_HZ = 20.0
# report status palette (== render_disruption_figure._status)
C_SAFE, C_NEAR, C_VIOL, PURPLE = "#28b450", "#e6aa1e", "#dc2828", "#5b21b6"

_REPO = Path(__file__).resolve().parent.parent
_FONT_CANDIDATES = {
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             str(_REPO / "venv/lib/python3.12/site-packages/matplotlib/mpl-data/"
                 "fonts/ttf/DejaVuSans-Bold.ttf")],
    "reg": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            str(_REPO / "venv/lib/python3.12/site-packages/matplotlib/mpl-data/"
                "fonts/ttf/DejaVuSans.ttf")],
}


def _font(kind: str, size: int):
    for p in _FONT_CANDIDATES[kind]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _status(sep):
    if not np.isfinite(sep) or sep < PROX_THRESH:
        return "VIOLATION", C_VIOL
    if sep < NEAR_THRESH:
        return "NEAR", C_NEAR
    return "SAFE", C_SAFE


def hud(frame: np.ndarray, row: dict, kind: str, label: str) -> np.ndarray:
    """Report-style HUD: translucent chips, ISO status palette, bottom-left
    coloured `min sep`, bottom-right status; veto adds a Q->PASS/VETO chip,
    speed-scaling adds a `scale =` chip; red border on violation."""
    img = Image.fromarray(frame).convert("RGB")
    W, H = img.size
    sep, t = row["sep"], row["t"]
    if kind == "speedscale":
        viol = bool(row["ssm_actual"])
        st_txt, st_hex = ("SSM VIOLATION", C_VIOL) if viol else ("SSM OK", C_SAFE)
        sep_hex = _status(sep)[1]
    else:
        viol = (not np.isfinite(sep)) or sep < PROX_THRESH
        st_txt, st_hex = _status(sep)
        sep_hex = st_hex

    s = H / 720.0                                   # scale fonts to frame height
    boxes = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(boxes)
    texts = []
    BLACK = (0, 0, 0, 170)
    pad = max(5, int(8 * s))

    def chip(x, y, text, fg, box_rgba, anchor, font):
        l, tt, r, b = od.textbbox((x, y), text, font=font, anchor=anchor)
        od.rounded_rectangle([l - pad, tt - pad, r + pad, b + pad],
                             radius=max(4, int(6 * s)), fill=box_rgba)
        texts.append(((x, y), text, font, fg, anchor))

    m = max(12, int(20 * s))
    if label:
        chip(m, m, label, (255, 255, 255), BLACK, "la", _font("bold", int(26 * s)))
    chip(W - m, m, f"t = {t:.1f} s", (255, 255, 255), BLACK, "ra", _font("reg", int(21 * s)))
    sep_txt = "contact" if not np.isfinite(sep) else f"{sep:.2f} m"
    chip(m, H - m, f"min sep  {sep_txt}", _hex(sep_hex), BLACK, "ld", _font("bold", int(25 * s)))
    chip(W - m, H - m, st_txt, (255, 255, 255), _hex(st_hex) + (235,), "rd", _font("bold", int(25 * s)))

    if kind == "veto" and row.get("q") is not None:
        interv = bool(row["intervened"])
        sign, verd, col = (("<", "VETO", C_VIOL) if interv else ("\u2265", "PASS", C_SAFE))
        chip(W // 2, m, f"Q(s,a) = {row['q']:.2f}   {sign}   R = {row['R']:.2f}   \u2192   {verd}",
             _hex(col), BLACK, "ma", _font("bold", int(21 * s)))
    elif kind == "speedscale":
        chip(m, H - m - int(46 * s), f"scale = {row['scale']:.2f}", (255, 255, 255),
             _hex(PURPLE) + (235,), "ld", _font("reg", int(22 * s)))
        chip(W // 2, H - m, f"robot vel {row['robot_vel']:.2f} m/s", (255, 255, 255),
             BLACK, "md", _font("reg", int(19 * s)))

    img = Image.alpha_composite(img.convert("RGBA"), boxes).convert("RGB")
    d = ImageDraw.Draw(img)
    for xy, text, font, fill, anchor in texts:
        d.text(xy, text, font=font, fill=fill, anchor=anchor)
    d.rectangle([2, 2, W - 3, H - 3], outline=C_VIOL if viol else (120, 120, 120),
                width=max(2, int(7 * s)) if viol else 2)
    return np.asarray(img)


def build_runner(snapshot, task, disruption, obs_mode, human_model, *, filter_kind,
                 svf_critic=None, R=2.25, fallback="zero_velocity",
                 d_slow=0.5, d_stop=0.15, num_demos_for_stats=0):
    from safety_bigym.benchmark.loader import load_policy
    from safety_bigym.benchmark.runners import build_cell_runner

    meta, payload = load_policy(snapshot)
    filter_critic = speedscale_config = None
    if filter_kind == "svf":
        from safety_bigym.benchmark.filter_attach import load_critic
        filter_critic = load_critic(svf_critic)
    elif filter_kind == "speedscale":
        speedscale_config = dict(d_slow=d_slow, d_stop=d_stop)
    runner, renderable = build_cell_runner(
        meta, payload, snapshot_path=snapshot, task=task, disruption=disruption,
        obs_mode=obs_mode, human_model=human_model,
        filter_critic=filter_critic, filter_threshold=R, fallback_name=fallback,
        speedscale_config=speedscale_config, num_demos_for_stats=num_demos_for_stats)
    return runner, renderable


def run_arm(arm, seeds, n_eps, task, disruption, obs_mode, human_model,
            max_steps, fps, out_dir, d_slow=0.5, d_stop=0.15):
    """arm = dict(snapshot, filter_kind, label, kind, color, svf_critic, R, fallback)."""
    import imageio.v2 as imageio
    from safety_bigym.agents.cqn_as.eval_video import render_frame

    runner, renderable = build_runner(
        arm["snapshot"], task, disruption, obs_mode, human_model,
        filter_kind=arm["filter_kind"], svf_critic=arm.get("svf_critic"),
        R=arm.get("R", 2.25), fallback=arm.get("fallback", "zero_velocity"),
        d_slow=d_slow, d_stop=d_stop)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    try:
        for ep in range(n_eps):
            runner.reset(seed=seeds * 100_000 + ep)
            frames, viol, ssm, interv, steps, cum, success = [], 0, 0, 0, 0, 0.0, False
            while steps < max_steps:
                rec = runner.step(); steps += 1
                cum += rec.reward
                ts = (rec.info or {}).get("task_success")
                if (ts is not None and float(ts) > 0.0) or (ts is None and cum > 0.0):
                    success = True
                safety = (rec.info or {}).get("safety") or {}
                filt = (rec.info or {}).get("safety_filter") or {}
                sep = rec.min_separation
                if (not np.isfinite(sep)) or sep < PROX_THRESH:
                    viol += 1
                if bool(safety.get("ssm_violation_actual", False)):
                    ssm += 1
                if bool(filt.get("intervened", False)):
                    interv += 1
                scale = filt.get("scale")
                if scale is None and arm["kind"] == "speedscale":
                    scale = float(np.clip((sep - d_stop) / (d_slow - d_stop), 0, 1)) \
                        if np.isfinite(sep) else 1.0
                row = dict(t=steps / CTRL_HZ, sep=sep,
                           ssm_actual=bool(safety.get("ssm_violation_actual", False)),
                           robot_vel=float(safety.get("robot_vel", 0.0)),
                           q=filt.get("q_value"), R=float(arm.get("R", 2.25)),
                           intervened=bool(filt.get("intervened", False)),
                           scale=float(scale) if scale is not None else 1.0)
                fr = render_frame(renderable, global_step=0)
                if fr is not None:
                    frames.append(hud(fr, row, arm["kind"], arm["label"]))
                if rec.done:
                    break
            mp4 = out_dir / f"{arm['tag']}_ep{ep}.mp4"
            imageio.mimsave(str(mp4), frames, fps=fps, macro_block_size=1)
            stats.append(dict(ep=ep, prox=viol / max(steps, 1), ssm=ssm / max(steps, 1),
                              interv=interv / max(steps, 1), success=success,
                              steps=steps, mp4=mp4))
            logger.info("%s ep%d: prox=%.3f ssm=%.3f interv=%.3f success=%s steps=%d",
                        arm["tag"], ep, stats[-1]["prox"], stats[-1]["ssm"],
                        stats[-1]["interv"], success, steps)
    finally:
        runner.close()
    return stats


def sbs(a_mp4, b_mp4, out, fps):
    import imageio.v2 as imageio
    a = list(imageio.mimread(str(a_mp4), memtest=False))
    b = list(imageio.mimread(str(b_mp4), memtest=False))
    n = max(len(a), len(b)); gap = 8
    frames = []
    for i in range(n):
        fa = a[min(i, len(a) - 1)][..., :3]
        fb = b[min(i, len(b) - 1)][..., :3]
        h = max(fa.shape[0], fb.shape[0])
        def pad(f):
            return f if f.shape[0] == h else np.vstack(
                [f, np.zeros((h - f.shape[0], f.shape[1], 3), np.uint8)])
        frames.append(np.hstack([pad(fa), np.zeros((h, gap, 3), np.uint8), pad(fb)]))
    imageio.mimsave(str(out), frames, fps=fps, macro_block_size=1)


# preset arms per mode -------------------------------------------------------
def _arms(args):
    if args.mode == "rq1":
        return (dict(tag="baseline", snapshot=args.baseline, filter_kind="none",
                     label="BASELINE (unconstrained)", kind="baseline"),
                dict(tag="lagrangian", snapshot=args.lagrangian, filter_kind="none",
                     label="CONSTRAINED policy", kind="lagrangian"))
    if args.mode == "veto":
        base = dict(snapshot=args.policy, filter_kind="svf", svf_critic=args.svf_critic,
                    R=args.R, kind="veto")
        return (dict(**base, tag="freeze", fallback="zero_velocity",
                     label="SVF veto -> FREEZE"),
                dict(**base, tag="flee", fallback="retreat",
                     label="SVF veto -> FLEE"))
    # speedscale
    return (dict(tag="nofilter", snapshot=args.policy, filter_kind="none",
                 label="NO FILTER", kind="speedscale"),
            dict(tag="speedscale", snapshot=args.policy, filter_kind="speedscale",
                 label="ISO-SSM SPEED-SCALING", kind="speedscale"))


def _pick(mode, a, b):
    """Return the episode index that best shows the effect (right arm = b)."""
    ai = {s["ep"]: s for s in a}; bi = {s["ep"]: s for s in b}
    common = sorted(set(ai) & set(bi))
    if mode == "rq1":      # baseline prox high, lagrangian prox low, lagrangian ok
        cand = [(e, ai[e]["prox"] - bi[e]["prox"]) for e in common if bi[e]["success"]] \
            or [(e, ai[e]["prox"] - bi[e]["prox"]) for e in common]
    elif mode == "veto":   # episode where the veto fires the most (clearest)
        cand = [(e, ai[e]["interv"] + bi[e]["interv"]) for e in common]
    else:                  # speedscale: no-filter SSM high, scaled SSM low
        cand = [(e, ai[e]["ssm"] - bi[e]["ssm"]) for e in common]
    return max(cand, key=lambda c: c[1])[0] if cand else common[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["rq1", "veto", "speedscale"])
    p.add_argument("--baseline"); p.add_argument("--lagrangian")  # rq1
    p.add_argument("--policy")                                    # veto / speedscale
    p.add_argument("--svf-critic", default="checkpoints/svf_coworker_train_g1_0p3_v3.pt")
    p.add_argument("--R", type=float, default=2.25)
    p.add_argument("--d-slow", type=float, default=0.5)
    p.add_argument("--d-stop", type=float, default=0.15)
    p.add_argument("--task", default=None)        # default depends on mode
    p.add_argument("--disruption", default="coworker_train")
    p.add_argument("--human-model", default="g1")
    p.add_argument("--obs-mode", default="noisy")  # filter rows need noisy
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s")

    task = a.task or ("dishwasher_close" if a.mode == "speedscale" else "saucepan_to_hob")
    left, right = _arms(a)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(seeds=a.seed_base, n_eps=a.episodes, task=task,
                  disruption=a.disruption, obs_mode=a.obs_mode, human_model=a.human_model,
                  max_steps=a.max_steps, fps=a.fps, d_slow=a.d_slow, d_stop=a.d_stop)
    lstats = run_arm(left, out_dir=a.out_dir / left["tag"], **common)
    rstats = run_arm(right, out_dir=a.out_dir / right["tag"], **common)

    ai = {s["ep"]: s for s in lstats}; bi = {s["ep"]: s for s in rstats}
    print("\n=== per-episode ===")
    print(f"{'ep':>3} {'L_prox':>7} {'R_prox':>7} {'L_ssm':>6} {'R_ssm':>6} "
          f"{'L_ok':>5} {'R_ok':>5}")
    for e in sorted(set(ai) & set(bi)):
        print(f"{e:>3} {ai[e]['prox']:>7.3f} {bi[e]['prox']:>7.3f} {ai[e]['ssm']:>6.3f} "
              f"{bi[e]['ssm']:>6.3f} {str(ai[e]['success']):>5} {str(bi[e]['success']):>5}")
    ep = _pick(a.mode, lstats, rstats)
    out = a.out_dir / f"{a.mode}_sidebyside_ep{ep}.mp4"
    sbs(ai[ep]["mp4"], bi[ep]["mp4"], out, a.fps)
    print(f"\nPICKED ep{ep}  ({left['label']}  |  {right['label']})")
    print(f"side-by-side -> {out}")
    print(f"individual HUD clips -> {a.out_dir}/{left['tag']}/  and  {a.out_dir}/{right['tag']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
