#!/usr/bin/env python3
"""Render matched policy rollouts with a human-robot SEPARATION HUD burned in.

Plain benchmark videos (eval_video.render_frame) are raw camera frames with no
distance readout, so a safer policy isn't visibly safer — proximity is a
closest-joint measure the camera doesn't make legible. This drives the benchmark
runner directly (reset/step), reads ``min_separation`` off each StepRecord, and
overlays the live separation + SAFE/NEAR/CLOSE status + a red border while in
proximity (<0.3 m).

Because the policy is NOT bit-reproducible across processes (CUDA nondeterminism
compounds over a closed-loop rollout), a single episode is noisy. So this runs a
whole seed's worth of episodes sequentially on ONE reused runner per policy
(exactly as the benchmark does), captures frames + separation IN THE SAME PASS as
the recorded metrics, writes a HUD mp4 per episode, then auto-picks the episode
where the Lagrangian is most clearly safer (and still succeeds) and composes a
side-by-side comparison. The picked clip's on-screen numbers are therefore the
true numbers for that exact rollout — no cross-run mismatch.

Run headless with ``MUJOCO_GL=egl PYOPENGL_PLATFORM=egl``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("render_hud_rollout")

PROX_THRESH = 0.30
NEAR_THRESH = 0.50


def _status(sep: float) -> Tuple[str, Tuple[int, int, int]]:
    if not np.isfinite(sep) or sep < PROX_THRESH:
        return "CLOSE", (220, 40, 40)
    if sep < NEAR_THRESH:
        return "NEAR", (230, 170, 30)
    return "SAFE", (40, 180, 70)


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _overlay(frame: np.ndarray, label: str, sep: float, viol_pct: float) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    status, col = _status(sep)
    bar_h = max(34, H // 11)
    d.rectangle([0, 0, W, bar_h], fill=(0, 0, 0, 175))
    f = _font(int(bar_h * 0.46)); fs = _font(int(bar_h * 0.38))
    sep_txt = "contact" if not np.isfinite(sep) else f"{sep:.2f} m"
    d.text((8, bar_h * 0.12), label, fill=(255, 255, 255), font=f)
    d.text((8, bar_h * 0.55), f"sep {sep_txt}", fill=col, font=f)
    d.text((int(W * 0.66), bar_h * 0.32), status, fill=col, font=f)
    d.text((8, H - bar_h * 0.7), f"in proximity (<{PROX_THRESH:.1f} m): {viol_pct:4.0f}% of steps",
           fill=(255, 255, 255), font=fs)
    if not np.isfinite(sep) or sep < PROX_THRESH:
        b = max(4, W // 110)
        d.rectangle([0, 0, W - 1, H - 1], outline=(235, 30, 30), width=b)
    return np.asarray(img)


def run_policy(snapshot, label, tag, n_eps, seed_base, task, disruption, human_model,
               obs_mode, max_steps, fps, out_dir):
    import imageio.v2 as imageio
    from safety_bigym.benchmark.loader import load_policy
    from safety_bigym.benchmark.runners import build_cell_runner
    from safety_bigym.agents.cqn_as.eval_video import render_frame

    meta, payload = load_policy(snapshot)
    runner, renderable = build_cell_runner(
        meta, payload, snapshot_path=snapshot, task=task, disruption=disruption,
        obs_mode=obs_mode, human_model=human_model, filter_critic=None,
        filter_threshold=None, fallback_name=None, num_demos_for_stats=0)
    stats = []
    try:
        for ep in range(n_eps):
            runner.reset(seed=seed_base * 100_000 + ep)
            frames, viol, steps = [], 0, 0
            cum_reward, success = 0.0, False  # success: same rule as benchmark run_episode
            while steps < max_steps:
                rec = runner.step(); steps += 1
                cum_reward += rec.reward
                ts = (rec.info or {}).get("task_success")
                if (ts is not None and float(ts) > 0.0) or (ts is None and cum_reward > 0.0):
                    success = True
                sep = rec.min_separation
                if (not np.isfinite(sep)) or sep < PROX_THRESH:
                    viol += 1
                fr = render_frame(renderable, global_step=0)
                if fr is not None:
                    frames.append(_overlay(fr, label, sep, 100.0 * viol / steps))
                if rec.done:
                    break
            mp4 = out_dir / f"hud_{tag}_ep{ep}.mp4"
            imageio.mimsave(str(mp4), frames, fps=fps, macro_block_size=1)
            prox = viol / max(steps, 1)
            stats.append({"ep": ep, "prox": prox, "success": success, "steps": steps, "mp4": mp4})
            logger.info("%s ep%d: prox=%.3f success=%s steps=%d", tag, ep, prox, success, steps)
    finally:
        runner.close()
    return stats


def _sbs(a_mp4, b_mp4, out, fps):
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
            return f if f.shape[0] == h else np.vstack([f, np.zeros((h - f.shape[0], f.shape[1], 3), np.uint8)])
        frames.append(np.hstack([pad(fa), np.zeros((h, gap, 3), np.uint8), pad(fb)]))
    imageio.mimsave(str(out), frames, fps=fps, macro_block_size=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--lagrangian", required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--task", default="saucepan_to_hob")
    p.add_argument("--disruption", default="coworker_train")
    p.add_argument("--human-model", default="g1")
    p.add_argument("--obs-mode", default="oracle")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s")
    bdir = a.out_dir / "baseline"; ldir = a.out_dir / "lagrangian"
    bdir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)

    bstats = run_policy(a.baseline, "BASELINE (unconstrained)", "baseline", a.episodes,
                        a.seed_base, a.task, a.disruption, a.human_model, a.obs_mode,
                        a.max_steps, a.fps, bdir)
    lstats = run_policy(a.lagrangian, "LAGRANGIAN λ=0.1", "lagrangian", a.episodes,
                        a.seed_base, a.task, a.disruption, a.human_model, a.obs_mode,
                        a.max_steps, a.fps, ldir)

    # auto-pick: largest (baseline_prox - lagrangian_prox) where Lagrangian succeeds
    bp = {s["ep"]: s for s in bstats}; lp = {s["ep"]: s for s in lstats}
    cands = []
    for ep in sorted(set(bp) & set(lp)):
        gap = bp[ep]["prox"] - lp[ep]["prox"]
        cands.append((ep, gap, lp[ep]["success"], bp[ep]["success"]))
    print("\n=== per-episode (this run) ===")
    print(f"{'ep':>3} {'base_prox':>9} {'lag_prox':>8} {'gap':>6} {'lag_ok':>6} {'base_ok':>7}")
    for ep in sorted(bp):
        print(f"{ep:>3} {bp[ep]['prox']:>9.3f} {lp[ep]['prox']:>8.3f} "
              f"{bp[ep]['prox']-lp[ep]['prox']:>6.3f} {str(lp[ep]['success']):>6} {str(bp[ep]['success']):>7}")
    winners = [c for c in cands if c[2]] or cands
    target = max(winners, key=lambda c: c[1])[0]
    out = a.out_dir / f"hud_sidebyside_ep{target}.mp4"
    _sbs(bp[target]["mp4"], lp[target]["mp4"], out, a.fps)
    print(f"\nPICKED ep{target}: baseline prox={bp[target]['prox']:.3f} -> lagrangian prox={lp[target]['prox']:.3f}")
    print(f"side-by-side -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
