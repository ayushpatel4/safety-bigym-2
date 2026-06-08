#!/usr/bin/env python3
"""Generate MOCK / illustrative safety-HUD videos of TARGET behaviour.

These are NOT measured rollouts — no current policy achieves them (see the
safety/task tradeoff finding). They are clearly-labelled concept animations of
what a *good constrained policy* and a *good hybrid (policy + SVF filter)* should
look like: human-robot proximity violations drop to ~zero while the task still
completes. Top-down schematic so "proximity violation" is finally visible — a red
0.3 m danger ring around the human that the robot must stay out of.

Two mechanisms, visibly different:
  * good_policy : robot arcs around the human on its own; no interventions.
  * good_hybrid : base intent heads straight at the human; the SVF filter VETOES
                  at close-approach, nudging it out right at the threshold.

Pure matplotlib/numpy — CPU only, no env/GPU. Writes mp4s via imageio.

    python scripts/mock_safety_hud.py --out-dir ../vids/mock
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

PROX = 0.30   # m: violation ring
NEAR = 0.50   # m: caution ring

PICK = np.array([0.35, 1.55])   # saucepan
HOB = np.array([1.75, 1.55])    # hob
HOME = np.array([1.05, 0.30])   # robot rest


def _smooth(a, b, u):
    u = np.clip(u, 0, 1); s = u * u * (3 - 2 * u)  # smoothstep
    return a + (b - a) * s


def human_pos(t):
    """Coworker walks right->left across the top work line during the traverse."""
    if t < 0.30:
        return np.array([1.95, 1.52])
    if t < 0.78:
        return _smooth(np.array([1.95, 1.52]), np.array([0.30, 1.52]), (t - 0.30) / 0.48)
    return np.array([0.30, 1.52])


def robot_nominal(t):
    """Task path: home -> pick -> (straight traverse along top) -> hob -> home."""
    if t < 0.30:
        return _smooth(HOME, PICK, t / 0.30)
    if t < 0.40:
        return PICK.copy()                                   # grab
    if t < 0.82:
        return _smooth(PICK, HOB, (t - 0.40) / 0.42)          # traverse (conflict)
    if t < 0.90:
        return HOB.copy()                                    # place
    return _smooth(HOB, HOME, (t - 0.90) / 0.10)


def robot_good_policy(t):
    """Anticipatory detour: dip below the human throughout the traverse."""
    p = robot_nominal(t).copy()
    if 0.40 <= t < 0.82:
        w = np.sin(np.pi * (t - 0.40) / 0.42)   # bump, peak mid-traverse
        p[1] -= 0.55 * w                        # arc down, well clear
    return p, False


def robot_good_hybrid(t):
    """Base intent = straight traverse; SVF vetoes only when it would violate."""
    p = robot_nominal(t)
    h = human_pos(t)
    d = np.linalg.norm(p - h)
    vetoed = False
    margin = 0.34                               # filter holds just outside PROX
    if d < margin:
        away = (p - h)
        n = np.linalg.norm(away)
        away = np.array([0.0, -1.0]) if n < 1e-6 else away / n
        p = h + away * margin                   # pushed to the safe boundary
        vetoed = True
    return p, vetoed


def _font(sz, bold=True):
    from matplotlib import font_manager as fm
    for name in (("DejaVu Sans", bold),):
        pass
    return {"fontsize": sz, "fontweight": ("bold" if bold else "normal")}


def render(mode: str, out: Path, n=270, fps=30, dpi=100):
    import imageio.v2 as imageio
    title = "GOOD CONSTRAINED POLICY" if mode == "policy" else "GOOD HYBRID  (policy + SVF filter)"
    pos_fn = robot_good_policy if mode == "policy" else robot_good_hybrid
    frames = []
    trail = []
    viol = 0
    completed_at = None
    for i in range(n):
        t = i / (n - 1)
        rp, vetoed = pos_fn(t)
        hp = human_pos(t)
        sep = float(np.linalg.norm(rp - hp))
        if sep < PROX:
            viol += 1
        trail.append(rp)
        # task complete when robot reaches hob (place done ~t>=0.90)
        if completed_at is None and t >= 0.90:
            completed_at = i
        viol_pct = 100.0 * viol / (i + 1)

        fig = plt.figure(figsize=(6.4, 6.0), dpi=dpi)
        ax = fig.add_axes([0.06, 0.06, 0.90, 0.74])
        ax.set_xlim(-0.1, 2.1); ax.set_ylim(-0.1, 2.0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(plt.Rectangle((-0.1, -0.1), 2.2, 2.1, fc="#0e1117", ec="none", zorder=0))
        # task markers
        ax.add_patch(plt.Rectangle(PICK - 0.07, 0.14, 0.14, fc="#2ecc71", ec="white", lw=1, zorder=2))
        ax.text(PICK[0], PICK[1] + 0.16, "saucepan", color="#2ecc71", ha="center", fontsize=8)
        ax.add_patch(plt.Rectangle(HOB - 0.07, 0.14, 0.14, fc="#e74c3c", ec="white", lw=1, zorder=2))
        ax.text(HOB[0], HOB[1] + 0.16, "hob", color="#e74c3c", ha="center", fontsize=8)
        # human + rings
        col = "#2ecc71" if sep >= NEAR else ("#e6a817" if sep >= PROX else "#e63030")
        ax.add_patch(Circle(hp, NEAR, fill=False, ec="#e6a817", ls=":", lw=1.0, alpha=0.6, zorder=1))
        ax.add_patch(Circle(hp, PROX, fill=False, ec="#e63030", ls="--", lw=1.4, alpha=0.9, zorder=1))
        ax.add_patch(Circle(hp, 0.13, fc="#e69138", ec="white", lw=1.2, zorder=3))
        ax.text(hp[0], hp[1] - 0.27, "human", color="#e69138", ha="center", fontsize=8)
        # robot trail + body
        if len(trail) > 1:
            tr = np.array(trail)
            ax.plot(tr[:, 0], tr[:, 1], color="#3498db", lw=1.4, alpha=0.5, zorder=2)
        ax.add_patch(Circle(rp, 0.11, fc="#3498db", ec="white", lw=1.2, zorder=4))
        ax.text(rp[0], rp[1] - 0.22, "robot", color="#5dade2", ha="center", fontsize=8)
        # link line coloured by separation
        ax.plot([rp[0], hp[0]], [rp[1], hp[1]], color=col, lw=1.2, alpha=0.7, zorder=2)
        ax.text((rp[0] + hp[0]) / 2, (rp[1] + hp[1]) / 2 + 0.04, f"{sep:.2f} m",
                color=col, ha="center", fontsize=8, fontweight="bold")
        if vetoed:
            ax.text(rp[0], rp[1] + 0.22, "SVF VETO", color="#ffd000", ha="center",
                    fontsize=10, fontweight="bold", zorder=5)
            ax.add_patch(Circle(rp, 0.18, fill=False, ec="#ffd000", lw=2.0, zorder=5))

        # HUD bar
        hud = fig.add_axes([0.0, 0.80, 1.0, 0.20]); hud.axis("off")
        hud.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#000000", ec="none"))
        status = "SAFE" if sep >= NEAR else ("NEAR" if sep >= PROX else "CLOSE")
        hud.text(0.02, 0.74, title, color="white", fontsize=12, fontweight="bold")
        hud.text(0.02, 0.44, "MOCK — TARGET BEHAVIOUR (illustrative, not measured)",
                 color="#9aa0a6", fontsize=8, style="italic")
        hud.text(0.42, 0.70, f"separation  {sep:.2f} m", color=col, fontsize=12, fontweight="bold")
        hud.text(0.80, 0.70, status, color=col, fontsize=13, fontweight="bold")
        hud.text(0.42, 0.30, f"proximity (<0.3 m): {viol_pct:4.1f}%",
                 color="white", fontsize=10)
        tstat = "DONE ✓" if completed_at is not None else "running"
        tcol = "#2ecc71" if completed_at is not None else "#cccccc"
        hud.text(0.73, 0.30, f"task: {tstat}", color=tcol, fontsize=10, fontweight="bold")
        hud.set_xlim(0, 1); hud.set_ylim(0, 1)
        # red border on violation
        if sep < PROX:
            for sp in ax.spines.values():
                pass
            ax.add_patch(plt.Rectangle((-0.1, -0.1), 2.2, 2.1, fill=False, ec="#e63030", lw=6, zorder=9))

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
        plt.close(fig)

    imageio.mimsave(str(out), frames, fps=fps, macro_block_size=1)
    return {"frames": len(frames), "viol_pct": 100.0 * viol / n,
            "completed": completed_at is not None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--frames", type=int, default=270)
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for mode, name in (("policy", "mock_good_policy.mp4"), ("hybrid", "mock_good_hybrid.mp4")):
        out = a.out_dir / name
        s = render(mode, out, n=a.frames, fps=a.fps)
        print(f"{name}: {s['frames']} frames, proximity {s['viol_pct']:.1f}%, task_complete={s['completed']} -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
