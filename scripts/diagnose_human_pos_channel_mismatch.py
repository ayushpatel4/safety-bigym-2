#!/usr/bin/env python
"""Compare ``human_pos_estimate`` distributions between demo replay and live G1.

Step 4 sub-diagnostic of the G1 base-curriculum recovery plan
(``/Users/ayushpatel/.claude/plans/read-safety-bigym-docs-implementation-st-kind-aurora.md``).
The CNN-OOD hypothesis was weakened by Step 0b (G1 is barely visible / off-camera
in ``coworker_idle``), so the next likely root cause is the **low-dim** channel
mismatch.

Setup (concretely different from SMPL-H training):
- DEMOS inject ``human_pos_estimate`` from ``AMASSDemoPositionProvider``
  (AMASS gait clips, Z oscillates ~0.7-1.1 with stride, spawn distance 1.5-3.0).
- LIVE env drives the G1 coworker via scripted waypoints
  (``g1_spec.STANDING_PELVIS_Z = 0.793`` fixed, idle distance 3.0-3.6).

If the demo channel statistics diverge meaningfully from the live channel
statistics, the CNN-AS encoder + critic + actor are partly trained against an
OOD demo distribution that never appears at runtime. That's a candidate root
cause for the G1 retreat-from-task degeneration (attempts 1-4) and is fixable
without re-recording demos — see Step 4 of the plan.

Usage (GPU box):
    export AMASS_DATA_DIR=/path/to/CMU/CMU
    export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
    python scripts/diagnose_human_pos_channel_mismatch.py \\
        env=safety_bigym/saucepan_to_hob \\
        disruption=coworker_idle \\
        bodyslam=oracle \\
        num_demos=10 \\
        +live_steps=500 \\
        +outdir=/tmp/g1_channel_diag

    Note: ``num_demos`` is already a key in ``cqn_as_config.yaml`` (default 0),
    so it takes a plain ``num_demos=10`` override (no ``+`` prefix); Hydra
    strict mode rejects ``+num_demos`` because the key already exists.
    ``live_steps`` and ``outdir`` ARE new keys, so they need the ``+`` prefix.

Output (under ``outdir``):
    demo_channel.npy    # (N_demo_steps, 3) AMASS-injected pelvis xyz
    live_channel.npy    # (N_live_steps, 3) live G1 pelvis xyz
    summary.txt         # per-dim mean / std / min / max for both, plus a verdict.
    histograms.png      # 3 hists per dim (demo vs live overlay)

What to look for in summary.txt:
- ``z mean`` — demos likely 0.85-1.0 (AMASS), live likely 0.79 (fixed). Magnitude
  gap of ~0.05-0.2 in absolute units is a clear distribution shift.
- ``z std`` — demos non-zero (gait), live near-zero (mocap fixed). If demos
  std > 0.05 and live std < 0.01, the channel encodes "human is walking" vs
  "human is rigid" — the policy can't generalise.
- ``x/y`` magnitudes — demos at 1.5-3.0, live at 3.0-3.6 (coworker_idle) or
  0.9-1.4 (coworker_train). Big magnitude jumps when stages 0 → 2 transition
  would compound an already-mismatched channel.
- If the verdict line says ``MISMATCH``: Step 4b in the plan is the right fix.
- If it says ``MATCH (within tolerance)``: drop to Step 1 (pixels=false ablation).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import hydra
import numpy as np
from omegaconf import DictConfig

if not os.environ.get("AMASS_DATA_DIR"):
    raise RuntimeError(
        "AMASS_DATA_DIR is not set. Export it (see safety_bigym/docs/CLAUDE.md)."
    )

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def _collect_demo_channel(cfg: DictConfig, num_demos: int) -> np.ndarray:
    """Replay ``num_demos`` BiGym demos through the adapter to harvest the
    AMASS-injected ``human_pos_estimate`` xyz."""
    # Adapter import is here (not at module level) so a partial environment
    # without tensordict / mujoco can still import the script for --help.
    from safety_bigym.agents.cqn_as.env_adapter import make as make_adapter

    adapter = make_adapter(cfg, frame_stack=1)
    # SafetyBiGymCQNAdapter is wrapped by ExtendedTimeStepWrapper; reach in.
    inner = adapter._env if hasattr(adapter, "_env") else adapter
    if not getattr(inner, "_inject_human_pos", False):
        raise RuntimeError(
            "demo bodyslam injection is off — re-run with bodyslam=oracle "
            "(or any non-off mode)."
        )
    demos = inner.get_demos(num_demos)
    xyzs: list[np.ndarray] = []
    for demo in demos:
        for ts in demo:
            ld = ts.low_dim_obs
            # The bodyslam channel is the LAST 6 floats of low_dim_obs (the
            # adapter concatenates state keys then appends human_pos_estimate).
            # Pull just xyz (first 3 of the 6 channel dims).
            ld = np.asarray(ld, dtype=np.float32).reshape(-1)
            if ld.shape[0] < 6:
                continue
            xyz = ld[-6:-3]
            xyzs.append(xyz.copy())
    if not xyzs:
        raise RuntimeError("no demo channel samples were collected.")
    return np.stack(xyzs)


def _collect_live_channel(cfg: DictConfig, steps: int) -> np.ndarray:
    """Step the live G1 env with a random policy and harvest channel xyz.

    Reads ``info["safety"]["human_pos"]`` (the ground truth fed into the
    BodySLAMWrapper in ``oracle`` mode). This is the exact source the
    adapter ultimately exposes via ``human_pos_estimate[:3]``.
    """
    from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory

    factory = SafetyBiGymEnvFactory()
    env = factory._create_env(cfg)
    obs, _info = env.reset()
    xyzs: list[np.ndarray] = []
    # Try to read from obs first (live env attaches human_pos_estimate via
    # SafetyBiGymEnvFactory's BodySLAMWrapper); fall back to info.
    if "human_pos_estimate" in obs:
        xyzs.append(np.asarray(obs["human_pos_estimate"][:3], dtype=np.float32))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    for _ in range(steps):
        if hasattr(env.action_space, "sample"):
            action = env.action_space.sample()
        else:
            action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
        obs, _r, term, trunc, info = env.step(action)
        sample = None
        if isinstance(obs, dict) and "human_pos_estimate" in obs:
            sample = np.asarray(obs["human_pos_estimate"][:3], dtype=np.float32)
        elif info and info.get("safety", {}).get("human_pos") is not None:
            sample = np.asarray(info["safety"]["human_pos"], dtype=np.float32)[:3]
        if sample is not None:
            xyzs.append(sample.copy())
        if term or trunc:
            obs, _info = env.reset()
    if not xyzs:
        raise RuntimeError(
            "no live channel samples — check that BodySLAMWrapper is "
            "attached and bodyslam.mode is not off."
        )
    return np.stack(xyzs)


def _stats(xyz: np.ndarray) -> dict:
    return {
        "n": int(xyz.shape[0]),
        "mean": xyz.mean(axis=0).tolist(),
        "std": xyz.std(axis=0).tolist(),
        "min": xyz.min(axis=0).tolist(),
        "max": xyz.max(axis=0).tolist(),
    }


def _verdict(demo: dict, live: dict) -> str:
    """Heuristic: flag MISMATCH if any dim's mean differs by >0.1 OR the std
    ratio is >5x (one side rigid, the other not)."""
    demo_mean = np.array(demo["mean"])
    live_mean = np.array(live["mean"])
    demo_std = np.array(demo["std"])
    live_std = np.array(live["std"])
    mean_gap = np.abs(demo_mean - live_mean)
    # Avoid division by zero with a small floor.
    std_ratio = np.maximum(demo_std, 1e-4) / np.maximum(live_std, 1e-4)
    # Also the inverse, since we don't know which side is rigid.
    std_ratio_inv = np.maximum(live_std, 1e-4) / np.maximum(demo_std, 1e-4)
    big_gap = (mean_gap > 0.1).any()
    big_std_split = (np.maximum(std_ratio, std_ratio_inv) > 5.0).any()
    if big_gap or big_std_split:
        return (
            f"VERDICT: MISMATCH (mean gaps={mean_gap.round(3).tolist()}, "
            f"std ratios={np.maximum(std_ratio, std_ratio_inv).round(2).tolist()}). "
            "Step 4b in the plan (G1-style demo trajectory) is the likely fix."
        )
    return (
        f"VERDICT: MATCH within tolerance (mean gaps={mean_gap.round(3).tolist()}, "
        f"std ratios={np.maximum(std_ratio, std_ratio_inv).round(2).tolist()}). "
        "Channel mismatch is NOT the dominant cause. Drop to Step 1 (pixels=false)."
    )


def _write_hists(out_path: Path, demo: np.ndarray, live: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib unavailable; skipping %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        ax.hist(demo[:, i], bins=40, alpha=0.5, label="demo (AMASS)", density=True)
        ax.hist(live[:, i], bins=40, alpha=0.5, label="live (G1)", density=True)
        ax.set_title("xyz"[i])
        ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close(fig)
    logger.info("wrote %s", out_path)


@hydra.main(version_base=None, config_path="../cfgs", config_name="cqn_as_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    num_demos = int(cfg.get("num_demos", 10))
    live_steps = int(cfg.get("live_steps", 500))
    outdir = Path(str(cfg.get("outdir", "/tmp/g1_channel_diag"))).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "comparing channels: num_demos=%d live_steps=%d outdir=%s",
        num_demos, live_steps, outdir,
    )

    logger.info("collecting demo channel...")
    demo_xyz = _collect_demo_channel(cfg, num_demos)
    np.save(outdir / "demo_channel.npy", demo_xyz)

    logger.info("collecting live channel...")
    live_xyz = _collect_live_channel(cfg, live_steps)
    np.save(outdir / "live_channel.npy", live_xyz)

    demo_stats = _stats(demo_xyz)
    live_stats = _stats(live_xyz)
    verdict = _verdict(demo_stats, live_stats)

    summary = [
        "# human_pos_estimate channel mismatch diagnostic",
        f"# num_demos={num_demos} live_steps={live_steps}",
        f"# disruption={cfg.env.get('disruption_type', '?')}",
        "",
        "## demo (AMASS-injected during demo replay)",
        f"  n={demo_stats['n']}",
        f"  mean(x,y,z)={demo_stats['mean']}",
        f"  std(x,y,z)={demo_stats['std']}",
        f"  min(x,y,z)={demo_stats['min']}",
        f"  max(x,y,z)={demo_stats['max']}",
        "",
        "## live (G1 scripted)",
        f"  n={live_stats['n']}",
        f"  mean(x,y,z)={live_stats['mean']}",
        f"  std(x,y,z)={live_stats['std']}",
        f"  min(x,y,z)={live_stats['min']}",
        f"  max(x,y,z)={live_stats['max']}",
        "",
        verdict,
        "",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary))
    for line in summary:
        logger.info(line)

    _write_hists(outdir / "histograms.png", demo_xyz, live_xyz)
    logger.info("done. inspect %s/summary.txt", outdir)


if __name__ == "__main__":
    main()
