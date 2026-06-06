#!/usr/bin/env python
"""P6 — snapshot-evaluation benchmark harness.

Rolls out a policy snapshot (CQN-AS or ACT) — or a random policy when no snapshot is
given — per (task, disruption, obs-mode) cell over one or more seeds, optionally under the
Phase-2 SVF runtime filter, and appends one CSV row (the canonical per-cell schema, with
bootstrap CIs + CVaR/percentile tail-risk + filter mechanics) to ``--out``. The raw
per-episode rolls are also written to a parquet sidecar (``<out>.raw_episodes.parquet``)
so the CSV can be re-aggregated without re-rolling out, plus a live ``.episodes.jsonl``.

Examples
--------
    # smoke (random policy, G1, <5 min CPU): one cell, non-empty CSV
    python scripts/benchmark_policy.py --smoke --out results/smoke.csv

    # headline-style cell on a trained policy + the SVF filter
    python scripts/benchmark_policy.py \\
        --snapshot runs/saucepan_g1/final.pt \\
        --filter-snapshot checkpoints/svf_coworker_train_g1_0p3.pt --filter-threshold 2.25 \\
        --task saucepan_to_hob --disruption coworker_train --obs-mode noisy \\
        --human-model g1 --seeds 0,1,2 --episodes 20 --out results/row5.csv

Notes
-----
* Filtering requires ``--obs-mode oracle|noisy`` (the SVF critic consumes
  ``human_pos_estimate``); the harness hard-errors otherwise.
* CQN-AS eval re-derives demo action-stats via DemoStore, which needs ``AMASS_DATA_DIR``
  exported when ``--obs-mode != off`` (demo human-pos injection). G1 *live* rollouts are
  AMASS-free; only the CQN-AS demo-stat step needs it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("benchmark_policy")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, default=None, help="Policy checkpoint (ACT/CQN-AS). Omit for random.")
    p.add_argument("--safety-filter", choices=("none", "svf", "cbf"), default=None,
                   help="Runtime safety filter. 'svf' = learned Safety-Value-Function veto "
                        "(needs --filter-snapshot). 'cbf' = geometric CBF directional-dodge "
                        "(model-based, no checkpoint; CQN-AS only). Default: inferred — 'svf' "
                        "if --filter-snapshot is given, else 'none'.")
    p.add_argument("--filter-snapshot", type=Path, default=None, help="SVF critic checkpoint to wrap the policy.")
    p.add_argument("--filter-threshold", type=float, default=2.25, help="SVF Q-value threshold R (filter triggers q<R). 2.25 = G1 dense-0.3m-sweep operating point (snapshots.py).")
    p.add_argument("--fallback", default="zero_velocity")
    # CBF directional-dodge knobs (only used when --safety-filter cbf).
    p.add_argument("--cbf-d-target", type=float, default=0.45,
                   help="CBF barrier offset (m): keep robot-base<->human separation >= this. "
                        "Default 0.45 (margin above the 0.30 m violation threshold).")
    p.add_argument("--cbf-gain", type=float, default=1.0, help="CBF correction gain on (d_target - sep).")
    p.add_argument("--cbf-max-push", type=float, default=0.15, help="CBF per-step base-target offset cap (m).")
    p.add_argument("--cbf-beta", type=float, default=0.1, help="CBF approach-velocity term weight (with --cbf-use-velocity).")
    p.add_argument("--cbf-use-velocity", dest="cbf_use_velocity", action="store_true", default=True,
                   help="Add an approach-velocity term to the CBF push (default on).")
    p.add_argument("--cbf-no-velocity", dest="cbf_use_velocity", action="store_false",
                   help="Disable the CBF approach-velocity term (geometric-only dodge).")
    p.add_argument("--task", default="saucepan_to_hob")
    p.add_argument("--disruption", default="coworker_train")
    p.add_argument("--obs-mode", choices=("off", "oracle", "noisy"), default="noisy")
    p.add_argument("--human-model", choices=("g1", "smplh"), default="g1")
    p.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1,2.")
    p.add_argument("--episodes", type=int, default=20, help="Episodes per seed.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Episode step cap. Default: the snapshot's natural horizon "
                        "(CQN-AS: episode_length//demo_down_sample_rate; else 1000) so "
                        "long-horizon tasks aren't silently truncated. --smoke forces 50.")
    p.add_argument("--out", type=Path, required=True, help="Per-cell CSV (appended).")
    p.add_argument("--stats-seed", type=int, default=12345, help="Bootstrap RNG seed (reproducible CIs).")
    p.add_argument("--num-resamples", type=int, default=10000)
    p.add_argument("--num-demos-for-stats", type=int, default=0,
                   help="CQN-AS: cap demo count for action-stat derivation (0=use snapshot's "
                        "num_demos; lower it on memory-constrained machines).")
    p.add_argument("--render", action="store_true", help="Write rollout mp4(s) next to --out (best-effort).")
    p.add_argument("--render-episodes", type=int, default=1,
                   help="How many of the first scored episodes to record when --render is set "
                        "(one mp4 each: <out-stem>_videos/step_<i>_ep0.mp4).")
    p.add_argument("--smoke", action="store_true", help="1 seed x 2 episodes x 50 steps, single cell.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _derive_max_steps(meta, payload) -> int:
    """Default episode cap = the snapshot's natural horizon.

    For CQN-AS this is the adapter's hard timelimit
    (``episode_length // demo_down_sample_rate``); for ACT we read the same fields off the
    RoboBase cfg when present. Otherwise a generous 1000 (the env's own truncation usually
    fires first). Using a fixed small cap silently truncates long-horizon tasks like
    saucepan_to_hob (~1000 control steps) and collapses success_rate.
    """
    def _horizon(env_cfg) -> int | None:
        try:
            el = int(env_cfg.get("episode_length"))
            dr = int(env_cfg.get("demo_down_sample_rate", 1) or 1)
            return max(1, el // max(dr, 1))
        except Exception:
            return None

    if isinstance(payload, dict):
        if meta.kind == "cqn_as":
            cfg = payload.get("config") or {}
            h = _horizon((cfg.get("env") or {}) if isinstance(cfg, dict) else {})
            if h:
                return h
        elif meta.kind == "act":
            cfg = payload.get("cfg")
            try:
                h = _horizon(cfg["env"])  # OmegaConf supports __getitem__
                if h:
                    return h
            except Exception:
                pass
    return 1000


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _make_frame_sink(renderable, frames: List):
    """Return an on_step callback that captures one render frame per call (best-effort)."""
    from safety_bigym.agents.cqn_as.eval_video import render_frame

    def _sink():
        try:
            frame = render_frame(renderable, global_step=0)
            if frame is not None:
                frames.append(frame)
        except Exception as exc:  # pragma: no cover — rendering is best-effort
            logger.warning("Frame capture skipped (%s: %s)", type(exc).__name__, exc)

    return _sink


def _write_episode_video(frames: List, out_dir: Path, episode_index: int) -> None:
    from safety_bigym.agents.cqn_as.eval_video import write_eval_video

    if not frames:
        return
    try:
        write_eval_video(out_dir, frames, global_step=episode_index, wandb_run=None)
    except Exception as exc:  # pragma: no cover
        logger.warning("Video write skipped (%s: %s)", type(exc).__name__, exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from safety_bigym.benchmark.loader import load_policy
    from safety_bigym.benchmark.runners import build_cell_runner, run_episode
    from safety_bigym.benchmark.aggregate import aggregate_cell
    from safety_bigym.benchmark.schema import COLUMNS, assemble_row
    from safety_bigym.benchmark.records import write_jsonl_line, write_parquet

    if args.smoke:
        args.seeds = "0"
        args.episodes = 2
        args.max_steps = 50

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]

    # Resolve the filter mode. Default is inferred for backward-compat: 'svf' when a
    # critic checkpoint is given, else 'none'.
    if args.safety_filter is None:
        safety_filter = "svf" if args.filter_snapshot is not None else "none"
    else:
        safety_filter = args.safety_filter

    if safety_filter == "svf" and args.filter_snapshot is None:
        raise SystemExit("--safety-filter svf requires --filter-snapshot <critic.pt>.")
    if safety_filter == "cbf" and args.filter_snapshot is not None:
        raise SystemExit("--safety-filter cbf takes no --filter-snapshot (it is critic-free).")

    use_svf = safety_filter == "svf"
    use_cbf = safety_filter == "cbf"
    filtered = use_svf or use_cbf  # both populate intervention/filter-step bookkeeping

    if filtered and args.obs_mode == "off":
        which = "SVF critic" if use_svf else "CBF dodge"
        raise SystemExit(f"--safety-filter {safety_filter} requires --obs-mode oracle|noisy "
                         f"(the {which} consumes human_pos_estimate).")

    meta, payload = load_policy(args.snapshot)
    if use_cbf and meta.kind != "cqn_as":
        raise SystemExit(f"--safety-filter cbf is implemented for CQN-AS snapshots only; "
                         f"got policy kind {meta.kind!r}.")
    if args.max_steps is None:
        args.max_steps = _derive_max_steps(meta, payload)
    logger.info("Policy kind=%s (snapshot=%s) max_steps=%d safety_filter=%s",
                meta.kind, args.snapshot, args.max_steps, safety_filter)

    filter_critic = None
    if use_svf:
        from safety_bigym.benchmark.filter_attach import load_critic

        filter_critic = load_critic(args.filter_snapshot)

    cbf_config = None
    if use_cbf:
        cbf_config = dict(
            d_target=args.cbf_d_target,
            gain=args.cbf_gain,
            max_push=args.cbf_max_push,
            use_velocity=bool(args.cbf_use_velocity),
            beta=args.cbf_beta,
        )
        logger.info("CBF dodge config: %s", cbf_config)

    runner, renderable = build_cell_runner(
        meta, payload,
        snapshot_path=args.snapshot,
        task=args.task, disruption=args.disruption, obs_mode=args.obs_mode,
        human_model=args.human_model,
        filter_critic=filter_critic, filter_threshold=args.filter_threshold,
        fallback_name=args.fallback, num_demos_for_stats=args.num_demos_for_stats,
        cbf_config=cbf_config,
    )

    jsonl_path = args.out.with_suffix(".episodes.jsonl")
    parquet_path = args.out.with_suffix(".raw_episodes.parquet")

    # Render the first N scored episodes inline (no extra rollouts). Key the video
    # dir off the out-CSV stem so multiple cells sharing one out-dir (e.g. the E4.1
    # driver's per-row CSVs) don't overwrite each other's step_<i>_ep0.mp4.
    render_eps = int(args.render_episodes) if args.render else 0
    video_dir = args.out.parent / f"{args.out.stem}_videos"

    records = []
    idx = 0
    for seed in seeds:
        for ep in range(int(args.episodes)):
            env_seed = seed * 100_000 + ep
            frames: List = []
            sink = _make_frame_sink(renderable, frames) if idx < render_eps else None
            rec = run_episode(runner, seed=env_seed, episode_index=idx,
                              max_steps=int(args.max_steps), filtered=filtered, on_step=sink)
            if sink is not None:
                _write_episode_video(frames, video_dir, idx)
            records.append(rec)
            write_jsonl_line(jsonl_path, rec)
            idx += 1
            logger.info("seed=%d ep=%d: reward=%.3f success=%s prox_rate=%.3f%s",
                        seed, ep, rec.episode_reward, rec.success,
                        rec.ep_safety.get("ep_proximity_violation_rate", float("nan")),
                        f" interv={rec.n_interventions}" if filtered else "")

    if render_eps:
        logger.info("Wrote up to %d rollout video(s) to %s", render_eps, video_dir)
    runner.close()

    if use_cbf:
        filter_meta = {"fallback": f"cbf_dodge(d_target={args.cbf_d_target},"
                                   f"gain={args.cbf_gain},max_push={args.cbf_max_push},"
                                   f"use_vel={bool(args.cbf_use_velocity)},beta={args.cbf_beta})"}
    elif use_svf:
        filter_meta = {"fallback": args.fallback}
    else:
        filter_meta = None
    agg = aggregate_cell(records, filter_meta=filter_meta,
                         stats_seed=args.stats_seed, n_resamples=args.num_resamples)

    identification = {
        "task": args.task,
        "disruption": args.disruption,
        "obs_mode": args.obs_mode,
        "human_model": args.human_model,
        "policy_kind": meta.kind,
        "snapshot": str(args.snapshot) if args.snapshot else "",
        "filter_snapshot": ("cbf_dodge" if use_cbf else (str(args.filter_snapshot) if use_svf else "")),
        "filter_threshold": (args.cbf_d_target if use_cbf else (args.filter_threshold if use_svf else "")),
        "seeds": ",".join(str(s) for s in seeds),
        "episodes_per_seed": int(args.episodes),
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    row = assemble_row({**identification, **agg}, filtered=filtered)

    # Append CSV (header iff new file), then write the raw-rolls parquet.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    write_parquet(parquet_path, records)

    logger.info("Wrote 1 cell row to %s (%d episodes) + raw rolls to %s",
                args.out, len(records), parquet_path)
    print(f"success_rate={row['success_rate']} "
          f"ep_proximity_violation_rate={row['ep_proximity_violation_rate']} "
          f"ep_min_separation={row['ep_min_separation']}"
          + (f" filter_intervention_rate={row['filter_intervention_rate']}" if filtered else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
