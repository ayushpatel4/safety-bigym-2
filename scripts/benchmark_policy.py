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
        --filter-snapshot svf_coworker_train_v1.pt --filter-threshold 4.0 \\
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
    p.add_argument("--filter-snapshot", type=Path, default=None, help="SVF critic checkpoint to wrap the policy.")
    p.add_argument("--filter-threshold", type=float, default=4.0, help="SVF Q-value threshold R (filter triggers q<R).")
    p.add_argument("--fallback", default="zero_velocity")
    p.add_argument("--task", default="saucepan_to_hob")
    p.add_argument("--disruption", default="coworker_train")
    p.add_argument("--obs-mode", choices=("off", "oracle", "noisy"), default="noisy")
    p.add_argument("--human-model", choices=("g1", "smplh"), default="g1")
    p.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1,2.")
    p.add_argument("--episodes", type=int, default=20, help="Episodes per seed.")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", type=Path, required=True, help="Per-cell CSV (appended).")
    p.add_argument("--stats-seed", type=int, default=12345, help="Bootstrap RNG seed (reproducible CIs).")
    p.add_argument("--num-resamples", type=int, default=10000)
    p.add_argument("--num-demos-for-stats", type=int, default=0,
                   help="CQN-AS: cap demo count for action-stat derivation (0=use snapshot's "
                        "num_demos; lower it on memory-constrained machines).")
    p.add_argument("--render", action="store_true", help="Write an mp4 of one rollout next to --out (best-effort).")
    p.add_argument("--smoke", action="store_true", help="1 seed x 2 episodes x 50 steps, single cell.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _render_rollout(runner, renderable, *, seed: int, max_steps: int, out_dir: Path, global_step: int = 0):
    """Best-effort: roll one episode capturing frames -> mp4 (eval_video helpers)."""
    from safety_bigym.agents.cqn_as.eval_video import render_frame, write_eval_video

    frames: List = []
    try:
        runner.reset(seed)
        frame = render_frame(renderable, global_step=global_step)
        if frame is not None:
            frames.append(frame)
        for _ in range(max_steps):
            rec = runner.step()
            frame = render_frame(renderable, global_step=global_step)
            if frame is not None:
                frames.append(frame)
            if rec.done:
                break
        if frames:
            write_eval_video(out_dir, frames, global_step=global_step, wandb_run=None)
            logger.info("Wrote rollout video to %s", out_dir)
    except Exception as exc:  # pragma: no cover — rendering is best-effort
        logger.warning("Render skipped (%s: %s)", type(exc).__name__, exc)


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
    filtered = args.filter_snapshot is not None

    if filtered and args.obs_mode == "off":
        raise SystemExit("--filter-snapshot requires --obs-mode oracle|noisy "
                         "(the SVF critic consumes human_pos_estimate).")

    meta, payload = load_policy(args.snapshot)
    logger.info("Policy kind=%s (snapshot=%s)", meta.kind, args.snapshot)

    filter_critic = None
    if filtered:
        from safety_bigym.benchmark.filter_attach import load_critic

        filter_critic = load_critic(args.filter_snapshot)

    runner, renderable = build_cell_runner(
        meta, payload,
        snapshot_path=args.snapshot,
        task=args.task, disruption=args.disruption, obs_mode=args.obs_mode,
        human_model=args.human_model,
        filter_critic=filter_critic, filter_threshold=args.filter_threshold,
        fallback_name=args.fallback, num_demos_for_stats=args.num_demos_for_stats,
    )

    jsonl_path = args.out.with_suffix(".episodes.jsonl")
    parquet_path = args.out.with_suffix(".raw_episodes.parquet")

    records = []
    idx = 0
    for seed in seeds:
        for ep in range(int(args.episodes)):
            env_seed = seed * 100_000 + ep
            rec = run_episode(runner, seed=env_seed, episode_index=idx,
                              max_steps=int(args.max_steps), filtered=filtered)
            records.append(rec)
            write_jsonl_line(jsonl_path, rec)
            idx += 1
            logger.info("seed=%d ep=%d: reward=%.3f success=%s prox_rate=%.3f%s",
                        seed, ep, rec.episode_reward, rec.success,
                        rec.ep_safety.get("ep_proximity_violation_rate", float("nan")),
                        f" interv={rec.n_interventions}" if filtered else "")

    if args.render:
        _render_rollout(runner, renderable, seed=seeds[0] * 100_000,
                        max_steps=int(args.max_steps), out_dir=args.out.parent / "benchmark_videos")
    runner.close()

    filter_meta = {"fallback": args.fallback} if filtered else None
    agg = aggregate_cell(records, filter_meta=filter_meta,
                         stats_seed=args.stats_seed, n_resamples=args.num_resamples)

    identification = {
        "task": args.task,
        "disruption": args.disruption,
        "obs_mode": args.obs_mode,
        "human_model": args.human_model,
        "policy_kind": meta.kind,
        "snapshot": str(args.snapshot) if args.snapshot else "",
        "filter_snapshot": str(args.filter_snapshot) if filtered else "",
        "filter_threshold": (args.filter_threshold if filtered else ""),
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
