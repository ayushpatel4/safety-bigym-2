#!/usr/bin/env python3
"""GPU-pool dispatcher for the safety-phase operating-point eval.

Benchmarks the FINAL (deployed) checkpoint of each adaptive-lambda safety cell
under deployment-realistic noisy obs (3 rollout-seeds x 20 ep = 60 episodes,
bootstrap CIs) so the (task x budget) tradeoff vs the unconstrained curriculum
baseline is measured apples-to-apples — every cell's last-on-disk snapshot, no
selection bias, no snapshot-availability skew across seeds.

Headline metric = final-checkpoint deployed performance. Pooling the 3 train
seeds per (task,budget) is done afterwards by analyze_row3.py aggregate over the
per-cell *.episodes.jsonl. Readiness-gated on each cell's final_metrics.json so a
straggler's true final snapshot is captured, not a mid-training one.

  nohup venv/bin/python scripts/dispatch_safety_eval.py > logs/safety/eval_dispatch.log 2>&1 &
"""
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
AMASS = "/home/ap2322/Documents/CMU/CMU"
POLL = 90
GRAB_COOLDOWN = 180
MAX_RETRY = 1
SAFETY = REPO / "exp_local" / "safety"
OUT = REPO / "results" / "safety_eval"
OUT.mkdir(parents=True, exist_ok=True)
LOGDIR = REPO / "logs" / "safety"
LOGDIR.mkdir(parents=True, exist_ok=True)

# Deployable cells only (budgets 0.3, 0.5 across 3 seeds). The b01 cells collapsed
# to 0% task success in-training (budget too tight) — documented, not benchmarked.
DISH = ["dish_b03", "dish_b03_s1", "dish_b03_s2", "dish_b05", "dish_b05_s1", "dish_b05_s2"]
DRAW = ["drawers_b03", "drawers_b03_s1", "drawers_b03_s2", "drawers_b05", "drawers_b05_s1", "drawers_b05_s2"]
TASK = {**{c: "dishwasher_close" for c in DISH}, **{c: "drawers_open_all" for c in DRAW}}
CELLS = DISH + DRAW


def final_snapshot(cell):
    snaps = glob.glob(str(SAFETY / cell / "snapshot_[0-9]*.pt"))
    if not snaps:
        return None
    return max(snaps, key=lambda p: int(re.search(r"snapshot_(\d+)", os.path.basename(p)).group(1)))


def out_csv(cell):
    return OUT / f"{cell}.csv"


def is_done(cell):  # 180 ep target = 3 seeds x 20 ep x 3? no: 3 seeds x 20 = 60 per cell.
    # Use the live episodes.jsonl line count (written incrementally, reaches 60 just
    # BEFORE the proc exits + writes the CSV) as the race-free completion signal — the
    # CSV-only check raced with is_running at proc-exit and double-launched cells.
    j = out_csv(cell).with_suffix(".episodes.jsonl")
    if j.exists():
        try:
            if sum(1 for _ in j.open()) >= 60:
                return True
        except Exception:
            pass
    return out_csv(cell).exists() and out_csv(cell).stat().st_size > 0


def is_ready(cell):  # training cell finished -> final checkpoint is final
    return (SAFETY / cell / "final_metrics.json").exists() and final_snapshot(cell) is not None


def lock_path(cell):
    return OUT / f"{cell}.lock"


def is_running(cell):
    """Lock-based, deterministic across dispatcher restarts (reads /proc).
    A cell is 'running' iff its lockfile names a live PID whose cmdline still
    contains this cell's --out path. pgrep proved flaky at the proc-exit
    boundary and double-launched cells; /proc inspection does not."""
    lp = lock_path(cell)
    if not lp.exists():
        return False
    try:
        pid = int(lp.read_text().strip())
    except Exception:
        return False
    cl = Path(f"/proc/{pid}/cmdline")
    if not cl.exists():
        return False
    try:
        args = cl.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except Exception:
        return False
    return f"{out_csv(cell)}" in args and "benchmark_policy.py" in args


def num_gpus():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def gpu_free(i):
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(i)],
                       capture_output=True, text=True)
    return r.stdout.strip() == ""


def build_cmd(cell):
    return [
        "venv/bin/python", "scripts/benchmark_policy.py",
        "--snapshot", final_snapshot(cell),
        "--task", TASK[cell], "--disruption", "coworker_train", "--human-model", "g1",
        "--obs-mode", "noisy", "--num-demos-for-stats", "0",
        "--seeds", "0,1,2", "--episodes", "20",
        "--out", str(out_csv(cell)),
    ]


def launch(cell, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AMASS_DATA_DIR=AMASS,
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    log = open(LOGDIR / f"eval_{cell}.log", "a")
    log.write(f"\n==== benchmark {cell} on GPU{gpu} (snap={final_snapshot(cell)}) ====\n")
    log.flush()
    p = subprocess.Popen(build_cmd(cell), env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=REPO)
    lock_path(cell).write_text(str(p.pid))  # authoritative run marker (see is_running)
    return p.pid


def main():
    attempts = {c: 0 for c in CELLS}
    cooldown = {}
    ng = num_gpus()
    print(f"[safety-eval] {len(CELLS)} cells across {ng} GPUs", flush=True)
    while True:
        pending = [c for c in CELLS if not is_done(c)]
        if not pending:
            break
        actionable = [c for c in pending if is_ready(c) and not is_running(c)
                      and attempts[c] < MAX_RETRY + 1]
        waiting = [c for c in pending if not is_ready(c)]
        if not actionable and not any(is_running(c) for c in pending) and not waiting:
            print("[safety-eval] nothing actionable/running/waiting -> stop", flush=True)
            break
        now = time.time()
        for c in actionable:
            if is_running(c) or is_done(c):
                continue
            gpu = None
            for i in range(ng):
                if cooldown.get(i, 0) > now - GRAB_COOLDOWN:
                    continue
                if gpu_free(i):
                    gpu = i
                    break
            if gpu is None:
                continue
            attempts[c] += 1
            pid = launch(c, gpu)
            cooldown[gpu] = time.time()
            print(f"[safety-eval] gpu{gpu} <- {c} pid={pid}", flush=True)
        time.sleep(POLL)

    summary = {c: ({"done": is_done(c), "csv": str(out_csv(c)), "snapshot": final_snapshot(c)})
               for c in CELLS}
    (OUT / "eval_dispatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[safety-eval] DONE -> {OUT/'eval_dispatch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
