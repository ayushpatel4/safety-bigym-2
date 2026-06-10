#!/usr/bin/env python3
"""Multi-GPU dispatcher for the WCSAC (E3.7 / P9) external-baseline sweep.

Polls for *genuinely idle* GPUs (low memory AND low utilisation) so it
coexists with other running jobs -- it never grabs a GPU that's busy. Runs the
WCSAC cells (task x cost_budget) across whatever frees up, respawns crashed
cells up to MAX_RETRY, and exits once every cell has written final_metrics.json
(or its retries are exhausted).

Backgrounded, multi-hour. Launch:
    AMASS_DATA_DIR=/home/ap2322/Documents/CMU/CMU \
      venv/bin/python scripts/dispatch_wcsac.py

Tunables via env: FRAMES, BUDGETS ("5 15 30"), TASKS, SEED, DEMOS,
EVAL_EVERY, NUM_EVAL, FREE_MEM_MB, FREE_UTIL.
"""
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AMASS = os.environ.get("AMASS_DATA_DIR", "/home/ap2322/Documents/CMU/CMU")
OUTROOT = REPO / "exp_local" / "wcsac"
LOGDIR = REPO / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
OUTROOT.mkdir(parents=True, exist_ok=True)

FRAMES = int(os.environ.get("FRAMES", "150000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "10000"))
NUM_EVAL = int(os.environ.get("NUM_EVAL", "10"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2").split()]
DEMOS = int(os.environ.get("DEMOS", "0"))
BUDGETS = [float(b) for b in os.environ.get("BUDGETS", "5 15 30").split()]
TASKS = os.environ.get(
    "TASKS", "dishwasher_close drawers_open_all saucepan_to_hob").split()

FREE_MEM_MB = int(os.environ.get("FREE_MEM_MB", "2000"))  # idle iff mem < this ...
FREE_UTIL = int(os.environ.get("FREE_UTIL", "25"))        # ... AND util% < this
POLL = 30            # seconds between scheduling passes
GRAB_COOLDOWN = 120  # don't reuse a GPU within this window of launching on it
MAX_RETRY = 2        # extra relaunches per cell after a crash


def cells():
    out = []
    for task in TASKS:
        for b in BUDGETS:
            for seed in SEEDS:
                out.append(dict(name=f"wcsac_{task}_b{b:g}_s{seed}",
                                task=task, budget=b, seed=seed))
    return out


CELLS = cells()
procs = {}  # name -> (Popen, gpu)


def run_dir(c):
    return OUTROOT / c["name"]


def is_done(c):
    return (run_dir(c) / "final_metrics.json").exists()


def is_running(c):
    p = procs.get(c["name"])
    return p is not None and p[0].poll() is None


def build_cmd(c):
    return [
        "venv/bin/python", "train_cqn_as.py",
        f"env=safety_bigym/{c['task']}", "env.human_model=g1", "env.smplh_motion=amass",
        "agent=wcsac", f"agent.cost_budget={c['budget']:g}",
        "bodyslam=oracle", "disruption=coworker_train", f"num_demos={DEMOS}",
        f"num_train_frames={FRAMES}", f"eval_every_frames={EVAL_EVERY}",
        f"num_eval_episodes={NUM_EVAL}", f"seed={c['seed']}",
        "save_snapshot=true", "save_video=true",
        "wandb.use=true", "wandb.project=safety-critic", f"wandb.name={c['name']}",
        "+wandb.tags=[phase-3,wcsac,E3.7,external_baseline,"
        f"task:{c['task']},cost_budget:{c['budget']:g}]",
        f"hydra.run.dir={run_dir(c)}",
    ]


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        stats = {}
        for line in out.stdout.strip().splitlines():
            idx, mem, util = (x.strip() for x in line.split(","))
            stats[int(idx)] = (int(mem), int(util))
        return stats
    except Exception as e:  # nvidia-smi hiccup -> treat as no free GPUs this pass
        print(f"[wcsac] gpu_stats failed: {e}", flush=True)
        return {}


def launch(c, gpu):
    run_dir(c).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AMASS_DATA_DIR=AMASS,
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    log = open(LOGDIR / f"{c['name']}.log", "a")
    log.write(f"\n==== launch {c['name']} on GPU{gpu} FRAMES={FRAMES} "
              f"({time.ctime()}) ====\n")
    log.flush()
    p = subprocess.Popen(build_cmd(c), env=env, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO)
    procs[c["name"]] = (p, gpu)
    return p.pid


def main():
    attempts = {c["name"]: 0 for c in CELLS}
    cooldown = {}
    print(f"[wcsac] {len(CELLS)} cells x {FRAMES} frames: "
          f"{[c['name'] for c in CELLS]}", flush=True)
    while True:
        pending = [c for c in CELLS if not is_done(c)]
        if not pending:
            break
        actionable = [c for c in pending
                      if not is_running(c) and attempts[c["name"]] <= MAX_RETRY]
        if not actionable and not any(is_running(c) for c in pending):
            print("[wcsac] nothing actionable or running -> stop "
                  "(remaining cells exhausted retries)", flush=True)
            break
        stats = gpu_stats()
        now = time.time()
        my_gpus = {g for (p, g) in procs.values() if p.poll() is None}
        for c in actionable:
            if is_running(c) or is_done(c):
                continue
            gpu = None
            for i, (mem, util) in sorted(stats.items()):
                if i in my_gpus or cooldown.get(i, 0) > now - GRAB_COOLDOWN:
                    continue
                if mem < FREE_MEM_MB and util < FREE_UTIL:
                    gpu = i
                    break
            if gpu is None:
                continue  # no free GPU this pass; try next poll
            attempts[c["name"]] += 1
            pid = launch(c, gpu)
            cooldown[gpu] = time.time()
            my_gpus.add(gpu)
            stats[gpu] = (FREE_MEM_MB + 1, FREE_UTIL + 1)  # consumed this pass
            print(f"[wcsac] gpu{gpu} <- {c['name']} (budget {c['budget']:g}) "
                  f"pid={pid} try={attempts[c['name']]}", flush=True)
        done = sum(is_done(c) for c in CELLS)
        (OUTROOT / "dispatch_progress.json").write_text(json.dumps(
            {"ts": time.ctime(), "done": done, "total": len(CELLS),
             "running": [c["name"] for c in CELLS if is_running(c)],
             "attempts": attempts}, indent=2))
        time.sleep(POLL)

    summary = []
    for c in CELLS:
        fm = run_dir(c) / "final_metrics.json"
        rec = {"name": c["name"], "task": c["task"], "budget": c["budget"],
               "done": fm.exists(), "dir": str(run_dir(c))}
        if fm.exists():
            try:
                rec["final"] = json.loads(fm.read_text())
            except Exception:
                pass
        summary.append(rec)
    (OUTROOT / "dispatch_summary.json").write_text(json.dumps(summary, indent=2))
    ndone = sum(r["done"] for r in summary)
    print(f"[wcsac] DONE {ndone}/{len(CELLS)} -> "
          f"{OUTROOT / 'dispatch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
