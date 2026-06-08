#!/usr/bin/env python3
"""GPU-pool dispatcher for the human-curriculum phase (stage1 easy -> stage2 full).

Each cell runs scripts/run_curriculum_cell.sh, which does stage1 (coworker_easy)
then stage2 (coworker_train) serially, resuming from a stage-0 snapshot with the
working isolation recipe. Cells are launched detached, pooled across FREE GPUs
(polls for more — zz4723 currently holds 1/2/3/5), retried once on crash, and
skipped if their stage2 final_metrics.json exists (idempotent / resumable).

  nohup venv/bin/python scripts/dispatch_curriculum.py > logs/curriculum/dispatch.log 2>&1 &
"""
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
AMASS = "/home/ap2322/Documents/CMU/CMU"
POLL = 90
GRAB_COOLDOWN = 180     # curriculum cells take longer to grab a GPU (resume + build)
MAX_RETRY = 1
EXCLUDE_GPUS: set[int] = set()

OUTROOT = REPO / "exp_local" / "curriculum"
LOGDIR = REPO / "logs" / "curriculum"
LOGDIR.mkdir(parents=True, exist_ok=True)
ISO = REPO / "exp_local" / "isolation"

# name, task, rung, seed, goal, demos, start (stage-0 snapshot to resume)
CELLS = [
    dict(name="dish_rung1_a", task="dishwasher_close", rung="rung1", seed=0, goal=0.0,
         demos=69, start=ISO / "dish_rung1_seed2/snapshot_best.pt"),          # 1.00
    dict(name="dish_rung1_b", task="dishwasher_close", rung="rung1", seed=1, goal=0.0,
         demos=69, start=ISO / "rung1_dish_idle_nows_69demo/snapshot_best.pt"),  # 0.80
    dict(name="drawers_rung3_a", task="drawers_open_all", rung="rung3", seed=0, goal=1.0,
         demos=54, start=ISO / "drawers_rung3_seed1/snapshot_best.pt"),       # 0.90
    dict(name="drawers_rung3_b", task="drawers_open_all", rung="rung3", seed=1, goal=1.0,
         demos=54, start=ISO / "drawers_rung3_seed2/snapshot_best.pt"),       # 0.90
    dict(name="drawers_rung1_a", task="drawers_open_all", rung="rung1", seed=0, goal=1.0,
         demos=54, start=ISO / "drawers_rung1_seed1/snapshot_best.pt"),       # 0.80
    dict(name="drawers_rung1_b", task="drawers_open_all", rung="rung1", seed=1, goal=1.0,
         demos=54, start=ISO / "drawers_rung1_seed2/snapshot_best.pt"),       # 0.80
]


def run_dir(c):
    return OUTROOT / c["name"]


def is_done(c):
    return (run_dir(c) / "stage2_full" / "final_metrics.json").exists()


def is_running(c):
    n = c["name"]
    for pat in (f"exp_local/curriculum/{n}/", f"run_curriculum_cell.sh {n}"):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            return True
    return False


def num_gpus():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def gpu_free(i):
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(i)],
        capture_output=True, text=True)
    return r.stdout.strip() == ""


def launch(c, gpu):
    run_dir(c).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AMASS_DATA_DIR=AMASS,
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               NAME=c["name"], TASK=c["task"], RUNG=c["rung"], SEED=str(c["seed"]),
               GOAL=str(c["goal"]), DEMOS=str(c["demos"]), START_SNAP=str(c["start"]),
               STAGE1_FRAMES="30000", STAGE2_FRAMES="40000")
    log = open(LOGDIR / f"{c['name']}.log", "a")
    log.write(f"\n==== launch {c['name']} on GPU{gpu} ====\n")
    log.flush()
    p = subprocess.Popen(["bash", "scripts/run_curriculum_cell.sh", c["name"]],
                         env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=REPO)
    return p.pid


def main():
    for c in CELLS:
        if not Path(c["start"]).exists():
            print(f"[dispatch] WARNING start snapshot missing for {c['name']}: {c['start']}", flush=True)
    attempts = {c["name"]: 0 for c in CELLS}
    cooldown = {}
    ng = num_gpus()
    print(f"[dispatch] {len(CELLS)} curriculum cells across {ng} GPUs", flush=True)
    while True:
        pending = [c for c in CELLS if not is_done(c)]
        if not pending:
            break
        actionable = [c for c in pending
                      if not is_running(c) and attempts[c["name"]] < MAX_RETRY + 1]
        if not actionable and not any(is_running(c) for c in pending):
            print("[dispatch] nothing actionable, nothing running -> stop", flush=True)
            break
        now = time.time()
        for c in actionable:
            if is_running(c) or is_done(c):
                continue
            gpu = None
            for i in range(ng):
                if i in EXCLUDE_GPUS or cooldown.get(i, 0) > now - GRAB_COOLDOWN:
                    continue
                if gpu_free(i):
                    gpu = i
                    break
            if gpu is None:
                continue
            attempts[c["name"]] += 1
            pid = launch(c, gpu)
            cooldown[gpu] = time.time()
            tag = "" if attempts[c["name"]] == 1 else f" retry{attempts[c['name']]-1}"
            print(f"[dispatch] gpu{gpu} <- {c['name']} pid={pid}{tag}", flush=True)
        time.sleep(POLL)

    summary = []
    for c in CELLS:
        fm = run_dir(c) / "stage2_full" / "final_metrics.json"
        rec = {"name": c["name"], "task": c["task"], "rung": c["rung"], "seed": c["seed"],
               "done": fm.exists(), "dir": str(run_dir(c))}
        if fm.exists():
            try:
                rec["final"] = json.loads(fm.read_text())
            except Exception:
                pass
        summary.append(rec)
    (OUTROOT / "dispatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[dispatch] DONE -> {OUTROOT/'dispatch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
