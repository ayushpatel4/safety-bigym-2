#!/usr/bin/env python3
"""GPU-pool dispatcher for the base-task isolation campaign (rung1 vs rung3).

Runs the dishwasher_close / drawers_open_all trainability experiments across
whatever GPUs are free, polling for more as they free up. Each cell is launched
detached (survives the dispatcher), retried once on crash, and skipped if it
already has final_metrics.json (idempotent / resumable). Writes a summary JSON
at the end.

  nohup venv/bin/python scripts/dispatch_isolation.py > logs/isolation/dispatch.log 2>&1 &

The two seed-1 dishwasher cells (rung1 + rung3) are launched separately and run
on GPUs 0/3; this dispatcher fills the remaining GPUs with the drawers cells and
the 2nd-seed robustness cells, polling for 0/3 once they free.
"""
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

AMASS = "/home/ap2322/Documents/CMU/CMU"
POLL = 90                      # seconds between scheduling passes
GRAB_COOLDOWN = 150            # seconds a just-launched GPU is treated as taken
MAX_RETRY = 1                  # re-launches after a crash
EXCLUDE_GPUS: set[int] = set() # GPUs never to use (busy ones are skipped anyway)

OUTROOT = REPO / "exp_local" / "isolation"
LOGDIR = REPO / "logs" / "isolation"
LOGDIR.mkdir(parents=True, exist_ok=True)

# name, task, demos, progress(bool), beta, goal, seed, frames
CELLS = [
    dict(name="drawers_rung1_seed1", task="drawers_open_all", demos=54,
         progress=False, beta=1.0, goal=1.0, seed=1, frames=40000),
    dict(name="drawers_rung3_seed1", task="drawers_open_all", demos=54,
         progress=True, beta=1.0, goal=1.0, seed=1, frames=40000),
    dict(name="dish_rung3_seed2", task="dishwasher_close", demos=69,
         progress=True, beta=1.0, goal=0.0, seed=2, frames=30000),
    dict(name="drawers_rung3_seed2", task="drawers_open_all", demos=54,
         progress=True, beta=1.0, goal=1.0, seed=2, frames=40000),
    dict(name="dish_rung1_seed2", task="dishwasher_close", demos=69,
         progress=False, beta=1.0, goal=0.0, seed=2, frames=30000),
    dict(name="drawers_rung1_seed2", task="drawers_open_all", demos=54,
         progress=False, beta=1.0, goal=1.0, seed=2, frames=40000),
]


def run_dir(c):
    return OUTROOT / c["name"]


def is_done(c):
    return (run_dir(c) / "final_metrics.json").exists()


def is_running(c):
    # hydra.run.dir=<dir> is the LAST arg (no trailing space), so match the dir
    # followed by a space or end-of-cmdline to avoid false negatives / prefixes.
    r = subprocess.run(["pgrep", "-f", f"hydra.run.dir={run_dir(c)}($| )"],
                       capture_output=True)
    return r.returncode == 0


def num_gpus():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def gpu_free(i):
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(i)],
        capture_output=True, text=True)
    return r.stdout.strip() == ""


def build_cmd(c):
    rung = "rung3" if c["progress"] else "rung1"
    return [
        "venv/bin/python", "train_cqn_as.py",
        f"env=safety_bigym/{c['task']}", "env.human_model=g1", "env.smplh_motion=amass",
        "bodyslam=oracle", f"num_demos={c['demos']}",
        "env.safety.add_workspace_penalty=false",
        f"env.safety.add_progress_reward={'true' if c['progress'] else 'false'}",
        f"env.safety.progress_beta={c['beta']}",
        f"env.safety.progress_goal={c['goal']}",
        "env.safety.progress_gamma=1.0",
        "agent.v_min=-6.0", "agent.v_max=2.0", "agent.atoms=101",
        "disruption=coworker_idle",
        f"num_train_frames={c['frames']}", "eval_every_frames=2500", "num_eval_episodes=10",
        f"seed={c['seed']}",
        "save_snapshot=true", "save_video=true",
        "wandb.use=true", "wandb.project=safety-critic",
        f"wandb.name=isolation_{c['name']}",
        f"+wandb.tags=[isolation,{rung},task:{c['task']},seed:{c['seed']}]",
        f"hydra.run.dir={run_dir(c)}",
    ]


def launch(c, gpu):
    d = run_dir(c)
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AMASS_DATA_DIR=AMASS,
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    log = open(LOGDIR / f"{c['name']}.log", "a")
    log.write(f"\n==== launch {c['name']} on GPU{gpu} ====\n")
    log.flush()
    p = subprocess.Popen(build_cmd(c), env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=REPO)
    return p.pid


def main():
    attempts = {c["name"]: 0 for c in CELLS}
    cooldown = {}  # gpu -> ts last launched
    ng = num_gpus()
    print(f"[dispatch] {len(CELLS)} cells across {ng} GPUs (excl {EXCLUDE_GPUS})", flush=True)
    while True:
        pending = [c for c in CELLS if not is_done(c)]
        if not pending:
            break
        actionable = [c for c in pending
                      if not is_running(c) and attempts[c["name"]] < MAX_RETRY + 1]
        if not actionable and not any(is_running(c) for c in pending):
            print("[dispatch] nothing actionable and nothing running -> stop", flush=True)
            break
        now = time.time()
        for c in actionable:
            if is_running(c) or is_done(c):
                continue
            gpu = None
            for i in range(ng):
                if i in EXCLUDE_GPUS:
                    continue
                if cooldown.get(i, 0) > now - GRAB_COOLDOWN:
                    continue
                if gpu_free(i):
                    gpu = i
                    break
            if gpu is None:
                continue
            attempts[c["name"]] += 1
            pid = launch(c, gpu)
            cooldown[gpu] = time.time()
            tag = "" if attempts[c["name"]] == 1 else f" (retry {attempts[c['name']]-1})"
            print(f"[dispatch] gpu{gpu} <- {c['name']} pid={pid}{tag}", flush=True)
        time.sleep(POLL)

    summary = []
    for c in CELLS:
        fm = run_dir(c) / "final_metrics.json"
        rec = {"name": c["name"], "task": c["task"], "rung": "rung3" if c["progress"] else "rung1",
               "seed": c["seed"], "done": fm.exists(), "dir": str(run_dir(c))}
        if fm.exists():
            try:
                rec["final"] = json.loads(fm.read_text())
            except Exception:
                pass
        summary.append(rec)
    (OUTROOT / "dispatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[dispatch] DONE. summary -> {OUTROOT/'dispatch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
