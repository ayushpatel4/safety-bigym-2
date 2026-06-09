#!/usr/bin/env python3
"""GPU-pool dispatcher for the warm-started-cost-estimate variant of the budget sweep.

The plain budget sweep (dispatch_budget_sweep.py) showed the constraint binds late:
the PID's rolling-cost estimate warms up from 0 over ~20k frames (0.99 momentum), so
λ stayed inert through the first half and then bang-banged into a task-collapsing bind.
This variant sets `agent.rolling_cost_init` to each task's measured natural cost so the
PID sees the true cost from frame 0 and λ binds *gently from the start* — testing
whether early, graceful binding lets a budget at/just-below the ~0.21-0.23 cost floor
settle into a feasible (safe AND successful) policy instead of collapsing.

Budgets {0.20, 0.22} × {dishwasher_close, drawers_open_all}, seed 0, 40k frames.
Everything else matches dispatch_budget_sweep.py. Memory-threshold gpu_free (co-locates
with another user's light CUDA contexts on the shared box).

  nohup venv/bin/python scripts/dispatch_budget_warmstart.py > logs/budget_warmstart/dispatch.log 2>&1 &
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
GRAB_COOLDOWN = 180
MAX_RETRY = 1
FRAMES = 40000
FREE_MEM_MIB = 1500
CUR = REPO / "exp_local" / "curriculum"
OUTROOT = REPO / "exp_local" / "budget_warmstart"
LOGDIR = REPO / "logs" / "budget_warmstart"
LOGDIR.mkdir(parents=True, exist_ok=True)

# Per-task natural rolling cost (measured from the plain-sweep snapshots' PID state):
# dishwasher ~0.25, drawers ~0.22.
NATURAL = {"dishwasher_close": 0.25, "drawers_open_all": 0.22}
BUDGETS = [0.20, 0.22]


def _cells():
    out = []
    for b in BUDGETS:
        tag = str(b).replace("0.", "b").replace(".", "")  # 0.20 -> b20
        out.append(dict(name=f"dish_{tag}_ws", task="dishwasher_close", demos=69,
                        budget=b, warm="dish_rung1_a", seed=0))
        out.append(dict(name=f"drawers_{tag}_ws", task="drawers_open_all", demos=54,
                        budget=b, warm="drawers_rung1_a", seed=0))
    return out


CELLS = _cells()


def warm(cell):
    return CUR / cell / "stage2_full" / "snapshot_best.pt"


def run_dir(c):
    return OUTROOT / c["name"]


def is_done(c):
    return (run_dir(c) / "final_metrics.json").exists()


def is_ready(c):
    return warm(c["warm"]).exists()


def is_running(c):
    return subprocess.run(["pgrep", "-f", f"hydra.run.dir={run_dir(c)}($| )"],
                          capture_output=True).returncode == 0


def num_gpus():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def gpu_mem(i):
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(i)],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 1 << 30


def gpu_free(i):
    return gpu_mem(i) < FREE_MEM_MIB


def build_cmd(c):
    return [
        "venv/bin/python", "train_cqn_as.py",
        f"env=safety_bigym/{c['task']}", "env.human_model=g1", "env.smplh_motion=amass",
        "bodyslam=oracle", f"num_demos={c['demos']}",
        "agent=cqn_as_lagrangian", f"agent.cost_budget={c['budget']}",
        f"agent.rolling_cost_init={NATURAL[c['task']]}",
        "agent.v_min=-6.0", "agent.v_max=2.0", "agent.atoms=101",
        "env.safety.add_workspace_penalty=false",
        "disruption=coworker_train", f"num_train_frames={FRAMES}",
        "eval_every_frames=2500", "num_eval_episodes=10",
        f"seed={c['seed']}", "save_snapshot=true", "save_video=false",
        "wandb.use=true", "wandb.project=safety-critic", f"wandb.name=bws_{c['name']}",
        f"+wandb.tags=[budget_warmstart,adaptive_lambda,budget:{c['budget']},task:{c['task']}]",
        f"+snapshot_path={warm(c['warm'])}",
        f"hydra.run.dir={run_dir(c)}",
    ]


def launch(c, gpu):
    run_dir(c).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AMASS_DATA_DIR=AMASS,
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    log = open(LOGDIR / f"{c['name']}.log", "a")
    log.write(f"\n==== launch {c['name']} (budget {c['budget']}, ws_init {NATURAL[c['task']]}) on GPU{gpu} ====\n")
    log.flush()
    p = subprocess.Popen(build_cmd(c), env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=REPO)
    return p.pid


def main():
    attempts = {c["name"]: 0 for c in CELLS}
    cooldown = {}
    ng = num_gpus()
    print(f"[bws] {len(CELLS)} cells across {ng} GPUs; budgets={BUDGETS} natural={NATURAL}", flush=True)
    while True:
        pending = [c for c in CELLS if not is_done(c)]
        if not pending:
            break
        actionable = [c for c in pending if is_ready(c) and not is_running(c)
                      and attempts[c["name"]] < MAX_RETRY + 1]
        if not actionable and not any(is_running(c) for c in pending):
            print("[bws] nothing actionable/running -> stop", flush=True)
            break
        now = time.time()
        for c in actionable:
            if is_running(c) or is_done(c):
                continue
            gpu = None
            for i in sorted(range(ng), key=gpu_mem):
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
            print(f"[bws] gpu{gpu} <- {c['name']} (budget {c['budget']}) pid={pid}", flush=True)
        time.sleep(POLL)

    summary = [{"name": c["name"], "task": c["task"], "budget": c["budget"],
                "done": is_done(c), "dir": str(run_dir(c))} for c in CELLS]
    (OUTROOT / "dispatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[bws] DONE -> {OUTROOT/'dispatch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
