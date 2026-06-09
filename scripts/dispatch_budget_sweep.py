#!/usr/bin/env python3
"""GPU-pool dispatcher for the Tier-1 cost-budget RE-TARGET sweep.

Why: the 0.1/0.3/0.5 safety sweep found λ→0 at budgets 0.3/0.5 (cost already
under budget → constraint inert → deploy proximity == baseline), and λ→21.5 /
0% success at 0.1 (binding but task-fatal). The unconstrained policy's natural
rolling cost is ~0.25 (dish) / ~0.22 (drawers); the λ=21.5 floor bottomed at
~0.21 with the task destroyed. So the binding-but-feasible window — if it exists
— is ~0.21–0.25, which the original grid skipped entirely.

This sweeps budgets {0.16, 0.19, 0.22, 0.24} on both tasks (seed 0) to map the
knee: does any budget BIND (final λ>0, rolling-cost tracks budget) AND preserve
task success AND drop deploy proximity, or is the cost/success frontier a hard
cliff (every binding budget collapses → go straight to the SVF filter)?

Same recipe as dispatch_safety.py (adaptive-λ cqn_as_lagrangian, workspace OFF,
widened critic support, warm from curriculum stage-2), only cost_budget varies.

  nohup venv/bin/python scripts/dispatch_budget_sweep.py > logs/budget_sweep/dispatch.log 2>&1 &
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
CUR = REPO / "exp_local" / "curriculum"
OUTROOT = REPO / "exp_local" / "budget_sweep"
LOGDIR = REPO / "logs" / "budget_sweep"
LOGDIR.mkdir(parents=True, exist_ok=True)

DISH_WARM = "dish_rung1_a"
DRAW_WARM = "drawers_rung1_a"
BUDGETS = [0.16, 0.19, 0.22, 0.24]


def _cells():
    out = []
    for b in BUDGETS:
        tag = str(b).replace("0.", "b").replace(".", "")  # 0.16 -> b16
        out.append(dict(name=f"dish_{tag}", task="dishwasher_close", demos=69,
                        budget=b, warm=DISH_WARM, seed=0))
        out.append(dict(name=f"drawers_{tag}", task="drawers_open_all", demos=54,
                        budget=b, warm=DRAW_WARM, seed=0))
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


FREE_MEM_MIB = 1500  # a GPU is usable if <1.5GB used: blocks my own ~4GB jobs but
# tolerates another user's lightweight CUDA context (e.g. xd1125's ~0.5GB eval
# contexts on 24GB cards) — co-locating there is zero OOM risk, only mild contention.


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
        "agent.v_min=-6.0", "agent.v_max=2.0", "agent.atoms=101",
        "env.safety.add_workspace_penalty=false",
        "disruption=coworker_train", f"num_train_frames={FRAMES}",
        "eval_every_frames=2500", "num_eval_episodes=10",
        f"seed={c['seed']}", "save_snapshot=true", "save_video=false",
        "wandb.use=true", "wandb.project=safety-critic", f"wandb.name=budget_{c['name']}",
        f"+wandb.tags=[budget_sweep,adaptive_lambda,budget:{c['budget']},task:{c['task']}]",
        f"+snapshot_path={warm(c['warm'])}",
        f"hydra.run.dir={run_dir(c)}",
    ]


def launch(c, gpu):
    run_dir(c).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AMASS_DATA_DIR=AMASS,
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    log = open(LOGDIR / f"{c['name']}.log", "a")
    log.write(f"\n==== launch {c['name']} (budget {c['budget']}) on GPU{gpu} ====\n")
    log.flush()
    p = subprocess.Popen(build_cmd(c), env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=REPO)
    return p.pid


def main():
    attempts = {c["name"]: 0 for c in CELLS}
    cooldown = {}
    ng = num_gpus()
    print(f"[budget] {len(CELLS)} cells across {ng} GPUs; budgets={BUDGETS}", flush=True)
    while True:
        pending = [c for c in CELLS if not is_done(c)]
        if not pending:
            break
        actionable = [c for c in pending if is_ready(c) and not is_running(c)
                      and attempts[c["name"]] < MAX_RETRY + 1]
        if not actionable and not any(is_running(c) for c in pending):
            print("[budget] nothing actionable/running -> stop", flush=True)
            break
        now = time.time()
        for c in actionable:
            if is_running(c) or is_done(c):
                continue
            gpu = None
            # Prefer the least-loaded GPU so we fill idle-context cards before any
            # a co-tenant is actively computing on; skip recently-grabbed ones.
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
            print(f"[budget] gpu{gpu} <- {c['name']} (budget {c['budget']}) pid={pid}", flush=True)
        time.sleep(POLL)

    summary = []
    for c in CELLS:
        fm = run_dir(c) / "final_metrics.json"
        rec = {"name": c["name"], "task": c["task"], "budget": c["budget"],
               "done": fm.exists(), "dir": str(run_dir(c))}
        summary.append(rec)
    (OUTROOT / "dispatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[budget] DONE -> {OUTROOT/'dispatch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
