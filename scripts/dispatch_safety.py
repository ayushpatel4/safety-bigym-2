#!/usr/bin/env python3
"""GPU-pool dispatcher for the constrained-RL safety phase (adaptive-lambda).

Fine-tunes each stage-2 curriculum policy with the Lagrangian cost critic +
ADAPTIVE-lambda PID at a FEASIBLE cost_budget (the e3_2-validated recipe that
preserved task success on saucepan) — the fix for the original collapse, which
used frozen lambda=0.1 / budget=0 from collapsed warm-starts. Sweeps cost_budget
per task to trace the safety/task tradeoff.

Cells are readiness-gated: a cell only launches once its warm-start snapshot
exists (so drawers cells wait for their curriculum to finish). Pooled across
free GPUs, retried once, idempotent (skips cells with final_metrics.json).

  nohup venv/bin/python scripts/dispatch_safety.py > logs/safety/dispatch.log 2>&1 &
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
OUTROOT = REPO / "exp_local" / "safety"
LOGDIR = REPO / "logs" / "safety"
LOGDIR.mkdir(parents=True, exist_ok=True)


def warm(cell):  # stage-2 curriculum snapshot to fine-tune from (absolute)
    return CUR / cell / "stage2_full" / "snapshot_best.pt"


# name, task, demos, budget, workspace(bool), warmstart-cell, seed
DISH_WARM = "dish_rung1_a"        # stage2 peak 1.00
DRAW_WARM = "drawers_rung1_a"     # stage2 peak ~0.90 (best drawers under full coworker)
CELLS = [
    dict(name="dish_b01", task="dishwasher_close", demos=69, budget=0.1, ws=False, warm=DISH_WARM, seed=0),
    dict(name="dish_b03", task="dishwasher_close", demos=69, budget=0.3, ws=False, warm=DISH_WARM, seed=0),
    dict(name="dish_b05", task="dishwasher_close", demos=69, budget=0.5, ws=False, warm=DISH_WARM, seed=0),
    dict(name="drawers_b01", task="drawers_open_all", demos=54, budget=0.1, ws=False, warm=DRAW_WARM, seed=0),
    dict(name="drawers_b03", task="drawers_open_all", demos=54, budget=0.3, ws=False, warm=DRAW_WARM, seed=0),
    dict(name="drawers_b05", task="drawers_open_all", demos=54, budget=0.5, ws=False, warm=DRAW_WARM, seed=0),
    # 3-seed ROW3 robustness for the deployable budgets (0.3, 0.5); 0.1 stays 1 seed.
    dict(name="dish_b03_s1", task="dishwasher_close", demos=69, budget=0.3, ws=False, warm=DISH_WARM, seed=1),
    dict(name="dish_b03_s2", task="dishwasher_close", demos=69, budget=0.3, ws=False, warm=DISH_WARM, seed=2),
    dict(name="dish_b05_s1", task="dishwasher_close", demos=69, budget=0.5, ws=False, warm=DISH_WARM, seed=1),
    dict(name="dish_b05_s2", task="dishwasher_close", demos=69, budget=0.5, ws=False, warm=DISH_WARM, seed=2),
    dict(name="drawers_b03_s1", task="drawers_open_all", demos=54, budget=0.3, ws=False, warm=DRAW_WARM, seed=1),
    dict(name="drawers_b03_s2", task="drawers_open_all", demos=54, budget=0.3, ws=False, warm=DRAW_WARM, seed=2),
    dict(name="drawers_b05_s1", task="drawers_open_all", demos=54, budget=0.5, ws=False, warm=DRAW_WARM, seed=1),
    dict(name="drawers_b05_s2", task="drawers_open_all", demos=54, budget=0.5, ws=False, warm=DRAW_WARM, seed=2),
]


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


def gpu_free(i):
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(i)],
                       capture_output=True, text=True)
    return r.stdout.strip() == ""


def build_cmd(c):
    ws = [f"env.safety.add_workspace_penalty={'true' if c['ws'] else 'false'}"]
    if c["ws"]:
        ws += ["env.safety.workspace_beta=0.05", "env.safety.workspace_excess_cap=1.0"]
    return [
        "venv/bin/python", "train_cqn_as.py",
        f"env=safety_bigym/{c['task']}", "env.human_model=g1", "env.smplh_motion=amass",
        "bodyslam=oracle", f"num_demos={c['demos']}",
        "agent=cqn_as_lagrangian", f"agent.cost_budget={c['budget']}",
        "agent.v_min=-6.0", "agent.v_max=2.0", "agent.atoms=101",
        *ws,
        "disruption=coworker_train", f"num_train_frames={FRAMES}",
        "eval_every_frames=2500", "num_eval_episodes=10",
        f"seed={c['seed']}", "save_snapshot=true", "save_video=true",
        "wandb.use=true", "wandb.project=safety-critic", f"wandb.name=safety_{c['name']}",
        f"+wandb.tags=[safety,adaptive_lambda,budget:{c['budget']},task:{c['task']}]",
        f"+snapshot_path={warm(c['warm'])}",
        f"hydra.run.dir={run_dir(c)}",
    ]


def launch(c, gpu):
    run_dir(c).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AMASS_DATA_DIR=AMASS,
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    log = open(LOGDIR / f"{c['name']}.log", "a")
    log.write(f"\n==== launch {c['name']} on GPU{gpu} (warm={warm(c['warm'])}) ====\n")
    log.flush()
    p = subprocess.Popen(build_cmd(c), env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=REPO)
    return p.pid


def main():
    attempts = {c["name"]: 0 for c in CELLS}
    cooldown = {}
    ng = num_gpus()
    print(f"[safety] {len(CELLS)} cells across {ng} GPUs", flush=True)
    while True:
        pending = [c for c in CELLS if not is_done(c)]
        if not pending:
            break
        actionable = [c for c in pending if is_ready(c) and not is_running(c)
                      and attempts[c["name"]] < MAX_RETRY + 1]
        waiting = [c for c in pending if not is_ready(c)]
        if not actionable and not any(is_running(c) for c in pending) and not waiting:
            print("[safety] nothing actionable/running/waiting -> stop", flush=True)
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
            attempts[c["name"]] += 1
            pid = launch(c, gpu)
            cooldown[gpu] = time.time()
            print(f"[safety] gpu{gpu} <- {c['name']} (budget {c['budget']}) pid={pid}", flush=True)
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
    print(f"[safety] DONE -> {OUTROOT/'dispatch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
