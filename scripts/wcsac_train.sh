#!/usr/bin/env bash
# WCSAC (E3.7 / P9) external distributional safe-RL baseline -- FULL TRAINING.
#
#  *** HUMAN-RUN, MULTI-HOUR. Do NOT launch this from Claude Code. ***
#
# Trains the WCSAC agent (faithful Worst-Case SAC: stochastic actor, twin
# reward critics, Gaussian safety critic, CVaR_alpha constraint, learnable
# lambda + entropy temperature) on the same two Phase-3 tasks the Lagrangian
# fixed-budget runs used (dishwasher_close, drawers_open_all), with the same
# coworker disruption, so it lands in the same comparison table.
#
# KEY DIFFERENCES vs the CQN-AS Lagrangian dispatch (scripts/dispatch_safety.py),
# and why -- read before interpreting results:
#
#   1. NO WARM-START. The Lagrangian fine-tunes (FRAMES=40000) from a pretrained
#      CQN-AS stage-2 reward critic. WCSAC has a different architecture (SAC
#      actor-critic, not the C2F value critic), so it cannot load those
#      snapshots -- it trains FROM SCRATCH. From-scratch pixel SAC needs a far
#      larger budget (default 500k frames here) and may still underperform the
#      task. That is the documented honest-failure path (report S:disc:wcsac-honest):
#      a faithful reimplementation that we report straight even if it can't match
#      the warm-started value-based method.
#
#   2. cost_budget is the CVaR ceiling on the discounted cost RETURN, NOT the
#      per-step rolling-mean cost the Lagrangian sweeps ({0.1,0.3,0.5}). With
#      per-step c_t in [0,1] and gamma=0.99, the per-step budget maps to a return
#      budget ~100x larger. The right value depends on the realised cost scale,
#      so we sweep a grid. CALIBRATION: run one config, read qc_mean / cvar off
#      W&B once the safety critic settles, and set cost_budget below that so the
#      constraint actually binds (lambda > 0) without collapsing the policy.
#
#   3. DEMOS default 0 (faithful pure-online WCSAC). Set DEMOS>0 to pre-fill the
#      replay with demos as a mitigation lever (a "WCSAC + demo warm replay"
#      ablation); the Lagrangian used 69 (dish) / 54 (drawers).
#
# Override any of these via env vars, e.g.:
#   FRAMES=300000 BUDGETS="15 30" SEED=1 DEMOS=69 scripts/wcsac_train.sh dishwasher_close
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (safety_bigym)

export AMASS_DATA_DIR="${AMASS_DATA_DIR:-/home/ap2322/Documents/CMU/CMU}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PY=venv/bin/python
FRAMES="${FRAMES:-500000}"          # from-scratch SAC budget (no warm-start)
SEED="${SEED:-0}"
DEMOS="${DEMOS:-0}"                 # 0 = faithful online WCSAC; >0 = demo-warm replay
EVAL_EVERY="${EVAL_EVERY:-10000}"
NUM_EVAL="${NUM_EVAL:-10}"
BUDGETS="${BUDGETS:-5 15 30}"       # CVaR-of-cost-return ceilings to sweep
TASKS="${*:-dishwasher_close drawers_open_all}"

mkdir -p logs exp_local/wcsac

run() {
  local task="$1" budget="$2"
  local name="wcsac_${task}_b${budget}_s${SEED}"
  local dir="exp_local/wcsac/${name}"
  echo "==================================================================="
  echo "  $name   FRAMES=$FRAMES DEMOS=$DEMOS  ($(date))"
  echo "==================================================================="
  $PY train_cqn_as.py \
    env=safety_bigym/${task} env.human_model=g1 env.smplh_motion=amass \
    agent=wcsac agent.cost_budget=${budget} \
    bodyslam=oracle disruption=coworker_train num_demos=${DEMOS} \
    num_train_frames=${FRAMES} eval_every_frames=${EVAL_EVERY} num_eval_episodes=${NUM_EVAL} \
    seed=${SEED} save_snapshot=true save_video=true \
    wandb.use=true wandb.project=safety-critic wandb.name=${name} \
    "+wandb.tags=[phase-3,wcsac,E3.7,external_baseline,task:${task},cost_budget:${budget}]" \
    hydra.run.dir=${dir} \
    2>&1 | tee -a "logs/${name}.log"
}

for task in $TASKS; do
  for b in $BUDGETS; do
    run "$task" "$b"
  done
done

cat <<'EOF'

All WCSAC training runs finished.

EVALUATE a peak snapshot (pick the peak-by-curve step from W&B
pretrain/eval -- same protocol as the Lagrangian baselines). The CQN-AS
eval-only path is agent-agnostic (loads cfg.agent + load_state_dict + eval):

  AMASS_DATA_DIR=/home/ap2322/Documents/CMU/CMU MUJOCO_GL=egl \
  venv/bin/python train_cqn_as.py \
    env=safety_bigym/dishwasher_close env.human_model=g1 agent=wcsac \
    +snapshot_path=exp_local/wcsac/<run>/snapshot_<step>.pt \
    num_train_frames=0 num_demos=0 num_eval_episodes=50 \
    bodyslam=oracle disruption=coworker_train wandb.use=false

The eval emits the standard ep_* safety schema, so the existing tail-risk
aggregation (cvar95_ep_cost_integral, cvar95_ep_min_separation, ...) applies
unchanged -- WCSAC drops straight into the Lagrangian comparison table.
EOF
