#!/usr/bin/env bash
# Fixed-lambda Lagrangian — the stability fix for the d=0.3 PID instability.
#
# WHY: the 3-seed CONFIRM showed the PID-Lagrangian at d=0.3 is seed-unstable —
# lambda landed at 0 (seed1, unconstrained), 0.267 (seed0, graceful basin), and
# 3.855 (seed2, windup collapse) because d=0.3 sits at the task's inherent cost,
# so the dual variable is determined by each seed's stochastic cost trajectory.
# Only seed0's lambda~=0.27 produced the graceful ~21% proximity basin (verified
# constraint-driven vs the seed1 lambda=0 control). This launcher FREEZES lambda
# at the known-good value (zero PID gains -> LagrangianPID.update returns
# lambda_init unchanged; Q_c still trains; dual_select uses the fixed lambda), so
# every seed gets the same constraint weight. Tests whether the graceful regime
# reproduces robustly across seeds -> the real ROW3, and decouples lambda from
# seed (definitive constraint-causality).
#
# NOT launched from inside the agent — a human runs this on the GPU box. Shard the
# seeds across GPUs (one process per seed) for ~6-8 h wall-clock:
#   for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s LAMBDAS=0.27 SEEDS=$s \
#     WARMSTART=<stage1.pt> OUTDIR=exp_local/fixed_lambda RUN_TAG=fixlam0p27 \
#     nohup bash scripts/run_fixed_lambda.sh > logs/fixlam_seed$s.log 2>&1 & done
#
# Usage:
#   LAMBDAS="0.27" SEEDS="0 1 2" WARMSTART=<stage1.pt> bash scripts/run_fixed_lambda.sh
#   LAMBDAS="0.2 0.27 0.35" SEEDS="0 1 2" ...   # small lambda bracket if 0.27 over/under-shoots
#   SMOKE=1 bash scripts/run_fixed_lambda.sh    # 1 cell, 2000 frames, no W&B
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_ROOT}"

HUMAN_MODEL="${HUMAN_MODEL:-g1}"; TASK="${TASK:-saucepan_to_hob}"
NUM_DEMOS="${NUM_DEMOS:-36}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  LAMBDAS="${LAMBDAS:-0.27}"; SEEDS="${SEEDS:-0}"; FRAMES="${FRAMES:-2000}"
  WANDB=(wandb.use=false); export CUDA_LAUNCH_BLOCKING=1
else
  LAMBDAS="${LAMBDAS:-0.27}"; SEEDS="${SEEDS:-0 1 2}"
  FRAMES="${FRAMES:-40000}"   # the CONFIRM budget; the seed-0 basin formed within ~40k
  WANDB=(wandb.use=true)
  : "${WARMSTART:?Set WARMSTART to the P1 stage-1 snapshot (.pt)}"
  [[ -f "${WARMSTART}" ]] || { echo "ERROR: WARMSTART=${WARMSTART} not found" >&2; exit 1; }
fi

_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-fixed_lambda_${TASK}_${_STAMP}}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/exp_local/fixed_lambda/${RUN_TAG}}"; mkdir -p "${OUTDIR}"

# Identical base to run_e3_2_cost_budget.sh so lambda is the only treatment change,
# EXCEPT the PID is frozen (zero gains) -> lambda is pinned at lambda_init.
COMMON=(
  "env=safety_bigym/${TASK}" "env.human_model=${HUMAN_MODEL}"
  bodyslam=oracle disruption=coworker_train "num_demos=${NUM_DEMOS}"
  agent=cqn_as_lagrangian
  agent.lambda_k_i=0 agent.lambda_k_p=0 agent.lambda_k_d=0   # freeze the PID
  agent.cost_budget=0.0                                       # unused once PID frozen
  env.safety.add_workspace_penalty=true env.safety.workspace_beta=0.05 env.safety.workspace_excess_cap=1.0
  agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101
  save_snapshot=true save_video=true
)

echo "== fixed-lambda | TASK=${TASK} lambdas='${LAMBDAS}' seeds='${SEEDS}' frames=${FRAMES} =="
echo "   OUTDIR=${OUTDIR}  warmstart=${WARMSTART:-<smoke>}"

for L in ${LAMBDAS}; do
  L_TAG="lam${L//./p}"
  for SEED in ${SEEDS}; do
    NAME="${L_TAG}_seed${SEED}"; STAGE_DIR="${OUTDIR}/${NAME}"
    echo "== launch ${NAME} (lambda=${L}, PID frozen) -> ${STAGE_DIR} =="
    EXTRA=(); [[ -n "${WARMSTART:-}" ]] && EXTRA+=("+snapshot_path=${WARMSTART}")
    python train_cqn_as.py \
      "${COMMON[@]}" "${WANDB[@]}" \
      "agent.lambda_init=${L}" \
      seed="${SEED}" num_train_frames="${FRAMES}" \
      "hydra.run.dir=${STAGE_DIR}" \
      "wandb.name=${RUN_TAG}_${NAME}" \
      "+wandb.tags=[fixed_lambda,method:lagrangian_fixed,${L_TAG},task:${TASK},human:${HUMAN_MODEL},seed:${SEED}]" \
      "${EXTRA[@]}"
  done
done
echo "== fixed-lambda done. Sweep each seed's basin (run_basin_sweep.sh) -> pick -> pool (analyze_row3.py). =="
