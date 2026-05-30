#!/usr/bin/env bash
# E3.2 — cost-budget Pareto sweep (P4). Trains the continuous-cost Lagrangian
# policy on saucepan_to_hob/coworker_train for COST_BUDGET (d) x SEED,
# warm-started from the P1 stage-1 snapshot. Identifies the headline d (the
# Pareto knee) that becomes the P5 row-3 operating point. Mirrors
# run_e3_1_cost_signal.sh (support levers, workspace shaping, :-tags, SMOKE).
#
# NOT launched from inside the agent — a human runs this on the GPU box.
#
# Only the `continuous` cost form is swept here (E3.1 already compares forms);
# the treatment variable is the constraint target d = agent.cost_budget. Each
# run logs episode_lambda + episode_cost_integral (train_cqn_as default) so the
# PID-lambda trajectory and rolling cost-mean vs d can be plotted.
#
# Prereqs (GPU box):
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
#   cd safety_bigym && source venv/bin/activate
#   # Run the smoke gates first (docs/PROJECT_PLAN.md "Smoke-Gate Checklist").
#
# Usage:
#   WARMSTART=exp_local/cqn_as_base_curriculum/<run>/stage1_easy/snapshot_XXXXX.pt \
#     scripts/run_e3_2_cost_budget.sh                 # d in {0.001,0.01,0.05,0.1} x seeds {0,1,2}
#   SMOKE=1 scripts/run_e3_2_cost_budget.sh           # 1 cell, 2000 frames, no W&B
#   COST_BUDGETS="0.001 0.01 0.05 0.1" SEEDS="0 1 2" FRAMES=60000 WARMSTART=... \
#     scripts/run_e3_2_cost_budget.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HUMAN_MODEL="${HUMAN_MODEL:-g1}"
TASK="${TASK:-saucepan_to_hob}"
if [[ ! -f "${REPO_ROOT}/cfgs/env/safety_bigym/${TASK}.yaml" ]]; then
  echo "ERROR: TASK=${TASK} — no cfgs/env/safety_bigym/${TASK}.yaml" >&2
  exit 1
fi
case "${TASK}" in
  saucepan_to_hob)   NUM_DEMOS="${NUM_DEMOS:-36}" ;;
  drawers_open_all)  NUM_DEMOS="${NUM_DEMOS:-50}" ;;
  dishwasher_close)  NUM_DEMOS="${NUM_DEMOS:-50}" ;;
  *) NUM_DEMOS="${NUM_DEMOS:-36}"; echo "WARNING: TASK=${TASK} NUM_DEMOS=${NUM_DEMOS}" >&2 ;;
esac

# Optional P8 / E4.3 internalisation curve — set FILTER_PASSIVE to an SVF critic
# checkpoint to log eval/filter_intervention_rate per eval cycle (observe-only).
FILTER_PASSIVE="${FILTER_PASSIVE:-}"
FILTER_PASSIVE_R="${FILTER_PASSIVE_R:-4.0}"
PASSIVE_OVERRIDES=()
[[ -n "${FILTER_PASSIVE}" ]] && PASSIVE_OVERRIDES=(\
  "filter_passive.snapshot=${FILTER_PASSIVE}" "filter_passive.threshold=${FILTER_PASSIVE_R}")

if [[ "${SMOKE:-0}" == "1" ]]; then
  COST_BUDGETS="${COST_BUDGETS:-0.01}"
  SEEDS="${SEEDS:-0}"
  FRAMES="${FRAMES:-2000}"
  WANDB=(wandb.use=false)
  export CUDA_LAUNCH_BLOCKING=1
else
  COST_BUDGETS="${COST_BUDGETS:-0.001 0.01 0.05 0.1}"
  SEEDS="${SEEDS:-0 1 2}"
  FRAMES="${FRAMES:-60000}"
  WANDB=(wandb.use=true)
  : "${WARMSTART:?Set WARMSTART to the P1 stage-1 snapshot (.pt) to warm-start the cells}"
  [[ -f "${WARMSTART}" ]] || { echo "ERROR: WARMSTART=${WARMSTART} not found" >&2; exit 1; }
fi

_RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-e3_2_${TASK}_${_RUN_STAMP}}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/exp_local/e3_2_cost_budget/${RUN_TAG}}"
mkdir -p "${OUTDIR}"

# Shared overrides — identical base to E3.1 / run_base_curriculum.sh so d is the
# only treatment variable (continuous cost + workspace shaping + C51 support).
COMMON=(
  "env=safety_bigym/${TASK}"
  "env.human_model=${HUMAN_MODEL}"
  bodyslam=oracle
  disruption=coworker_train
  "num_demos=${NUM_DEMOS}"
  agent=cqn_as_lagrangian
  env.safety.add_workspace_penalty=true
  env.safety.workspace_beta=0.05
  env.safety.workspace_excess_cap=1.0
  agent.v_min=-6.0
  agent.v_max=2.0
  agent.atoms=101
  save_snapshot=true
  save_video=true
)

echo "== E3.2 cost-budget Pareto | TASK=${TASK} d='${COST_BUDGETS}' seeds='${SEEDS}' frames=${FRAMES} =="
echo "   OUTDIR=${OUTDIR}  warmstart=${WARMSTART:-<none/smoke>}"

LAUNCHED=()
for D in ${COST_BUDGETS}; do
  D_TAG="d${D//./p}"   # 0.01 -> d0p01 (dot-free dir/tag token)
  for SEED in ${SEEDS}; do
    NAME="${D_TAG}_seed${SEED}"
    STAGE_DIR="${OUTDIR}/${NAME}"
    echo "== launch ${NAME} (cost_budget=${D}) -> ${STAGE_DIR} =="
    EXTRA=()
    [[ -n "${WARMSTART:-}" ]] && EXTRA+=("+snapshot_path=${WARMSTART}")
    WB_TAGS="+wandb.tags=[e3_2,method:lagrangian_continuous,${D_TAG},task:${TASK},human:${HUMAN_MODEL},seed:${SEED}]"
    python train_cqn_as.py \
      "${COMMON[@]}" "${WANDB[@]}" \
      "agent.cost_budget=${D}" \
      "${PASSIVE_OVERRIDES[@]}" \
      seed="${SEED}" \
      num_train_frames="${FRAMES}" \
      "hydra.run.dir=${STAGE_DIR}" \
      "wandb.name=${RUN_TAG}_${NAME}" \
      "${WB_TAGS}" \
      "${EXTRA[@]}"
    LAUNCHED+=("${NAME}")
  done
done

echo ""
echo "== E3.2 done. launched: ${LAUNCHED[*]:-none} =="
echo "   Identify the knee (success_rate vs ep_proximity_violation_rate); that d -> P5 row 3."
