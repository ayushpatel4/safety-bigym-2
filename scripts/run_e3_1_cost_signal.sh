#!/usr/bin/env bash
# E3.1 — cost-signal form ablation (P3). Trains the Phase-3 constrained policy
# on saucepan_to_hob/coworker_train for COST_FORM x SEED, warm-started from the
# P1 stage-1 snapshot so each cell shares row-1's training protocol (60k frames
# on coworker_train from the same stage-1 start) and differs ONLY in the cost
# signal. Mirrors run_base_curriculum.sh's invocation (support levers, workspace
# shaping, :-separated W&B tags, SMOKE path).
#
# NOT launched from inside the agent — a human runs this on the GPU box.
#
# === IMPORTANT: only the `continuous` cell is runnable today ===
# env_adapter.py hardcodes the continuous cost c_t = compute_cost(info["safety"]).
# The `binary` (c_t = 1[ssm_violation]) and `fixed` (reward penalty, no Lagrangian)
# cells have NO code path yet — they need the cost-form selector described in
# docs/PROJECT_PLAN.md (Phase 3 / P3). This script LAUNCHES `continuous` and
# LOUDLY SKIPS the unwired forms (it never silently runs them as continuous).
# Until the selector lands, `continuous` alone is still useful: it is the P5
# row-3 input.
#
# Prereqs (GPU box):
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
#   cd safety_bigym && source venv/bin/activate
#   # Run the smoke gates first (docs/PROJECT_PLAN.md "Smoke-Gate Checklist").
#
# Usage:
#   WARMSTART=exp_local/cqn_as_base_curriculum/<run>/stage1_easy/snapshot_XXXXX.pt \
#     scripts/run_e3_1_cost_signal.sh                 # continuous x seeds {0,1,2}
#   SMOKE=1 scripts/run_e3_1_cost_signal.sh           # 1 cell, 2000 frames, no W&B
#   COST_FORMS="fixed binary continuous" WARMSTART=... scripts/run_e3_1_cost_signal.sh
#       # launches continuous; reports fixed+binary as BLOCKED (selector not wired)
#   COST_BUDGET=0.01 SEEDS="0 1 2" FRAMES=60000 WARMSTART=... scripts/run_e3_1_cost_signal.sh
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

# Cost-signal forms wired in code today. Extend WIRED_FORMS when the selector lands.
WIRED_FORMS=" continuous "
COST_BUDGET="${COST_BUDGET:-0.01}"   # E3.1 fixes d; P4 (run separately) sweeps it.

if [[ "${SMOKE:-0}" == "1" ]]; then
  COST_FORMS="${COST_FORMS:-continuous}"
  SEEDS="${SEEDS:-0}"
  FRAMES="${FRAMES:-2000}"
  WANDB=(wandb.use=false)
  export CUDA_LAUNCH_BLOCKING=1
else
  COST_FORMS="${COST_FORMS:-continuous}"
  SEEDS="${SEEDS:-0 1 2}"
  FRAMES="${FRAMES:-60000}"
  WANDB=(wandb.use=true)
  : "${WARMSTART:?Set WARMSTART to the P1 stage-1 snapshot (.pt) to warm-start row-3 cells}"
  [[ -f "${WARMSTART}" ]] || { echo "ERROR: WARMSTART=${WARMSTART} not found" >&2; exit 1; }
fi

_RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-e3_1_${TASK}_${_RUN_STAMP}}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/exp_local/e3_1_cost_signal/${RUN_TAG}}"
mkdir -p "${OUTDIR}"

# Shared overrides — identical base to run_base_curriculum.sh so the cost-signal
# form is the only treatment variable (workspace shaping + widened C51 support).
COMMON=(
  "env=safety_bigym/${TASK}"
  "env.human_model=${HUMAN_MODEL}"
  bodyslam=oracle
  disruption=coworker_train
  "num_demos=${NUM_DEMOS}"
  env.safety.add_workspace_penalty=true
  env.safety.workspace_beta=0.05
  env.safety.workspace_excess_cap=1.0
  agent.v_min=-6.0
  agent.v_max=2.0
  agent.atoms=101
  save_snapshot=true
  save_video=true
)

echo "== E3.1 cost-signal ablation | TASK=${TASK} forms='${COST_FORMS}' seeds='${SEEDS}' frames=${FRAMES} =="
echo "   OUTDIR=${OUTDIR}  d(cost_budget)=${COST_BUDGET}  warmstart=${WARMSTART:-<none/smoke>}"

LAUNCHED=(); BLOCKED=()
for FORM in ${COST_FORMS}; do
  if [[ "${WIRED_FORMS}" != *" ${FORM} "* ]]; then
    echo ""
    echo "############################################################"
    echo "# BLOCKED: cost_signal='${FORM}' is NOT wired in code yet."
    echo "#   binary  -> env_adapter must emit c_t = 1[ssm_violation]"
    echo "#   fixed   -> reward penalty r - 0.05*1[violation], agent=cqn_as (no lambda)"
    echo "# Implement the cost-form selector (docs/PROJECT_PLAN.md P3) first."
    echo "# Skipping so this run never produces silent-wrong (continuous) data."
    echo "############################################################"
    BLOCKED+=("${FORM}")
    continue
  fi
  for SEED in ${SEEDS}; do
    NAME="${FORM}_seed${SEED}"
    STAGE_DIR="${OUTDIR}/${NAME}"
    echo "== launch ${NAME} -> ${STAGE_DIR} =="
    # continuous + Lagrangian (agent=cqn_as_lagrangian). cost flows via env_adapter.
    AGENT_OVERRIDES=(agent=cqn_as_lagrangian "agent.cost_budget=${COST_BUDGET}")
    METHOD_TAG="lagrangian_${FORM}"
    EXTRA=()
    [[ -n "${WARMSTART:-}" ]] && EXTRA+=("+snapshot_path=${WARMSTART}")
    WB_TAGS="+wandb.tags=[e3_1,method:${METHOD_TAG},cost:${FORM},task:${TASK},human:${HUMAN_MODEL},seed:${SEED}]"
    python train_cqn_as.py \
      "${COMMON[@]}" "${WANDB[@]}" \
      "${AGENT_OVERRIDES[@]}" \
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
echo "== E3.1 done. launched: ${LAUNCHED[*]:-none} =="
if [[ ${#BLOCKED[@]} -gt 0 ]]; then
  echo "== BLOCKED (cost-form selector not wired): ${BLOCKED[*]} =="
  echo "   E3.1 is INCOMPLETE until those cells run. See docs/PROJECT_PLAN.md P3."
  exit 3
fi
