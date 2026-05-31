#!/usr/bin/env bash
# E3.1 — cost-signal form ablation (P3). Trains the Phase-3 policy on
# saucepan_to_hob/coworker_train for COST_FORM x SEED, warm-started from the P1
# stage-1 snapshot so every cell shares row-1's training protocol (60k frames on
# coworker_train from the same stage-1 start) and differs ONLY in the cost
# signal. Mirrors run_base_curriculum.sh (support levers, workspace shaping,
# :-separated W&B tags, SMOKE path).
#
# NOT launched from inside the agent — a human runs this on the GPU box.
#
# The three cells (all wired as of 2026-05-30):
#   continuous (ours) : agent=cqn_as_lagrangian, c_t = compute_cost (graded [0,1])
#   binary            : agent=cqn_as_lagrangian + env.safety.cost_form=binary
#                       (c_t = 1[ssm_violation]) — strips gradient richness
#   fixed             : agent=cqn_as (NO Lagrangian) + env.safety.add_violation_penalty
#                       (reward r - 0.05*1[violation]) — fixed-magnitude baseline
#
# Prereqs (GPU box):
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
#   cd safety_bigym && source venv/bin/activate
#   # Run the smoke gates first (docs/PROJECT_PLAN.md "Smoke-Gate Checklist").
#
# Usage:
#   WARMSTART=exp_local/cqn_as_base_curriculum/<run>/stage1_easy/snapshot_XXXXX.pt \
#     scripts/run_e3_1_cost_signal.sh                 # all 3 forms x seeds {0,1,2}
#   SMOKE=1 scripts/run_e3_1_cost_signal.sh           # 1 cell, 2000 frames, no W&B
#   COST_FORMS="continuous" WARMSTART=... scripts/run_e3_1_cost_signal.sh   # one form
#   COST_BUDGET=0.01 FIXED_PENALTY=0.05 SEEDS="0 1 2" FRAMES=60000 WARMSTART=... \
#     scripts/run_e3_1_cost_signal.sh
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

KNOWN_FORMS=" fixed binary continuous "
COST_BUDGET="${COST_BUDGET:-0.01}"      # E3.1 fixes d; P4 (run separately) sweeps it.
FIXED_PENALTY="${FIXED_PENALTY:-0.05}"  # reward-penalty magnitude for the `fixed` cell.
# Optional in-training passive-filter logging — set FILTER_PASSIVE to an SVF critic.
# CAVEAT: this trains on oracle, where the SVF Q collapses (100% would-be veto), so
# the logged curve is flat/meaningless. For the real E4.3 internalisation curve use
# scripts/run_e4_3_internalisation.sh (post-hoc, on noisy). Left here, off by default.
FILTER_PASSIVE="${FILTER_PASSIVE:-}"
# Default R from snapshots.py::SVF_FILTER_THRESHOLD_R (standalone load, no torch);
# R=2.25 for saucepan (dense-0.3m-sweep operating point). Override FILTER_PASSIVE_R=.
FILTER_PASSIVE_R="${FILTER_PASSIVE_R:-$(python -c "import importlib.util as u;sp=u.spec_from_file_location('s','safety_bigym/filters/snapshots.py');m=u.module_from_spec(sp);sp.loader.exec_module(m);print(m.SVF_FILTER_THRESHOLD_R.get('${TASK}',2.25))" 2>/dev/null || echo 2.25)}"
PASSIVE_OVERRIDES=()
[[ -n "${FILTER_PASSIVE}" ]] && PASSIVE_OVERRIDES=(\
  "filter_passive.snapshot=${FILTER_PASSIVE}" "filter_passive.threshold=${FILTER_PASSIVE_R}")

if [[ "${SMOKE:-0}" == "1" ]]; then
  COST_FORMS="${COST_FORMS:-continuous}"
  SEEDS="${SEEDS:-0}"
  FRAMES="${FRAMES:-2000}"
  WANDB=(wandb.use=false)
  export CUDA_LAUNCH_BLOCKING=1
else
  COST_FORMS="${COST_FORMS:-fixed binary continuous}"
  SEEDS="${SEEDS:-0 1 2}"
  FRAMES="${FRAMES:-60000}"
  WANDB=(wandb.use=true)
  : "${WARMSTART:?Set WARMSTART to the P1 stage-1 snapshot (.pt) to warm-start the cells}"
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
echo "   OUTDIR=${OUTDIR}  d(cost_budget)=${COST_BUDGET}  fixed_penalty=${FIXED_PENALTY}  warmstart=${WARMSTART:-<none/smoke>}"

LAUNCHED=(); SKIPPED=()
for FORM in ${COST_FORMS}; do
  if [[ "${KNOWN_FORMS}" != *" ${FORM} "* ]]; then
    echo "ERROR: unknown cost form '${FORM}' (expected: fixed | binary | continuous)" >&2
    SKIPPED+=("${FORM}")
    continue
  fi
  # Per-form overrides — the ONLY difference between cells.
  case "${FORM}" in
    continuous)
      FORM_OVERRIDES=(agent=cqn_as_lagrangian "agent.cost_budget=${COST_BUDGET}")
      METHOD_TAG="lagrangian_continuous" ;;
    binary)
      FORM_OVERRIDES=(agent=cqn_as_lagrangian "agent.cost_budget=${COST_BUDGET}" \
        env.safety.cost_form=binary)
      METHOD_TAG="lagrangian_binary" ;;
    fixed)
      # Plain agent (no Q_c / no lambda); safety enters via the env reward penalty.
      FORM_OVERRIDES=(agent=cqn_as env.safety.add_violation_penalty=true \
        "env.safety.violation_penalty=${FIXED_PENALTY}")
      METHOD_TAG="fixed_penalty" ;;
  esac
  for SEED in ${SEEDS}; do
    NAME="${FORM}_seed${SEED}"
    STAGE_DIR="${OUTDIR}/${NAME}"
    echo "== launch ${NAME} -> ${STAGE_DIR} =="
    EXTRA=()
    [[ -n "${WARMSTART:-}" ]] && EXTRA+=("+snapshot_path=${WARMSTART}")
    WB_TAGS="+wandb.tags=[e3_1,method:${METHOD_TAG},cost:${FORM},task:${TASK},human:${HUMAN_MODEL},seed:${SEED}]"
    python train_cqn_as.py \
      "${COMMON[@]}" "${WANDB[@]}" \
      "${FORM_OVERRIDES[@]}" \
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
echo "== E3.1 done. launched: ${LAUNCHED[*]:-none} =="
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo "== SKIPPED unknown forms: ${SKIPPED[*]} ==" >&2
  exit 3
fi
