#!/usr/bin/env bash
# E4.1 — headline feature-incremental eval driver (P5). Runs benchmark_policy.py
# for each table row (PURE EVAL, no training) and writes one CSV per row. Rows
# whose snapshot env var is unset are SKIPPED, so this is usable incrementally:
# run rows 1 & 4 now (need only STAGE2 + the SVF filter), then rows 2/3/5 once
# those snapshots exist (P5 row-2 training, P3 continuous + d_knee).
#
# Perception (PROJECT_PLAN "Perception Mode Policy"): rows 1-5 eval `oracle`;
# row5_hybrid_noisy is the single `noisy` sim-to-real diagnostic.
#
# Rows:
#   row1_baseline        STAGE2,  oracle, no filter   (unconstrained baseline)
#   row2_workspace       ROW2,    oracle, no filter   (+ workspace shaping ablation)
#   row3_lagrangian      ROW3,    oracle, no filter   (+ continuous-cost Lagrangian)
#   row4_baseline_filter STAGE2,  oracle, + filter    (baseline + runtime SVF)
#   row5_hybrid          ROW3,    oracle, + filter    (full hybrid)
#   row5_hybrid_noisy    ROW3,    noisy,  + filter    (sim-to-real diagnostic)
#
# NOT a training job — pure eval, minimal GPU. Run on the GPU box.
#
# Prereqs:
#   cd ~/Documents/safety_bigym && source venv/bin/activate
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
#   export AMASS_DATA_DIR=...   # the harness replays BiGym demos (AMASS human-pos
#                               # injection) to derive CQN-AS action stats
#
# Inputs (env vars; absolute or repo-relative paths):
#   STAGE2      row-1 + row-4 policy (P1 unconstrained baseline). Default = recorded G1 stage-2.
#   ROW3        row-3 + row-5 policy (P3 continuous + d_knee). Unset -> rows 3,5,5_noisy skipped.
#   ROW2        row-2 policy (+ workspace-shaping training run). Unset -> row 2 skipped.
#   SVF_FILTER  SVF critic for rows 4/5. Default checkpoints/svf_coworker_train_g1_v1.pt.
#   FILTER_R    veto threshold R. Default 4.0 (snapshots.py provisional knee).
#   SEEDS (0,1,2), EPISODES (20), DISRUPTION (coworker_train),
#   NUM_DEMOS_FOR_STATS (0 = faithful full count; cap on a laptop), OUTDIR.
#
# Usage:
#   STAGE2=... bash scripts/run_e4_1_headline.sh              # rows 1 & 4 now
#   STAGE2=... ROW3=... bash scripts/run_e4_1_headline.sh     # rows 1,3,4,5,5_noisy
#   SMOKE=1 STAGE2=... bash scripts/run_e4_1_headline.sh      # 1 seed x 2 ep x 50 steps
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-saucepan_to_hob}"
HUMAN_MODEL="${HUMAN_MODEL:-g1}"
DISRUPTION="${DISRUPTION:-coworker_train}"
# Recorded P1 stage-2 (filters/snapshots.py::G1_CURRICULUM). Repo-relative so it
# resolves from REPO_ROOT; override STAGE2=<abs path> to use another baseline.
STAGE2="${STAGE2:-exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage2_full/snapshot_28203.pt}"
SVF_FILTER="${SVF_FILTER:-checkpoints/svf_coworker_train_g1_v1.pt}"
FILTER_R="${FILTER_R:-4.0}"
NUM_DEMOS_FOR_STATS="${NUM_DEMOS_FOR_STATS:-0}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  COUNT_ARGS=(--smoke)            # --smoke overrides seeds/episodes/steps
else
  SEEDS="${SEEDS:-0,1,2}"
  EPISODES="${EPISODES:-20}"
  COUNT_ARGS=(--seeds "${SEEDS}" --episodes "${EPISODES}")
fi

_RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-e4_1_${TASK}_${_RUN_STAMP}}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/results/e4_1/${RUN_TAG}}"
mkdir -p "${OUTDIR}"
echo "== E4.1 headline eval | TASK=${TASK} disruption=${DISRUPTION} R=${FILTER_R} =="
echo "   OUTDIR=${OUTDIR}"
echo "   STAGE2=${STAGE2}"
echo "   ROW3=${ROW3:-<unset -> rows 3/5/5_noisy skipped>}  ROW2=${ROW2:-<unset -> row 2 skipped>}"

RAN=(); SKIPPED=()
run_row() {   # label  snapshot  obs-mode  filter|nofilter
  local label="$1" snap="$2" obs="$3" mode="$4"
  if [[ -z "${snap}" ]]; then
    echo "-- skip ${label} (snapshot env var unset)"
    SKIPPED+=("${label}")
    return
  fi
  local filt=()
  if [[ "${mode}" == "filter" ]]; then
    [[ -f "${SVF_FILTER}" ]] || { echo "ERROR: ${label} needs SVF_FILTER=${SVF_FILTER} (not found)" >&2; SKIPPED+=("${label}"); return; }
    filt=(--filter-snapshot "${SVF_FILTER}" --filter-threshold "${FILTER_R}")
  fi
  echo "== ${label}: snapshot=${snap} obs=${obs} ${mode} =="
  python scripts/benchmark_policy.py \
    --snapshot "${snap}" \
    "${filt[@]}" \
    --task "${TASK}" --disruption "${DISRUPTION}" --human-model "${HUMAN_MODEL}" \
    --obs-mode "${obs}" \
    --num-demos-for-stats "${NUM_DEMOS_FOR_STATS}" \
    "${COUNT_ARGS[@]}" \
    --out "${OUTDIR}/${label}.csv"
  RAN+=("${label}")
}

run_row row1_baseline        "${STAGE2}"   oracle nofilter
run_row row2_workspace       "${ROW2:-}"   oracle nofilter
run_row row3_lagrangian      "${ROW3:-}"   oracle nofilter
run_row row4_baseline_filter "${STAGE2}"   oracle filter
run_row row5_hybrid          "${ROW3:-}"   oracle filter
run_row row5_hybrid_noisy    "${ROW3:-}"   noisy  filter

echo ""
echo "== E4.1 eval done. ran: ${RAN[*]:-none} =="
[[ ${#SKIPPED[@]} -gt 0 ]] && echo "== skipped (inputs unset): ${SKIPPED[*]} =="
echo "   Per-row CSVs in ${OUTDIR}/  (each benchmark_policy invocation appends one row)."
echo "   Headline check: row5_hybrid should dominate rows 1-4 on ep_proximity_violation_rate (non-overlapping CIs)."
