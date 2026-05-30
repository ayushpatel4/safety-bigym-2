#!/usr/bin/env bash
# E4.1 — headline feature-incremental eval driver (P5). Runs benchmark_policy.py
# for each table row (PURE EVAL, no training) and writes one CSV per row. Rows
# whose snapshot env var is unset are SKIPPED, so this is usable incrementally:
# run rows 1 & 4 now (need only STAGE2 + the SVF filter), then rows 2/3/5 once
# those snapshots exist (P5 row-2 training, P3 continuous + d_knee).
#
# Perception: the WHOLE table runs on one obs mode, OBS_MODE (default `noisy`).
# Why noisy: the SVF filter is trained on `noisy` and its Q-values collapse on
# `oracle` (mean_q ~0.016 << R -> 100% intervention, task destroyed; observed
# 2026-05-30). Running every row on noisy keeps the filter in-distribution and
# the comparison apples-to-apples (the policy's oracle->noisy degradation is
# identical across rows, so it cancels). Use OBS_MODE=oracle for a policy-only
# clean-perception reference — but the filter rows (4/5) are meaningless there.
#
# Rows (all on $OBS_MODE):
#   row1_baseline        STAGE2,  no filter   (unconstrained baseline)
#   row2_workspace       ROW2,    no filter   (+ workspace shaping ablation)
#   row3_lagrangian      ROW3,    no filter   (+ continuous-cost Lagrangian)
#   row4_baseline_filter STAGE2,  + filter    (baseline + runtime SVF)
#   row5_hybrid          ROW3,    + filter    (full hybrid)
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
#   OBS_MODE    perception mode for ALL rows. Default `noisy` (filter's native
#               distribution). `oracle` = policy-only clean reference (filter rows break).
#   STAGE2      row-1 + row-4 policy (P1 unconstrained baseline). Default = recorded G1 stage-2.
#   ROW3        row-3 + row-5 policy (P3 continuous + d_knee). Unset -> rows 3,5 skipped.
#   ROW2        row-2 policy (+ workspace-shaping training run). Unset -> row 2 skipped.
#   SVF_FILTER  SVF critic for rows 4/5. Default checkpoints/svf_coworker_train_g1_0p3.pt.
#   FILTER_R    veto threshold R. Default = snapshots.py::SVF_FILTER_THRESHOLD_R
#               (R=2.25 for saucepan, the dense-0.3m-sweep operating point).
#   SEEDS (0,1,2), EPISODES (20), DISRUPTION (coworker_train),
#   NUM_DEMOS_FOR_STATS (0 = faithful full count; cap on a laptop), OUTDIR.
#   RENDER (0)         set 1 to write rollout mp4(s) per row (needs MUJOCO_GL).
#   RENDER_EPISODES (1) how many of the first scored episodes to record per row.
#
# Usage:
#   STAGE2=... bash scripts/run_e4_1_headline.sh              # rows 1 & 4 now (noisy)
#   STAGE2=... ROW3=... bash scripts/run_e4_1_headline.sh     # rows 1,3,4,5 (noisy headline)
#   OBS_MODE=oracle STAGE2=... ROW3=... bash scripts/run_e4_1_headline.sh  # policy-only reference
#   SMOKE=1 STAGE2=... bash scripts/run_e4_1_headline.sh      # 1 seed x 2 ep x 50 steps
#   RENDER=1 RENDER_EPISODES=2 STAGE2=... ROW3=... bash scripts/run_e4_1_headline.sh  # + mp4s
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-saucepan_to_hob}"
HUMAN_MODEL="${HUMAN_MODEL:-g1}"
DISRUPTION="${DISRUPTION:-coworker_train}"
# Whole table on one perception mode. Default noisy = the SVF filter's native
# (training) distribution; on oracle the filter Q collapses -> 100% intervention.
OBS_MODE="${OBS_MODE:-noisy}"
# Recorded P1 stage-2 (filters/snapshots.py::G1_CURRICULUM). Repo-relative so it
# resolves from REPO_ROOT; override STAGE2=<abs path> to use another baseline.
STAGE2="${STAGE2:-exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage2_full/snapshot_28203.pt}"
SVF_FILTER="${SVF_FILTER:-checkpoints/svf_coworker_train_g1_0p3.pt}"
# Default the veto threshold R from the single source of truth — snapshots.py::
# SVF_FILTER_THRESHOLD_R (loaded standalone: stdlib-only, no torch). This is the
# operating point pinned from the dense 0.3 m sweep (R=2.25 for saucepan, NOT the
# old 4.0). Override with FILTER_R=<value>; falls back to 2.25 if the lookup fails.
FILTER_R="${FILTER_R:-$(python -c "import importlib.util as u;sp=u.spec_from_file_location('s','safety_bigym/filters/snapshots.py');m=u.module_from_spec(sp);sp.loader.exec_module(m);print(m.SVF_FILTER_THRESHOLD_R.get('${TASK}',2.25))" 2>/dev/null || echo 2.25)}"
NUM_DEMOS_FOR_STATS="${NUM_DEMOS_FOR_STATS:-0}"
# Optional rollout videos: RENDER=1 writes RENDER_EPISODES mp4(s) per row under
# <OUTDIR>/<label>_videos/step_<i>_ep0.mp4 (best-effort; needs MUJOCO_GL working).
RENDER_ARGS=()
[[ "${RENDER:-0}" == "1" ]] && RENDER_ARGS=(--render --render-episodes "${RENDER_EPISODES:-1}")

if [[ "${SMOKE:-0}" == "1" ]]; then
  COUNT_ARGS=(--smoke)            # --smoke overrides seeds/episodes/steps
else
  SEEDS="${SEEDS:-0,1,2}"
  EPISODES="${EPISODES:-20}"
  COUNT_ARGS=(--seeds "${SEEDS}" --episodes "${EPISODES}")
fi

_RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-e4_1_${TASK}_${OBS_MODE}_${_RUN_STAMP}}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/results/e4_1/${RUN_TAG}}"
mkdir -p "${OUTDIR}"
echo "== E4.1 headline eval | TASK=${TASK} disruption=${DISRUPTION} obs=${OBS_MODE} R=${FILTER_R} =="
echo "   OUTDIR=${OUTDIR}"
echo "   STAGE2=${STAGE2}"
echo "   ROW3=${ROW3:-<unset -> rows 3/5 skipped>}  ROW2=${ROW2:-<unset -> row 2 skipped>}"

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
    "${RENDER_ARGS[@]}" \
    --out "${OUTDIR}/${label}.csv"
  RAN+=("${label}")
}

run_row row1_baseline        "${STAGE2}"   "${OBS_MODE}" nofilter
run_row row2_workspace       "${ROW2:-}"   "${OBS_MODE}" nofilter
run_row row3_lagrangian      "${ROW3:-}"   "${OBS_MODE}" nofilter
run_row row4_baseline_filter "${STAGE2}"   "${OBS_MODE}" filter
run_row row5_hybrid          "${ROW3:-}"   "${OBS_MODE}" filter

echo ""
echo "== E4.1 eval done (obs=${OBS_MODE}). ran: ${RAN[*]:-none} =="
[[ ${#SKIPPED[@]} -gt 0 ]] && echo "== skipped (inputs unset): ${SKIPPED[*]} =="
echo "   Per-row CSVs in ${OUTDIR}/  (each benchmark_policy invocation appends one row)."
echo "   Headline check: row5_hybrid should dominate rows 1-4 on ep_proximity_violation_rate (non-overlapping CIs)."
