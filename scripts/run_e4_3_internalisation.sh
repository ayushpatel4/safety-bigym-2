#!/usr/bin/env bash
# E4.3 / P8 — filter internalisation curve, POST-HOC on noisy. For each training
# snapshot in a P3/P4 run cell, evaluate the policy WITH the SVF filter on noisy
# and record filter_intervention_rate vs training frame. The rate should fall as
# the Lagrangian policy internalises safety (policy/filter complementarity).
#
# Why post-hoc + noisy (NOT the in-training FILTER_PASSIVE hook): P3/P4 train on
# `oracle`, and the SVF filter's Q collapses on oracle obs (mean_q~0.016 << R ->
# 100% intervention), so a FILTER_PASSIVE curve logged during oracle training is
# flat/meaningless. Evaluating the saved snapshots on `noisy` puts the filter in
# its training distribution, where intervention rate is meaningful.
#
# NOT free (the plan's old "piggybacks on P3" no longer holds): this is a
# post-hoc eval, ~few min per snapshot. Run on the GPU box.
#
# Prereqs:
#   cd ~/Documents/safety_bigym && source venv/bin/activate
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
#   export AMASS_DATA_DIR=...        # harness replays demos for CQN-AS action stats
#
# Usage:
#   RUN_DIR=exp_local/e3_2_cost_budget/<run>/d0p01_seed0 \
#     bash scripts/run_e4_3_internalisation.sh
#   SMOKE=1 RUN_DIR=... bash scripts/run_e4_3_internalisation.sh   # newest snapshot only, --smoke
#
# Inputs (env vars):
#   RUN_DIR (required)  dir holding snapshot_<N>.pt (one P3/P4 training cell)
#   SVF_FILTER  default checkpoints/svf_coworker_train_g1_0p3.pt
#   FILTER_R    default = snapshots.py::SVF_FILTER_THRESHOLD_R (R=2.25)
#   TASK (saucepan_to_hob), DISRUPTION (coworker_train), HUMAN_MODEL (g1),
#   SEEDS (0,1,2), EPISODES (10), OUT (results/e4_3/<run_tag>)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${RUN_DIR:?Set RUN_DIR to a P3/P4 cell dir containing snapshot_<N>.pt}"
[[ -d "${RUN_DIR}" ]] || { echo "ERROR: RUN_DIR=${RUN_DIR} not a directory" >&2; exit 1; }

TASK="${TASK:-saucepan_to_hob}"
DISRUPTION="${DISRUPTION:-coworker_train}"
HUMAN_MODEL="${HUMAN_MODEL:-g1}"
SVF_FILTER="${SVF_FILTER:-checkpoints/svf_coworker_train_g1_0p3.pt}"
# Veto threshold from the single source of truth (snapshots.py), standalone load.
FILTER_R="${FILTER_R:-$(python -c "import importlib.util as u;sp=u.spec_from_file_location('s','safety_bigym/filters/snapshots.py');m=u.module_from_spec(sp);sp.loader.exec_module(m);print(m.SVF_FILTER_THRESHOLD_R.get('${TASK}',2.25))" 2>/dev/null || echo 2.25)}"
[[ -f "${SVF_FILTER}" ]] || { echo "ERROR: SVF_FILTER=${SVF_FILTER} not found" >&2; exit 1; }

_RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-e4_3_$(basename "${RUN_DIR}")_${_RUN_STAMP}}"
OUT="${OUT:-${REPO_ROOT}/results/e4_3/${RUN_TAG}}"
mkdir -p "${OUT}"
CURVE="${OUT}/internalisation_curve.csv"
printf 'frame,filter_intervention_rate,success_rate,ep_proximity_violation_rate\n' > "${CURVE}"

# Frame-indexed snapshots, numeric order; skip snapshot_best.pt. (Plain
# while-read + index, not mapfile/negative-index, so it works on bash 3.2 too.)
SNAPS=()
while IFS= read -r _s; do SNAPS+=("${_s}"); done \
  < <(ls "${RUN_DIR}"/snapshot_*.pt 2>/dev/null | grep -v snapshot_best | sort -V)
[[ ${#SNAPS[@]} -gt 0 ]] || { echo "ERROR: no snapshot_<N>.pt in ${RUN_DIR}" >&2; exit 1; }

if [[ "${SMOKE:-0}" == "1" ]]; then
  SNAPS=("${SNAPS[$(( ${#SNAPS[@]} - 1 ))]}")   # newest only
  COUNT_ARGS=(--smoke)
else
  COUNT_ARGS=(--seeds "${SEEDS:-0,1,2}" --episodes "${EPISODES:-10}")
fi

echo "== E4.3 internalisation (noisy) | RUN_DIR=${RUN_DIR} | ${#SNAPS[@]} snapshot(s) | R=${FILTER_R} =="
for snap in "${SNAPS[@]}"; do
  b="$(basename "${snap}" .pt)"; frame="${b#snapshot_}"
  cell_csv="${OUT}/step_${frame}.csv"
  echo "-- frame ${frame}: ${snap}"
  python scripts/benchmark_policy.py \
    --snapshot "${snap}" \
    --filter-snapshot "${SVF_FILTER}" --filter-threshold "${FILTER_R}" \
    --task "${TASK}" --disruption "${DISRUPTION}" --human-model "${HUMAN_MODEL}" \
    --obs-mode noisy --num-demos-for-stats "${NUM_DEMOS_FOR_STATS:-0}" \
    "${COUNT_ARGS[@]}" \
    --out "${cell_csv}"
  # Append (frame, intervention, success, proximity) from the cell CSV's last row.
  python - "${cell_csv}" "${frame}" >> "${CURVE}" <<'PY'
import csv, sys
path, frame = sys.argv[1], sys.argv[2]
with open(path) as f:
    rows = list(csv.DictReader(f))
r = rows[-1] if rows else {}
g = lambda k: r.get(k, "")
print(f"{frame},{g('filter_intervention_rate')},{g('success_rate')},{g('ep_proximity_violation_rate')}")
PY
done

echo ""
echo "== E4.3 done. curve -> ${CURVE} =="
echo "   Expectation: filter_intervention_rate falls as frame grows (policy internalises safety)."
