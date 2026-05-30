#!/usr/bin/env bash
# Dense SVF threshold sweep on the G1 stage-2 baseline policy — adds the
# filterless R=0 baseline + fills the 26%→82% intervention gap left by the
# coarse {1,2,3,4,5,6,8} sweep in results/svf_sweep_g1_v1/sweep_seed{0,1,2}.csv.
#
# Why R=0.0 gives the filterless baseline: the SVF critic output is
# q = q_max·sigmoid(logit) ∈ (0, 100] (critic.py:73) and the veto fires on
# q < R (runtime_wrapper.py:79). With R=0 the veto never triggers, so the
# R=0.0 row IS the unfiltered policy under the *identical* env / policy / seeds
# — the apples-to-apples denominator for the "≥30% proximity reduction"
# acceptance check, and the missing left end of the Pareto figure.
#
# Eval-only (no training). ~2 GPU-h for 10 thresholds × 20 ep × 3 seeds.
# Run on the GPU box (swirl). G1 is AMASS-free, so AMASS_DATA_DIR is not needed.
#
# Required env vars (paths live on the GPU box):
#   SVF_CRITIC      — retrained G1 SVF checkpoint (e.g. checkpoints/svf_coworker_train_g1_v1.pt)
#   STAGE2_SNAPSHOT — P1 stage-2 G1 baseline snapshot (CQN-AS .pt)
# Optional:
#   POLICY          — "snapshot" (default, deployment-accurate) or "random"
#                     MUST match whatever produced your existing CSVs for the
#                     old rows to be comparable. snapshot loads the CQN-AS G1
#                     policy via _CQNASSnapshotPolicy.
#   THRESHOLDS, SEEDS, EPISODES, MAXSTEPS, OUTDIR — see defaults below.
set -euo pipefail

: "${SVF_CRITIC:?Set SVF_CRITIC to the retrained G1 SVF checkpoint path}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

POLICY="${POLICY:-snapshot}"
# 0.0 = filterless baseline; 1.5/2.25/2.5/2.75 fill the intervention gap; the
# rest reproduce the coarse sweep so each seed CSV is one self-consistent Pareto.
THRESHOLDS="${THRESHOLDS:-0.0 1.0 1.5 2.0 2.25 2.5 2.75 3.0 4.0 5.0}"
SEEDS="${SEEDS:-0 1 2}"
EPISODES="${EPISODES:-20}"   # match the existing sweep (5000 steps / 20 ep)
MAXSTEPS="${MAXSTEPS:-250}"  # match the existing sweep (250 steps/episode)
OUTDIR="${OUTDIR:-${REPO_ROOT}/results/svf_sweep_g1_v1}"

SNAPSHOT_FLAGS=()
if [[ "${POLICY}" == "snapshot" ]]; then
  : "${STAGE2_SNAPSHOT:?POLICY=snapshot needs STAGE2_SNAPSHOT (the P1 stage-2 G1 .pt)}"
  SNAPSHOT_FLAGS=(--snapshot-override "saucepan_to_hob=${STAGE2_SNAPSHOT}")
fi

mkdir -p "${OUTDIR}"
echo "SVF dense sweep | policy=${POLICY} | R={${THRESHOLDS}} | seeds={${SEEDS}}"
echo "  critic=${SVF_CRITIC}"
[[ "${POLICY}" == "snapshot" ]] && echo "  baseline policy=${STAGE2_SNAPSHOT}"
echo "  out=${OUTDIR}/sweep_dense_seed<seed>.csv"

for SEED in ${SEEDS}; do
  echo "=== seed ${SEED} ==="
  python "${SCRIPT_DIR}/svf_threshold_sweep.py" \
    --critic-path "${SVF_CRITIC}" \
    --task saucepan_to_hob \
    --disruption coworker_train \
    --human-model g1 \
    --bodyslam-mode noisy \
    --policy "${POLICY}" \
    "${SNAPSHOT_FLAGS[@]}" \
    --thresholds ${THRESHOLDS} \
    --episodes-per-R "${EPISODES}" \
    --max-steps "${MAXSTEPS}" \
    --seed "${SEED}" \
    --output-csv "${OUTDIR}/sweep_dense_seed${SEED}.csv"
done

echo "Done. Filterless baseline = the R=0.0 row's proximity_violation_rate."
echo "Acceptance: (baseline_prox − prox_at_R) / baseline_prox ≥ 0.30 at intervention ≤ 0.25."
