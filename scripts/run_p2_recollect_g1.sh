#!/usr/bin/env bash
# P2 RE-DO (2026-06-01): re-collect → retrain → re-sweep the G1 SVF after the
# action de-normalisation fix. The old `svf_coworker_train_g1_0p3.pt` was
# collected with `_CQNASSnapshotPolicy` de-normalising via env.action_space
# instead of the agent's demo-derived stats, so the critic saw a mis-scaled
# policy and over-vetoes at runtime (benchmark_policy mean_q~0.02 → ~100%
# intervention). The fix makes the snapshot policy de-normalise like deployment;
# this rebuilds the dataset/critic/operating-point on the CORRECT policy.
#
# Output: datasets/svf_coworker_train_g1_0p3_v2/, checkpoints/svf_coworker_train_g1_0p3_v2.pt,
#         results/svf_sweep_g1_0p3_v2/sweep_dense_seed{0,1,2}.csv
#
# NOT launched from inside the agent — a human runs this on the GPU box.
# ~6 GPU-h (collect ~3h + train ~1h + sweep ~1h + preflight).
#
# Prereqs:
#   cd ~/Documents/safety_bigym && git pull && source venv/bin/activate
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
#   export AMASS_DATA_DIR=~/Documents/CMU/CMU     # get_demos injects human-pos (needs AMASS)
#   export STAGE2=.../stage2_full/snapshot_28203.pt
#
# Usage:
#   STAGE2=$STAGE2 bash scripts/run_p2_recollect_g1.sh
#   STAGES=preflight bash scripts/run_p2_recollect_g1.sh        # just validate the fix
#   STAGES=collect,train,sweep STAGE2=$STAGE2 bash scripts/run_p2_recollect_g1.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${STAGE2:?Set STAGE2 to the P1 stage-2 G1 snapshot (.pt)}"
TASK="${TASK:-saucepan_to_hob}"
DATASET="${DATASET:-datasets/svf_coworker_train_g1_0p3_v2}"
CKPT="${CKPT:-checkpoints/svf_coworker_train_g1_0p3_v2.pt}"
SWEEP_DIR="${SWEEP_DIR:-results/svf_sweep_g1_0p3_v2}"
STAGES="${STAGES:-preflight,collect,train,sweep}"
PROX="${PROX:-0.3}"
OVR=(--snapshot-override "${TASK}=${STAGE2}")

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

# --- PREFLIGHT: prove the de-norm fix engages (else abort before wasting compute)
if has_stage preflight; then
  echo "== preflight: smoke-collect snapshot source; check de-norm uses DEMO stats =="
  PRE="$(mktemp)"
  python scripts/svf_collect_dataset.py --smoke --source snapshot \
    --tasks "${TASK}" --disruption-space coworker_train --bodyslam-mode noisy \
    --human-model g1 --proximity-threshold "${PROX}" \
    --output-dir "/tmp/svf_preflight_$$" "${OVR[@]}" 2>&1 | tee "${PRE}"
  if grep -q "DEMO-derived action stats" "${PRE}"; then
    echo "== preflight PASS: snapshot policy de-normalises with demo stats. =="
  elif grep -q "FALLS BACK to env.action_space" "${PRE}"; then
    echo "== preflight FAIL: de-norm FELL BACK to env.action_space (the bug). ==" >&2
    echo "   Likely AMASS_DATA_DIR / demos not available. Fix before collecting." >&2
    rm -f "${PRE}"; exit 2
  else
    echo "== preflight INCONCLUSIVE: no de-norm log line found; inspect ${PRE}. ==" >&2
    exit 2
  fi
  rm -f "${PRE}"; rm -rf "/tmp/svf_preflight_$$"
fi

# --- COLLECT: random + snapshot, coworker_train, noisy, tau=0.3 ----------------
if has_stage collect; then
  echo "== collect -> ${DATASET} =="
  python scripts/svf_collect_dataset.py \
    --source random --source snapshot \
    --tasks "${TASK}" --disruption-space coworker_train --bodyslam-mode noisy \
    --human-model g1 --proximity-threshold "${PROX}" \
    --episodes-per-cell "${EPISODES_PER_CELL:-210}" --max-steps "${MAX_STEPS:-250}" \
    --seed "${SEED:-0}" --output-dir "${DATASET}" "${OVR[@]}"
fi

# --- TRAIN: same hyperparams as the original g1 critic ------------------------
if has_stage train; then
  echo "== train -> ${CKPT} =="
  python scripts/svf_train_critic.py \
    --dataset-dir "${DATASET}" --proximity-threshold "${PROX}" \
    --num-steps "${NUM_STEPS:-200000}" --batch-size 512 --cql-alpha 5.0 \
    --target-tau 5e-3 --lr 3e-4 --gamma 0.99 --output "${CKPT}"
fi

# --- SWEEP: dense R on the de-norm-fixed critic, snapshot policy, 3 seeds -----
if has_stage sweep; then
  mkdir -p "${SWEEP_DIR}"
  for SEED in ${SWEEP_SEEDS:-0 1 2}; do
    echo "== sweep seed ${SEED} -> ${SWEEP_DIR}/sweep_dense_seed${SEED}.csv =="
    python scripts/svf_threshold_sweep.py \
      --critic-path "${CKPT}" --task "${TASK}" --disruption coworker_train \
      --human-model g1 --bodyslam-mode noisy --policy snapshot "${OVR[@]}" \
      --thresholds 0 1 1.5 2 2.25 2.5 2.75 3 3.5 4 \
      --episodes-per-R "${EPISODES_PER_R:-20}" --max-steps "${MAX_STEPS:-250}" \
      --seed "${SEED}" --output-csv "${SWEEP_DIR}/sweep_dense_seed${SEED}.csv"
  done
  echo "== sweep done. Find the new knee, update snapshots.py::SVF_FILTERS + "
  echo "   SVF_FILTER_THRESHOLD_R to ${CKPT##*/} + the new R, then re-run E4.1 rows 4/5. =="
fi
