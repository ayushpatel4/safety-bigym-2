#!/usr/bin/env bash
# P2 RE-DO (2026-06-01, v3): re-collect → retrain → re-sweep the G1 SVF after the
# TWO snapshot-policy fixes that align collection with deployment:
#   1. action de-normalisation — de-normalise via the agent's DEMO-derived stats,
#      not env.action_space (v2 fixed this: benchmark mean_q 0.02 → 0.97).
#   2. action-execution mode — execute open-loop action_sequence chunks +
#      temporal-ensemble blend (mirroring benchmark CQNASRunner), NOT receding-
#      horizon chunk[0]. v2 still had this bug: with action_sequence=16 +
#      temporal_ensemble=true the policy deploys BLENDED actions, but the v2
#      critic trained on raw chunk[0] -> ~89% spurious veto, success 0.
#   3. env control_frequency (THE root cause) — _build_live_env now sets
#      control_frequency = CONTROL_FREQUENCY_MAX // demo_down_sample_rate (20 Hz
#      for saucepan), matching the factory _create_env the deployment adapter uses.
#      Before, the collection env ran at the full 500 Hz: each action moved the
#      robot ~25x less/step (policy never completed the task: 0% vs 85% success),
#      action_scale was 25x off, and 1000 steps covered ~2s not ~50s (coworker
#      barely approached). MAX_STEPS=1000 (=deployment horizon) is now meaningful;
#      episodes run longer, so EPISODES_PER_CELL defaults to 150 (was 210).
# v3 rebuilds the dataset/critic/operating-point on a snapshot policy that runs
# EXACTLY what deployment runs (execution mode + horizon), so the sweep finally
# predicts the benchmark.
#
# Output: datasets/svf_coworker_train_g1_0p3_v3/, checkpoints/svf_coworker_train_g1_0p3_v3.pt,
#         results/svf_sweep_g1_0p3_v3/sweep_dense_seed{0,1,2}.csv
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
#   STAGE2=$STAGE2 bash scripts/run_p2_recollect_g1.sh                 # full v3 re-do
#   STAGES=preflight STAGE2=$STAGE2 bash scripts/run_p2_recollect_g1.sh   # validate the fix only
#   STAGES=collect,train,sweep STAGE2=$STAGE2 bash scripts/run_p2_recollect_g1.sh
#
# Optional ~1-GPU-h pre-check — confirm the execution-mode fix makes the sweep
# match the benchmark on the EXISTING v2 critic (expect the v2 sweep to now
# COLLAPSE like the benchmark: high intervention, mean_q~1, validating the
# diagnosis before spending the full re-collect):
#   STAGES=sweep VER=v2 SWEEP_DIR=results/svf_sweep_g1_0p3_v2_ensemblecheck \
#     STAGE2=$STAGE2 bash scripts/run_p2_recollect_g1.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${STAGE2:?Set STAGE2 to the P1 stage-2 G1 snapshot (.pt)}"
TASK="${TASK:-saucepan_to_hob}"
# Version tag for dataset/critic/sweep. v3 = execution-mode fix (open-loop +
# ensemble). v2 = de-norm-only (SUPERSEDED: trained on receding-horizon chunk[0]).
VER="${VER:-v3}"
DATASET="${DATASET:-datasets/svf_coworker_train_g1_0p3_${VER}}"
CKPT="${CKPT:-checkpoints/svf_coworker_train_g1_0p3_${VER}.pt}"
SWEEP_DIR="${SWEEP_DIR:-results/svf_sweep_g1_0p3_${VER}}"
STAGES="${STAGES:-preflight,collect,train,sweep}"
PROX="${PROX:-0.3}"
OVR=(--snapshot-override "${TASK}=${STAGE2}")

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

# --- PREFLIGHT: prove the de-norm fix engages (else abort before wasting compute)
# NOT `--smoke`: CollectionPlan.smoke() hardcodes source=random task=dishwasher_close
# bodyslam=oracle and never loads a snapshot policy, so it can't exercise the
# de-norm path. Run a tiny REAL snapshot collection (1 ep) that reaches
# _load_cqn_as_snapshot_policy, which logs DEMO-derived (PASS) or FALLS BACK / no
# num_demos (FAIL). Cleanup runs on every exit path.
if has_stage preflight; then
  echo "== preflight: 1-episode snapshot collection; check de-norm uses DEMO stats =="
  PRE="$(mktemp)"
  PRE_OUT="/tmp/svf_preflight_$$"
  python scripts/svf_collect_dataset.py --source snapshot \
    --tasks "${TASK}" --disruption-space coworker_train --bodyslam-mode noisy \
    --human-model g1 --proximity-threshold "${PROX}" \
    --episodes-per-cell 1 --max-steps "${PREFLIGHT_STEPS:-8}" \
    --output-dir "${PRE_OUT}" "${OVR[@]}" 2>&1 | tee "${PRE}"
  if grep -q "DEMO-derived action stats" "${PRE}"; then
    echo "== preflight PASS: snapshot policy de-normalises with demo stats. =="
    rm -f "${PRE}"; rm -rf "${PRE_OUT}"
  elif grep -qE "FALLS BACK to env.action_space|cfg has no num_demos" "${PRE}"; then
    echo "== preflight FAIL: de-norm fell back to env.action_space (the bug). ==" >&2
    echo "   Likely AMASS_DATA_DIR / demos not available. Fix before collecting." >&2
    rm -f "${PRE}"; rm -rf "${PRE_OUT}"; exit 2
  else
    echo "== preflight INCONCLUSIVE: snapshot policy never logged a de-norm line. ==" >&2
    echo "   Did the snapshot source load? (Check for a 'skip'/None snapshot, or a" >&2
    echo "   crash before policy build.) Inspect ${PRE}." >&2
    rm -rf "${PRE_OUT}"; exit 2
  fi
fi

# --- COLLECT: snapshot (+ optional random), coworker_train, noisy, tau=0.3 -----
# SOURCES default = "snapshot" only. At the correct 20 Hz control rate the RANDOM
# source flails violently into the close (0.6 m) coworker -> MuJoCo contact-solver
# overload (~1-2 physics steps/sec, effectively hangs) + frequent NaN instability.
# The snapshot source (sensible robot, aggressive coworker) is fast + stable + is
# the deployment-matched distribution, and with the tightened coworker it still
# visits proximity<tau (unsafe) states. Add random back with
# SOURCES="snapshot random" only if svf_train_critic reports too few violations
# (it logs the unsafe fraction) — but expect random to be slow.
if has_stage collect; then
  SOURCES="${SOURCES:-snapshot}"
  SRC_ARGS=()
  for _s in ${SOURCES}; do SRC_ARGS+=(--source "${_s}"); done
  echo "== collect (sources: ${SOURCES}) -> ${DATASET} =="
  python scripts/svf_collect_dataset.py \
    "${SRC_ARGS[@]}" \
    --tasks "${TASK}" --disruption-space coworker_train --bodyslam-mode noisy \
    --human-model g1 --proximity-threshold "${PROX}" \
    --episodes-per-cell "${EPISODES_PER_CELL:-100}" --max-steps "${MAX_STEPS:-1000}" \
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
      --episodes-per-R "${EPISODES_PER_R:-12}" --max-steps "${MAX_STEPS:-1000}" \
      --seed "${SEED}" --output-csv "${SWEEP_DIR}/sweep_dense_seed${SEED}.csv"
  done
  echo "== sweep done. Picking the knee (seed-averaged, P2 acceptance bar): =="
  python scripts/analyze_svf_sweep.py --sweep-dir "${SWEEP_DIR}" || true
  echo "== Then set snapshots.py::SVF_FILTERS['${TASK}']='checkpoints/${CKPT##*/}' + "
  echo "   SVF_FILTER_THRESHOLD_R['${TASK}']=<knee R above>, commit, and re-run E4.1 rows 1/4. =="
fi
