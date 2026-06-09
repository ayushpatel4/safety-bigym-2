#!/usr/bin/env bash
# One-off orchestrator: wait for the two in-flight fixed-lambda=0.15 training runs
# to exit, then run basin sweeps + operating-point picks + multiseed plots for
# BOTH lambda=0.1 and lambda=0.15 (3 seeds each). PURE EVAL after training.
# NOT -e: a single failed sweep cell must not abort the whole batch.
set -uo pipefail
cd /home/ap2322/Documents/safety_bigym
source venv/bin/activate
# Headless GL backend — REQUIRED for benchmark_policy offscreen rendering.
# (First run omitted this -> every sweep died with mujoco OpenGL FatalError.)
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# --- in-flight training runs to wait on (both exited; loop is now a no-op) ---
WAIT_PIDS="1381848 1369699"
FIX=exp_local/fixed_lambda/fixlam_0p1
GPUS="0 1 2 4 5"   # gpu3 excluded: another user's job is active there
mkdir -p results/figs

echo "[$(date)] waiting for training PIDs: $WAIT_PIDS"
for pid in $WAIT_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  echo "[$(date)] pid $pid exited"
done
echo "[$(date)] both lam0p15 runs finished -> starting basin sweeps"

run_lambda() {
  local TAG="$1"   # lam0p1 | lam0p15
  echo "============================================================"
  echo "[$(date)] === basin sweeps for $TAG ==="
  for s in 0 1 2; do
    STEPS=$(ls "$FIX/${TAG}_seed$s"/snapshot_[0-9]*.pt | sed 's/.*snapshot_//;s/\.pt//' | awk '$1>=7000' | sort -n | tr '\n' ' ')
    echo "[$(date)] $TAG seed$s STEPS: $STEPS"
    STAGE_DIR="$FIX/${TAG}_seed$s" OBS=noisy STEPS="$STEPS" GPUS="$GPUS" \
      bash scripts/run_basin_sweep.sh || echo "[$(date)] WARN: sweep $TAG seed$s returned non-zero"
  done
  for s in 0 1 2; do
    echo "[$(date)] pick $TAG seed$s:"
    python scripts/analyze_row3.py pick \
      --sweep-dir "results/e4_1/basin_${TAG}_seed${s}_noisy" --success-floor 0.75 \
      || echo "[$(date)] WARN: pick $TAG seed$s returned non-zero"
  done
}

run_lambda lam0p1
run_lambda lam0p15

echo "[$(date)] === plots ==="
python scripts/plot_basin_multiseed.py \
  --sweep-dir results/e4_1/basin_lam0p1_seed0_noisy results/e4_1/basin_lam0p1_seed1_noisy results/e4_1/basin_lam0p1_seed2_noisy \
  --baseline-prox 0.296 --baseline-succ 0.85 --ci --out results/figs/fixlam0p1_3seed.png \
  --title "fixed-λ=0.1: graceful operating point across 3 seeds" \
  || echo "[$(date)] WARN: lam0p1 plot failed"

python scripts/plot_basin_multiseed.py \
  --sweep-dir results/e4_1/basin_lam0p15_seed0_noisy results/e4_1/basin_lam0p15_seed1_noisy results/e4_1/basin_lam0p15_seed2_noisy \
  --baseline-prox 0.296 --baseline-succ 0.85 --ci --out results/figs/fixlam0p15_3seed.png \
  --title "fixed-λ=0.15: graceful operating point across 3 seeds" \
  || echo "[$(date)] WARN: lam0p15 plot failed"

echo "[$(date)] ALL DONE"
