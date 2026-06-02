#!/usr/bin/env bash
# scripts/dispatch_p3p4_pool.sh
# GPU-pool dispatcher for the REMAINING E3.1/E3.2 (P3/P4) cells.
#
# Launches one cell per free GPU, only onto GPUs that have no running compute
# process, and picks GPUs up as they free. De-dups E3.1 `continuous` == E3.2
# `d0.01` (runs continuous once, symlinks it in as d0p01 at the end). Idempotent
# and crash-retrying: skips cells whose hydra dir already has final_metrics.json,
# skips cells already running, re-queues a crashed cell up to MAX_RETRY times.
#
# PREREQS — kill the serial launcher LOOPS first so they don't launch cells behind
# this dispatcher's back (any in-flight cell keeps running; that's fine):
#     pkill -f run_e3_1_cost_signal.sh ; pkill -f run_e3_2_cost_budget.sh
# Run this under tmux/nohup so it survives your shell. Cells are nohup'd so they
# survive the dispatcher too.
#
# Usage:
#     bash scripts/dispatch_p3p4_pool.sh
#     GPUS="0 3 4 5" POLL=30 bash scripts/dispatch_p3p4_pool.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

GPUS="${GPUS:-0 3 4 5}"
POLL="${POLL:-30}"
FRAMES="${FRAMES:-60000}"
MAX_RETRY="${MAX_RETRY:-1}"
STAGE1="${WARMSTART:-$REPO_ROOT/exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage1_easy/snapshot_2588.pt}"
E31="${E31:-$REPO_ROOT/exp_local/e3_1_cost_signal/e3_1_saucepan_to_hob_20260531_190551}"
E32="${E32:-$REPO_ROOT/exp_local/e3_2_cost_budget/e3_2_saucepan_to_hob_20260531_190307}"
LOGDIR="$REPO_ROOT/logs/dispatch_p3p4"; mkdir -p "$LOGDIR"

[[ -f "$STAGE1" ]] || { echo "FATAL: warm-start snapshot not found: $STAGE1" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi not on PATH" >&2; exit 1; }
if pgrep -f 'run_e3_1_cost_signal.sh|run_e3_2_cost_budget.sh' >/dev/null 2>&1; then
  echo "WARN: a run_e3_* launcher LOOP is still alive — kill it first or it will launch" >&2
  echo "      d0.01/continuous behind my back:  pkill -f run_e3_1_cost_signal.sh ; pkill -f run_e3_2_cost_budget.sh" >&2
fi

# Remaining de-duped pool ("kind:param:seed"). d0.01 is ABSENT (== continuous).
# binary:0 and d0.001:2 may be in-flight (orphaned) — the done/running guards skip
# them automatically; if either died, it gets launched normally.
#
# 2026-06-02: added the LOOSER budgets (0.2/0.3/0.5). The original sweep (<=0.1)
# sits BELOW the task's inherent per-step cost (~0.2-0.3, baseline proximity ~0.30),
# so every tested cell collapsed to ~0 task success (safety by task-abandonment).
# 0.2-0.3 is the graceful Lagrangian regime; 0.5 is the near-unconstrained anchor.
# The done-guard skips any already-finished cell, so re-running is safe.
QUEUE=(
  e31:binary:0 e31:binary:1 e31:binary:2
  e31:continuous:0 e31:continuous:1 e31:continuous:2
  e32:0.05:0 e32:0.05:1 e32:0.05:2
  e32:0.1:0 e32:0.1:1 e32:0.1:2
  e32:0.001:2
  e32:0.2:0 e32:0.2:1 e32:0.2:2
  e32:0.3:0 e32:0.3:1 e32:0.3:2
  e32:0.5:0 e32:0.5:1 e32:0.5:2
)
declare -A ATTEMPTS PID_ON CELL_ON

cell_dir() { if [[ "$1" == e31 ]]; then echo "$E31/${2}_seed${3}"; else echo "$E32/d${2//./p}_seed${3}"; fi; }
cell_done()    { [[ -f "$(cell_dir "$1" "$2" "$3")/final_metrics.json" ]]; }
cell_running() { pgrep -f "hydra.run.dir=$(cell_dir "$1" "$2" "$3") " >/dev/null 2>&1; }
gpu_busy()     { local o; o="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" 2>/dev/null)"; [[ -n "${o//[$' \t\n']/}" ]]; }

launch_cell() { local g="$1" k="$2" p="$3" s="$4" name log
  if [[ "$k" == e31 ]]; then name="${p}_seed${s}"; log="$LOGDIR/e31_${name}.log"
    CUDA_VISIBLE_DEVICES="$g" COST_FORMS="$p" SEEDS="$s" COST_BUDGET=0.01 FRAMES="$FRAMES" \
      WARMSTART="$STAGE1" OUTDIR="$E31" RUN_TAG=e3_1 nohup bash scripts/run_e3_1_cost_signal.sh >"$log" 2>&1 &
  else name="d${p//./p}_seed${s}"; log="$LOGDIR/e32_${name}.log"
    CUDA_VISIBLE_DEVICES="$g" COST_BUDGETS="$p" SEEDS="$s" FRAMES="$FRAMES" \
      WARMSTART="$STAGE1" OUTDIR="$E32" RUN_TAG=e3_2 nohup bash scripts/run_e3_2_cost_budget.sh >"$log" 2>&1 &
  fi; echo "$!"; }

echo "== P3/P4 pool dispatcher | GPUS='$GPUS' POLL=${POLL}s frames=$FRAMES =="
echo "   E31=$E31"; echo "   E32=$E32"; echo "   warmstart=$STAGE1"; echo "   pool=${#QUEUE[@]} cells; logs in $LOGDIR"

while :; do
  # 1) reap finished assignments (verify completion; retry crashes)
  for g in $GPUS; do
    pid="${PID_ON[$g]:-}"; [[ -z "$pid" ]] && continue
    kill -0 "$pid" 2>/dev/null && continue
    c="${CELL_ON[$g]}"; IFS=: read -r k p s <<<"$c"
    if cell_done "$k" "$p" "$s"; then
      echo "[$(date +%H:%M)] gpu$g  DONE  $c"
    else
      a="${ATTEMPTS[$c]:-0}"
      if (( a < MAX_RETRY )); then ATTEMPTS[$c]=$((a+1)); QUEUE+=("$c")
        echo "[$(date +%H:%M)] gpu$g  FAIL  $c -> requeued (retry $((a+1))/$MAX_RETRY)"
      else echo "[$(date +%H:%M)] gpu$g  FAIL  $c -> giving up (see $LOGDIR)"; fi
    fi
    unset 'PID_ON[$g]' 'CELL_ON[$g]'
  done

  # 2) prune already-finished cells from the queue
  if (( ${#QUEUE[@]} )); then
    nq=(); for c in "${QUEUE[@]}"; do IFS=: read -r k p s <<<"$c"; cell_done "$k" "$p" "$s" || nq+=("$c"); done
    QUEUE=("${nq[@]}")
  fi

  # 3) assign queued cells to free GPUs
  for g in $GPUS; do
    [[ -n "${PID_ON[$g]:-}" ]] && continue     # we already run a cell here
    gpu_busy "$g" && continue                   # external proc (orphaned cell / curriculum) -> wait
    (( ${#QUEUE[@]} )) || continue
    idx=-1
    for i in "${!QUEUE[@]}"; do IFS=: read -r k p s <<<"${QUEUE[$i]}"
      cell_done "$k" "$p" "$s" && continue; cell_running "$k" "$p" "$s" && continue; idx="$i"; break; done
    [[ "$idx" -lt 0 ]] && continue
    c="${QUEUE[$idx]}"; IFS=: read -r k p s <<<"$c"
    np="$(launch_cell "$g" "$k" "$p" "$s")"; PID_ON[$g]="$np"; CELL_ON[$g]="$c"
    echo "[$(date +%H:%M)] gpu$g  <-    $c  (pid $np)"
    unset 'QUEUE[$idx]'; QUEUE=("${QUEUE[@]}"); sleep 5
  done

  # 4) done when queue empty AND nothing we launched is still running
  act=0; for g in $GPUS; do [[ -n "${PID_ON[$g]:-}" ]] && act=$((act+1)); done
  (( ${#QUEUE[@]} == 0 && act == 0 )) && { echo "== all pool cells complete =="; break; }
  sleep "$POLL"
done

# 5) de-dup: expose continuous_seed* as the E3.2 d=0.01 Pareto point
for s in 0 1 2; do
  src="$E31/continuous_seed$s"; dst="$E32/d0p01_seed$s"
  [[ -d "$src" && ! -e "$dst" ]] && ln -s "$src" "$dst" && echo "symlink $dst -> continuous_seed$s"
done
echo "== DONE.  cost-form table : python scripts/analyze_e3.py --in-dir $E31"
echo "==        budget Pareto   : python scripts/analyze_e3.py --in-dir $E32"
