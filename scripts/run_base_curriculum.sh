#!/usr/bin/env bash
# Staged human-curriculum re-validation of the CQN-AS base policy on
# saucepan_to_hob, with the 2026-05-20 reward/critic-support fix applied.
#
# Why this script exists: the 50k single-stage validation produced a degenerate
# "retreat from human" policy. Root cause = the dense workspace penalty's
# discounted return blew past the C51 critic support [-2,2], saturating value
# learning (full writeup: docs/phase3_base_validation_findings.md). The fix is
# four levers, all applied here:
#   (1) bounded penalty:  workspace_beta=0.05, workspace_excess_cap=1.0
#   (2) widened support:  agent.v_min=-6 agent.v_max=2 agent.atoms=101
#   (3) more demos:       num_demos=36
#   (4) human curriculum: 3 stages via snapshot-resume (the env is stateless
#       w.r.t. training step, so no within-run ramp — we stage it instead):
#         stage 0  disruption=null          (no human; "can it learn the task?")
#         stage 1  disruption=coworker_easy  (gentle coworker)
#         stage 2  disruption=coworker_train (full coworker)
#
# Each stage resumes the previous stage's final snapshot via +snapshot_path=...
# (train_cqn_as.py load_snapshot). The env is rebuilt per stage, so the new
# disruption config takes effect with no code change.
#
# NOT launched from inside the agent — a human runs this on the GPU box.
#
# Prereqs:
#   export AMASS_DATA_DIR=/path/to/CMU/CMU
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0   # headless GPU
#   cd safety_bigym && source venv/bin/activate    # needs tensordict
#
# Usage:
#   scripts/run_base_curriculum.sh                 # full run (stages 0->1->2)
#   SMOKE=1 scripts/run_base_curriculum.sh         # ≤2000-frame stage-0 smoke only
#   STAGE0_FRAMES=60000 STAGE1_FRAMES=30000 STAGE2_FRAMES=40000 \
#       scripts/run_base_curriculum.sh             # override per-stage budgets
#
#   # Resume ONLY stage 2 (e.g. after a machine crash) — skips stages 0/1 and
#   # restarts stage 2 from the newest snapshot it can find. Point RESUME_DIR at
#   # the prior run dir; the resumed run writes to <RESUME_DIR>/stage2_full_resume.
#   RESUME_STAGE2=1 RESUME_DIR=exp_local/cqn_as_base_curriculum/<run_tag> \
#       scripts/run_base_curriculum.sh
#   # ...or pass an explicit checkpoint:
#   RESUME_STAGE2=1 RESUME_DIR=<dir> RESUME_SNAPSHOT=<path.pt> \
#       scripts/run_base_curriculum.sh

set -euo pipefail

if [[ -z "${AMASS_DATA_DIR:-}" ]]; then
  echo "ERROR: export AMASS_DATA_DIR before running (see CLAUDE.md)." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-base_curriculum_$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/exp_local/cqn_as_base_curriculum/${RUN_TAG}}"
mkdir -p "${OUTDIR}"

# Task selector (2026-05-25). The curriculum has been validated on
# `saucepan_to_hob` (the anchor task). Use `TASK=<name>` to point at any other
# yaml under cfgs/env/safety_bigym/ — e.g. `TASK=wall_cupboard_close` to
# confirm generality of the G1 operating config across tasks.
TASK="${TASK:-saucepan_to_hob}"

# Shared overrides — the reward/support fix (levers 1-3) + cadence/logging.
COMMON=(
  env=safety_bigym/${TASK}
  bodyslam=oracle
  num_demos=36
  agent.v_min=-6.0
  agent.v_max=2.0
  agent.atoms=101
  save_snapshot=true
  save_video=true
)

# G1 OPERATING CONFIG (locked 2026-05-25).
# The G1 base-curriculum bisection (attempts 1–6 over 2026-05-22 → 2026-05-25)
# established that:
#   - G1 RGB in the CNN encoder caused the policy to flee the workspace
#     under any reward signal (workspace shaping ON -> race-to-fail at the
#     10m fail() boundary, eps collapse to 300; shaping OFF -> long drift,
#     no task engagement). Recoloring G1 (skin-tone), disabling G1<->floor
#     contacts, neither closed the gap to the SMPL-H baseline.
#   - The one G1 stage-0 config that converges to task-attempting behaviour
#     with stable 800-1000 step episodes is MASK_PIXELS=1 (rgb_obs zeroed to
#     the agent) + WORKSPACE_PENALTY=1 (bounded shaping pulls the EE back
#     toward the task centre). At 30k frames it reached ep_reward -5.8,
#     beating the SMPL-H baseline's -7.2 best on the same anchor.
# DEFAULTS BELOW are set for that config. Override to 0 if you ever want to
# revisit the RGB / shaping-off paths. The visual axis being absent is a
# documented limitation, not a transient toggle — actor + critic are blind
# to RGB during G1 training, by design.
#
# Toggles:
#   WORKSPACE_PENALTY=1  (default) add_workspace_penalty=true (beta=0.05, cap=1.0)
#   WORKSPACE_PENALTY=0           add_workspace_penalty=false
#   MASK_PIXELS=1        (default) env_adapter zeros rgb_obs (encoder shape unchanged)
#   MASK_PIXELS=0                 rgb_obs passes through (re-enables CNN on G1; degenerate)
if [[ "${WORKSPACE_PENALTY:-1}" == "1" ]]; then
  COMMON+=(
    env.safety.add_workspace_penalty=true
    env.safety.workspace_beta=0.05
    env.safety.workspace_excess_cap=1.0
  )
else
  COMMON+=(
    env.safety.add_workspace_penalty=false
  )
fi

if [[ "${MASK_PIXELS:-1}" == "1" ]]; then
  COMMON+=(
    mask_pixels=true
  )
else
  COMMON+=(
    mask_pixels=false
  )
fi

# G1 floor-contact bisection toggle (2026-05-24). G1's feet contact the floor
# every step (SMPL-H's pelvis was welded to world / weldid=0, so its floor
# contacts were filtered by mjOPT_FILTERPARENT — G1 has no such weld). When
# DISABLE_FLOOR_COLLISION=1, _configure_collision_bits leaves the floor on
# its default bit-0 channel so human<->floor pairs become collision-filtered.
# Tests whether G1 foot-floor contacts inject solver noise that propagates to
# the robot and causes it to flee the workspace in stage 0.
# Human<->robot contacts and SSM tracking are unaffected.
#   DISABLE_FLOOR_COLLISION=0  -> default (G1 feet collide with floor)
#   DISABLE_FLOOR_COLLISION=1  -> human<->floor contacts filtered out
if [[ "${DISABLE_FLOOR_COLLISION:-0}" == "1" ]]; then
  COMMON+=(
    env.safety.disable_human_floor_collision=true
  )
else
  COMMON+=(
    env.safety.disable_human_floor_collision=false
  )
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  STAGE0_FRAMES="${STAGE0_FRAMES:-2000}"
  STAGE1_FRAMES=0
  STAGE2_FRAMES=0
  WANDB=(wandb.use=false)
  # Synchronous CUDA so any device-side assert points at the real op/line
  # (smoke is short; the slowdown is irrelevant here).
  export CUDA_LAUNCH_BLOCKING=1
else
  # Stage budgets (2026-05-26 bump): the 2026-05-25 saucepan_to_hob run with
  # stage0=30k plateaued at success_rate≈0.1-0.2 throughout stage 0 — base
  # policy wasn't task-competent before stage 1 ramped the disruption.
  # Doubling stage 0 to 60k gives the policy time to lock in the grasp/place
  # sequence under the easy distribution before the human approaches.
  STAGE0_FRAMES="${STAGE0_FRAMES:-60000}"
  STAGE1_FRAMES="${STAGE1_FRAMES:-30000}"
  STAGE2_FRAMES="${STAGE2_FRAMES:-40000}"
  WANDB=(wandb.use=true)
fi

# Latest snapshot (by mtime) in a stage dir; the final-state save at loop exit
# is the newest snapshot_<step>.pt.
latest_snapshot() {
  ls -t "$1"/snapshot_*.pt 2>/dev/null | head -1
}

run_stage() {
  local name="$1" disruption="$2" frames="$3" resume_from="${4:-}"
  local stage_dir="${OUTDIR}/${name}"
  if [[ "${frames}" -le 0 ]]; then
    echo "== skip ${name} (frames=${frames}) =="
    return
  fi
  echo "== ${name}: disruption=${disruption} frames=${frames} -> ${stage_dir} =="
  local extra=()
  [[ -n "${resume_from}" ]] && extra+=("+snapshot_path=${resume_from}")
  # W&B run tags grouped by (curriculum stage, method, task) so the thesis
  # plots — convergence curves per stage, Pareto frontier per method, etc.
  # — can be filtered with the W&B UI. `stageN` is the canonical stage
  # axis; `unconstrained` is hard-coded here because the base curriculum
  # trains the vendored CQN-AS reward critic only (P3.1's Lagrangian
  # launcher is a separate script that uses `lagrangian` / `hybrid`).
  #
  # Tags must NOT contain `=` — Hydra's override grammar treats `=` as the
  # key-value separator and fails to parse `[stage0,method=foo]`. So we
  # use flat labels (`unconstrained`, `task-${TASK}`) instead.
  local stage_tag
  case "${name}" in
    stage0*) stage_tag="stage0" ;;
    stage1*) stage_tag="stage1" ;;
    stage2*) stage_tag="stage2" ;;
    *)       stage_tag="${name}" ;;
  esac
  local wandb_tags="+wandb.tags=[${stage_tag},unconstrained,task-${TASK}]"
  python train_cqn_as.py \
    "${COMMON[@]}" "${WANDB[@]}" \
    "disruption=${disruption}" \
    num_train_frames="${frames}" \
    "hydra.run.dir=${stage_dir}" \
    "wandb.name=${RUN_TAG}_${name}" \
    "${wandb_tags}" \
    "${extra[@]}"
}

# --- Resume-only path: rerun stage 2 from the newest available snapshot ---
if [[ "${RESUME_STAGE2:-0}" == "1" ]]; then
  if [[ -z "${RESUME_DIR:-}" ]]; then
    echo "ERROR: RESUME_STAGE2=1 requires RESUME_DIR=<prior run dir>." >&2
    exit 1
  fi
  OUTDIR="${RESUME_DIR}"
  RUN_TAG="${RUN_TAG:-$(basename "${OUTDIR}")}"
  # Explicit snapshot wins; else newest from a prior stage-2 (resume or not),
  # else fall back to stage 1's final snapshot.
  SNAP="${RESUME_SNAPSHOT:-}"
  if [[ -z "${SNAP}" ]]; then
    SNAP="$(ls -t "${OUTDIR}"/stage2_full*/snapshot_*.pt 2>/dev/null | head -1)"
  fi
  [[ -z "${SNAP}" ]] && SNAP="$(latest_snapshot "${OUTDIR}/stage1_easy")"
  if [[ -z "${SNAP}" ]]; then
    echo "ERROR: no snapshot found under ${OUTDIR} (stage2_full*/ or stage1_easy/)." >&2
    exit 1
  fi
  echo "== resuming stage 2 from: ${SNAP} =="
  run_stage stage2_full_resume coworker_train "${STAGE2_FRAMES}" "${SNAP}"
  echo "== stage 2 resume complete. snapshots under ${OUTDIR}/stage2_full_resume =="
  exit 0
fi

# Stage 0: human present but idle/distant (no interference; keeps obs width and
# model architecture identical to stages 1-2 so snapshots resume cleanly). This
# is the GATE — episode_reward must climb >0 on some episodes with returns inside
# [-6, 2]. If it fails, the problem is demos/CQN-AS, not safety: stop and reassess.
run_stage stage0_idle coworker_idle "${STAGE0_FRAMES}"
SNAP0="$(latest_snapshot "${OUTDIR}/stage0_idle")"

run_stage stage1_easy coworker_easy "${STAGE1_FRAMES}" "${SNAP0}"
SNAP1="$(latest_snapshot "${OUTDIR}/stage1_easy")"

run_stage stage2_full coworker_train "${STAGE2_FRAMES}" "${SNAP1}"

echo "== curriculum complete. snapshots under ${OUTDIR} =="
