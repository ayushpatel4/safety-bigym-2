#!/usr/bin/env bash
# WCSAC (E3.7 / P9) <=100-frame WIRING SMOKE -- not training.
#
# Proves the external safe-RL baseline integrates end-to-end on the CQN-AS
# stack: env -> act -> env.step -> replay -> agent.update (twin reward critic,
# Gaussian safety critic, CVaR, lambda dual update, entropy temp) -> eval, all
# producing finite metrics. Runs ~1-2 min after the (slow) first env build.
#
# Episodes are shrunk to ~20 control steps (env.episode_length=400 /
# demo_down_sample_rate=20) so 100 frames completes several episodes and
# actually exercises the update + eval paths.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (safety_bigym)

export AMASS_DATA_DIR="${AMASS_DATA_DIR:-/home/ap2322/Documents/CMU/CMU}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

TASK="${1:-dishwasher_close}"

venv/bin/python train_cqn_as.py \
  env=safety_bigym/${TASK} agent=wcsac \
  num_train_frames=100 num_seed_frames=0 env.episode_length=400 \
  replay_buffer_num_workers=1 num_demos=0 \
  eval_every_frames=60 num_eval_episodes=2 agent.num_expl_steps=10 \
  save_snapshot=false save_video=false wandb.use=false device=cuda seed=0

echo
echo "Smoke OK if you saw [train]/[eval] lines, a rising episode_lambda, and"
echo "'final metrics written: ...'. Inspect the run's metrics.jsonl for the"
echo "per-update train/ metrics (actor_loss, cvar, safety_var_loss, lambda)."
