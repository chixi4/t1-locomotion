#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/isaacgym/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

TASK="T1Upper9CameraStableOfficial_from7000LegFrozen_train500"
LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"

echo "==== START train $(date '+%Y-%m-%d %H:%M:%S') task=${TASK} label=upper9_camera_stable_frozen500 ===="
stdbuf -oL -eL python3 -u train_shoulder4_frozen.py \
  --task "${TASK}" \
  --leg_checkpoint "${LEG_CKPT}" \
  --max_iterations 500 \
  --headless True
echo "==== DONE train $(date '+%Y-%m-%d %H:%M:%S') label=upper9_camera_stable_frozen500 ===="
