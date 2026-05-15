#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

RUN_ID=T1CircleGridFace6Tight15_from3000_train4000_env24576_20260505_1112
LOG="artifacts/autoresearch/${RUN_ID}.log"
CHECKPOINT="logs/2026-05-04-21-58-46/nn/model_3000.pth"

{
  echo "train_start=$(date '+%Y-%m-%d_%H:%M:%S_%Z')"
  echo "task=T1CircleGridFace6Tight15"
  echo "checkpoint=${CHECKPOINT}"
  echo "num_envs=24576"
  echo "max_iterations=4000"
  echo "note=continuation run; 31x31x31 grid, linear/yaw max 1.5, face6 unlock"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  /opt/conda/bin/python -u train.py \
    --task T1CircleGridFace6Tight15 \
    --checkpoint "${CHECKPOINT}" \
    --num_envs 24576 \
    --max_iterations 4000 \
    --headless True
  rc=$?
  echo "train_end=$(date '+%Y-%m-%d_%H:%M:%S_%Z')"
  echo "exit_code=${rc}"
  exit "${rc}"
} 2>&1 | tee "${LOG}"
