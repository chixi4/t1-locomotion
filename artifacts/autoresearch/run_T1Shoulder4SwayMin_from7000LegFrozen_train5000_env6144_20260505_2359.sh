#!/usr/bin/env bash
set -uo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

RUN_ID=T1Shoulder4SwayMin_from7000LegFrozen_train5000_env6144_20260505_2359
LOG="artifacts/autoresearch/${RUN_ID}.log"
LEG_CHECKPOINT="logs/2026-05-05-11-09-07/nn/model_4000.pth"

{
  echo "train_start=$(date '+%Y-%m-%d_%H:%M:%S_%Z')"
  echo "task=T1Shoulder4SwayMin_from7000LegFrozen_train5000"
  echo "leg_checkpoint=${LEG_CHECKPOINT}"
  echo "num_envs=6144"
  echo "max_iterations=5000"
  echo "note=frozen model7000 12D leg actor; train new 4D shoulder actor; allowed curriculum mask from model7000; sway-based face6 unlock; adaptive_lr disabled to keep LR=3e-4"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  /opt/conda/bin/python -u train_shoulder4_frozen.py \
    --task T1Shoulder4SwayMin_from7000LegFrozen_train5000 \
    --leg_checkpoint "${LEG_CHECKPOINT}" \
    --num_envs 6144 \
    --max_iterations 5000 \
    --headless True
  rc=$?
  echo "train_end=$(date '+%Y-%m-%d_%H:%M:%S_%Z')"
  echo "exit_code=${rc}"
  exit "${rc}"
} 2>&1 | tee "${LOG}"
