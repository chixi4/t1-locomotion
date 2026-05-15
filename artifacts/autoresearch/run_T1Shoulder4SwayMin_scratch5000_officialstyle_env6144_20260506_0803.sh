#!/usr/bin/env bash
set -uo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

RUN_ID=T1Shoulder4SwayMin_scratch5000_officialstyle_env6144_20260506_0803
LOG="artifacts/autoresearch/${RUN_ID}.log"
LEG_CHECKPOINT="logs/2026-05-05-11-09-07/nn/model_4000.pth"

{
  echo "train_start=$(date '+%Y-%m-%d_%H:%M:%S_%Z')"
  echo "task=T1Shoulder4SwayMin_from7000LegFrozen_train5000"
  echo "leg_checkpoint=${LEG_CHECKPOINT}"
  echo "num_envs=6144"
  echo "max_iterations=5000"
  echo "note=scratch shoulder-only run; frozen model7000 leg actor; official-style low LR start; bounded KL adaptive LR; positive reward floor"
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
