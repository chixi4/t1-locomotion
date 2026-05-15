#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:$PATH
RUN_ID=T1Shoulder4Pitch09Roll08_train1000_env32768_20260507_0900
LOG="artifacts/autoresearch/${RUN_ID}.log"
exec /opt/conda/bin/python -u train_shoulder4_frozen.py \
  --task T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000 \
  --num_envs 32768 \
  --max_iterations 1000 \
  --headless True \
  2>&1 | tee "$LOG"
