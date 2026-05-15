#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

RUN_ID=T1Shoulder4SwayMin_satguard2000_from7000_env24576_20260506_1354
LOG="artifacts/autoresearch/${RUN_ID}.log"
PID_FILE="artifacts/autoresearch/${RUN_ID}.pid"

nohup /opt/conda/bin/python -u train_shoulder4_frozen.py \
  --task T1Shoulder4SwayMin_from7000LegFrozen_satguard2000 \
  --num_envs 24576 \
  --max_iterations 2000 \
  --headless True \
  > "${LOG}" 2>&1 &

pid=$!
echo "${pid}" > "${PID_FILE}"
echo "PID=${pid}"
echo "LOG=${LOG}"
