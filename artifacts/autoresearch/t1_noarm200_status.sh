#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
RUN_ID="T1Shoulder4FreeArmNoPenaltyLogstd17_train200_env32768_20260507_1857_schtasks_cuda"
LOG="artifacts/autoresearch/${RUN_ID}.log"
printf '=== gpu ===\n'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
printf '=== process ===\n'
ps -eo pid,ppid,cmd | grep -E 'train_shoulder4_frozen|FreeArmNoPenalty|python -u train' | grep -v grep || true
printf '=== log tail ===\n'
tail -80 "${LOG}" || true
printf '=== checkpoints ===\n'
ls -td logs/2026-* 2>/dev/null | head -3
latest=$(ls -td logs/2026-* 2>/dev/null | head -1 || true)
if [ -n "${latest}" ]; then find "${latest}/nn" -maxdepth 1 -type f -name 'model_*.pth' -printf '%f %s\n' | sort -V | tail -5 || true; fi
