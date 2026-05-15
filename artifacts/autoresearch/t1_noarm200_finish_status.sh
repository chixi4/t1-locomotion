#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
RUN_ID="T1Shoulder4FreeArmNoPenaltyLogstd17_train200_env32768_20260507_1857_schtasks_cuda"
LOG="artifacts/autoresearch/${RUN_ID}.log"
printf '=== gpu ===\n'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
printf '=== process ===\n'
ps -eo pid,ppid,cmd | grep -E 'train_shoulder4_frozen|FreeArmNoPenalty|python -u train' | grep -v grep || true
printf '=== latest logs ===\n'
ls -td logs/2026-* 2>/dev/null | head -5 || true
printf '=== noarm run checkpoints ===\n'
find logs/2026-05-07-18-57-33/nn -maxdepth 1 -type f -name 'model_*.pth' -printf '%f %s %TY-%Tm-%Td %TH:%TM\n' | sort -V || true
printf '=== log tail ===\n'
tail -120 "$LOG" || true
printf '=== eval script cli ===\n'
/opt/conda/bin/python artifacts/autoresearch/eval_shoulder4_frozen.py --help 2>&1 | tail -80 || true
