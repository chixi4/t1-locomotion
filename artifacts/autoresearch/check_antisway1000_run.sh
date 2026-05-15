#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
RUN_ID="T1Shoulder4AntiSwayBaseline_train1000_env32768_20260507_1335_schtasks_cuda"
LOG="artifacts/autoresearch/${RUN_ID}.log"
echo "RUN_ID=${RUN_ID}"
echo "--- process ---"
pgrep -af "train_shoulder4_frozen.py.*T1Shoulder4AntiSwayBaseline" || true
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
echo "--- log tail ---"
if [ -f "$LOG" ]; then
  tail -80 "$LOG"
  echo "--- warnings ---"
  grep -E -i "non-finite|nan|inf|traceback|error|exception" "$LOG" | tail -20 || true
else
  echo "missing log $LOG"
fi
echo "--- checkpoints ---"
find logs -path '*/nn/model_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -10
