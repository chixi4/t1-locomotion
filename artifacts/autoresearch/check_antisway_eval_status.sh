#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
LABEL=shoulder4_antisway_baseline_model1000
OUT_DIR=artifacts/autoresearch/${LABEL}_eval
LOG=artifacts/autoresearch/${LABEL}_eval_run.log
echo "--- process ---"
pgrep -af "eval_shoulder4_frozen.py.*${LABEL}" || true
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
echo "--- files ---"
find "$OUT_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort || true
echo "--- log tail ---"
if [ -f "$LOG" ]; then
  tail -120 "$LOG"
  echo "--- warnings ---"
  grep -E -i "traceback|error|exception|nan|inf|broken|closed" "$LOG" | tail -30 || true
else
  echo "missing $LOG"
fi
