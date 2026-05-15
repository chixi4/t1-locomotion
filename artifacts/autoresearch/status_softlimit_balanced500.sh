#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

RUN_ID="T1Shoulder4SoftLimitBalanced_train500_env32768_20260507_2100_schtasks_cuda"
LOG="artifacts/autoresearch/${RUN_ID}.log"
PID_FILE="artifacts/autoresearch/${RUN_ID}.wslpid"

echo "RUN_ID=${RUN_ID}"
if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  echo "pid=${PID}"
  ps -p "${PID}" -o pid,etime,cmd || true
else
  echo "pid_file_missing=${PID_FILE}"
fi

echo
echo "latest checkpoints:"
find logs -path "*/nn/model_*.pth" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -8 | cut -d' ' -f2-

echo
echo "last log lines:"
if [[ -f "${LOG}" ]]; then
  tail -40 "${LOG}"
else
  echo "log_missing=${LOG}"
fi
