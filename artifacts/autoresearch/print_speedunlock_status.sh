#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

LABEL="upper9_camera_stable_openleg18_legresidual_speedunlock300"
TASK="T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedUnlock_train300"
DONE="artifacts/autoresearch/${LABEL}_full_auto_done.json"
FAILED="artifacts/autoresearch/${LABEL}_full_auto_failed.json"

echo "label=${LABEL}"
echo "task=${TASK}"
date "+now=%Y-%m-%d %H:%M:%S"
echo
echo "process:"
ps -eo pid,etime,cmd | grep -E "queue_speedunlock|run_openleg18_speedunlock|train_fullbody_residual.py --task ${TASK}" | grep -v grep || true
echo
echo "gpu:"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader || true
echo
echo "done:"
if [[ -f "${DONE}" ]]; then cat "${DONE}"; else echo "not yet"; fi
echo
echo "failed:"
if [[ -f "${FAILED}" ]]; then cat "${FAILED}"; else echo "not yet"; fi
echo
echo "latest log:"
latest="$(ls -t artifacts/autoresearch/${TASK}_*_progress.log 2>/dev/null | head -1 || true)"
if [[ -n "${latest}" ]]; then
  echo "${latest}"
  tail -30 "${latest}"
else
  queue_log="$(ls -t artifacts/autoresearch/${LABEL}_queue_*.log 2>/dev/null | head -1 || true)"
  if [[ -n "${queue_log}" ]]; then
    echo "${queue_log}"
    tail -30 "${queue_log}"
  else
    echo "none"
  fi
fi


