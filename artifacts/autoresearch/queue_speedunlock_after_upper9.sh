#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH=/opt/conda/bin:/usr/lib/wsl/lib:${PATH}
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

WAIT_LABEL="upper9_camera_stable_frozen500"
NEXT_LABEL="upper9_camera_stable_openleg18_legresidual_speedunlock300"
WAIT_DONE="artifacts/autoresearch/${WAIT_LABEL}_full_auto_done.json"
WAIT_FAILED="artifacts/autoresearch/${WAIT_LABEL}_full_auto_failed.json"
QUEUE_LOG="artifacts/autoresearch/${NEXT_LABEL}_queue_$(date +%Y%m%d_%H%M%S).log"

{
  echo "==== QUEUED speedunlock after ${WAIT_LABEL} at $(date '+%Y-%m-%d %H:%M:%S') ===="
  while [[ ! -f "${WAIT_DONE}" && ! -f "${WAIT_FAILED}" ]]; do
    echo "waiting for ${WAIT_LABEL}; $(date '+%Y-%m-%d %H:%M:%S')"
    sleep 60
  done
  if [[ -f "${WAIT_FAILED}" ]]; then
    echo "blocked because ${WAIT_FAILED} exists"
    cat "${WAIT_FAILED}"
    exit 1
  fi
  echo "found ${WAIT_DONE}"
  cat "${WAIT_DONE}"
  echo "==== START speedunlock at $(date '+%Y-%m-%d %H:%M:%S') ===="
  bash artifacts/autoresearch/run_openleg18_speedunlock300_progress.sh
} 2>&1 | tee "${QUEUE_LOG}"


