#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Upper9CameraStableOfficial_from7000LegFrozen_train500"
LOG="artifacts/autoresearch/${TASK}_$(date +%Y%m%d_%H%M%S)_trainonly.log"
PID_FILE="artifacts/autoresearch/upper9_camera_stable_frozen500_train.pid"

mkdir -p artifacts/autoresearch
nohup bash artifacts/autoresearch/train_upper9_camera500_only.sh > "${LOG}" 2>&1 < /dev/null &
PID="$!"
echo "${PID}" > "${PID_FILE}"
echo "pid=${PID}"
echo "log=${LOG}"
