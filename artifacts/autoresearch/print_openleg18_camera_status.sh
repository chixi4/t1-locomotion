#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Upper9CameraStableOfficialOpenLeg18LegResidualCameraHard_train500"
LABEL="upper9_camera_stable_openleg18_legresidual_camerahard500"
FINAL_ITER="500"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"
FAILED_JSON="artifacts/autoresearch/${LABEL}_full_auto_failed.json"
TRAIN_LOG="$(ls -t artifacts/autoresearch/${TASK}_*_progress.log 2>/dev/null | head -1 || true)"
if [[ -z "${TRAIN_LOG}" ]]; then
  TRAIN_LOG="/dev/null"
fi

PHASE="waiting"
if [[ -f "${DONE_JSON}" ]]; then
  PHASE="done"
elif [[ -f "${FAILED_JSON}" ]]; then
  PHASE="failed"
elif ps -eo cmd | grep -E 'run_openleg18_residual500_progress.sh|train_fullbody_residual.py.*T1Upper9CameraStableOfficialOpenLeg18LegResidualCameraHard' | grep -v grep >/dev/null; then
  PHASE="bc/train"
fi

python3 artifacts/autoresearch/show_openleg18_progress.py \
  --task "${TASK}" \
  --label "${LABEL}" \
  --final-iter "${FINAL_ITER}" \
  --train-log "${TRAIN_LOG}" \
  --phase "${PHASE}"

echo
if [[ -f "${DONE_JSON}" ]]; then
  echo "Done JSON:"
  cat "${DONE_JSON}"
elif [[ -f "${FAILED_JSON}" ]]; then
  echo "Failed JSON:"
  cat "${FAILED_JSON}"
else
  echo "Latest log: ${TRAIN_LOG}"
fi
