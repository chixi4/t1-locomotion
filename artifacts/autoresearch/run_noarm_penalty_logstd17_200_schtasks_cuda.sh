#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

RUN_ID="T1Shoulder4FreeArmNoPenaltyLogstd17_train200_env32768_20260507_1857_schtasks_cuda"
LOG="artifacts/autoresearch/${RUN_ID}.log"
PID_FILE="artifacts/autoresearch/${RUN_ID}.wslpid"

echo "START $(date '+%Y-%m-%d %H:%M:%S') RUN_ID=${RUN_ID}" > "${LOG}"
echo "$$" > "${PID_FILE}"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

exec /opt/conda/bin/python -u train_shoulder4_frozen.py \
  --task T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200 \
  --num_envs 32768 \
  --max_iterations 200 \
  --headless True >> "${LOG}" 2>&1
