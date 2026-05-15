#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200"
LABEL="shoulder4_freearm_logstd17_model200"
ARM_CKPT="logs/2026-05-07-18-57-33/nn/model_200.pth"
LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"
OUT_DIR="artifacts/autoresearch/${LABEL}_eval"
LOG_FIXED="artifacts/autoresearch/${LABEL}_fixed_eval_run.log"
LOG_RANDOM="artifacts/autoresearch/${LABEL}_random_eval_run.log"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "START fixed $(date '+%Y-%m-%d %H:%M:%S')" | tee "${LOG_FIXED}"
/opt/conda/bin/python -u artifacts/autoresearch/eval_shoulder4_frozen.py \
  --task "${TASK}" \
  --arm_checkpoint "${ARM_CKPT}" \
  --leg_checkpoint "${LEG_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --label "${LABEL}" \
  --num_envs 256 \
  --fixed_duration_s 10 \
  --warmup_s 1 \
  --mode fixed >> "${LOG_FIXED}" 2>&1

echo "START random $(date '+%Y-%m-%d %H:%M:%S')" | tee "${LOG_RANDOM}"
/opt/conda/bin/python -u artifacts/autoresearch/eval_shoulder4_frozen.py \
  --task "${TASK}" \
  --arm_checkpoint "${ARM_CKPT}" \
  --leg_checkpoint "${LEG_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --label "${LABEL}" \
  --num_envs 256 \
  --fixed_duration_s 10 \
  --warmup_s 1 \
  --random_seconds 60 \
  --fps 50 \
  --mode random >> "${LOG_RANDOM}" 2>&1

echo "DONE $(date '+%Y-%m-%d %H:%M:%S')"
ls -lh "${OUT_DIR}"
cat "${OUT_DIR}/${LABEL}_summary.json"
