#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:/usr/lib/wsl/lib:$PATH
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

LABEL=shoulder4_antisway_baseline_model1000
TASK=T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000
ARM_CKPT=logs/2026-05-07-13-32-44/nn/model_1000.pth
LEG_CKPT=logs/2026-05-05-11-09-07/nn/model_4000.pth
OUT_DIR=artifacts/autoresearch/${LABEL}_eval
LOG=artifacts/autoresearch/${LABEL}_eval_run.log

{
  /opt/conda/bin/python -u artifacts/autoresearch/eval_shoulder4_frozen.py \
    --clean \
    --num_envs 256 \
    --label "${LABEL}" \
    --task "${TASK}" \
    --arm_checkpoint "${ARM_CKPT}" \
    --leg_checkpoint "${LEG_CKPT}" \
    --out_dir "${OUT_DIR}" \
    --mode both \
    --random_seconds 60
} 2>&1 | tee "${LOG}"
