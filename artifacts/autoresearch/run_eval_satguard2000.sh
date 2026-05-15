#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:$PATH

LABEL=shoulder4_satguard2000_model2000
TASK=T1Shoulder4SwayMin_from7000LegFrozen_satguard2000
ARM_CKPT=logs/2026-05-06-13-58-17/nn/model_2000.pth
LEG_CKPT=logs/2026-05-05-11-09-07/nn/model_4000.pth
LOG=artifacts/autoresearch/${LABEL}_eval_run.log

{
  echo "== fixed eval =="
  /opt/conda/bin/python -u artifacts/autoresearch/eval_shoulder4_frozen.py \
    --clean \
    --num_envs 256 \
    --label "${LABEL}" \
    --task "${TASK}" \
    --arm_checkpoint "${ARM_CKPT}" \
    --leg_checkpoint "${LEG_CKPT}" \
    --mode fixed

  echo "== random replay eval =="
  /opt/conda/bin/python -u artifacts/autoresearch/eval_shoulder4_frozen.py \
    --clean \
    --num_envs 256 \
    --label "${LABEL}" \
    --task "${TASK}" \
    --arm_checkpoint "${ARM_CKPT}" \
    --leg_checkpoint "${LEG_CKPT}" \
    --mode random \
    --random_seconds 60
} 2>&1 | tee "${LOG}"
