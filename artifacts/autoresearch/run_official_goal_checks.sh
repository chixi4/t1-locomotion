#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
mkdir -p artifacts/autoresearch/official_zero_swing_velopp_goal_checks
for iter in 100 200 300 400; do
  echo "=== goal_check model_${iter} ==="
  CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3 artifacts/autoresearch/arm_goal_check.py \
    --task T1Shoulder4OfficialZeroSwingVelOpp_from7000LegFrozen_train400 \
    --arm_checkpoint "logs/2026-05-13-19-47-28/nn/model_${iter}.pth" \
    --leg_checkpoint logs/2026-05-05-11-09-07/nn/model_4000.pth \
    --out "artifacts/autoresearch/official_zero_swing_velopp_goal_checks/model_${iter}_arm_goal_check.json" \
    --num_envs 128 \
    --seconds 5 \
    --warmup_s 1 \
    --event_vel 0.18
done
