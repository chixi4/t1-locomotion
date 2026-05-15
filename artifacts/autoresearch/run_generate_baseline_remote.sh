#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PATH=/opt/conda/bin:${PATH}
export CUDA_VISIBLE_DEVICES=0
/opt/conda/bin/python artifacts/autoresearch/generate_fixed_arm_sway_baseline.py \
  --task T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000 \
  --num_envs 4096 \
  --seconds 10.0 \
  --warmup_s 1.0 \
  --seed 123 \
  --output artifacts/autoresearch/fixed_arm_sway_baseline_model7000.pt \
  2>&1 | tee artifacts/autoresearch/fixed_arm_sway_baseline_model7000.log
