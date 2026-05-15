#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
/opt/conda/bin/python -u train_shoulder4_frozen.py \
  --task T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000 \
  --num_envs 128 \
  --max_iterations 1 \
  --headless True
