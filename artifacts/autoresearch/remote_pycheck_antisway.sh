#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
/opt/conda/bin/python -m py_compile \
  envs/t1.py \
  envs/__init__.py \
  utils/shoulder_runner.py \
  artifacts/autoresearch/generate_fixed_arm_sway_baseline.py
echo py_compile_ok
