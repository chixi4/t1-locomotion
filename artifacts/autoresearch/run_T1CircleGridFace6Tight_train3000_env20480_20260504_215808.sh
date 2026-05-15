#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
echo train_start=2026-05-04_21:58:08_CST
/opt/conda/bin/python -u train.py --task T1CircleGridFace6Tight --num_envs 20480 --max_iterations 3000 --headless True 2>&1 | tee "artifacts/autoresearch/T1CircleGridFace6Tight_train3000_env20480_20260504_215808.log"
echo train_end=2026-05-04_21:58:08_CST
