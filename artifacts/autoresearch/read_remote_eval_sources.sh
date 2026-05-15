#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
echo "__EVAL__"
sed -n '1,320p' artifacts/autoresearch/eval_shoulder4_frozen.py
echo "__EXTRACT__"
sed -n '1,360p' artifacts/autoresearch/extract_shoulder4_model900_assets.py
echo "__RUNSAT__"
sed -n '1,260p' artifacts/autoresearch/run_eval_satguard2000.sh
