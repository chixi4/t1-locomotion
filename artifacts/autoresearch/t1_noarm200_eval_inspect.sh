#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
sed -n '500,660p' artifacts/autoresearch/eval_shoulder4_frozen.py
