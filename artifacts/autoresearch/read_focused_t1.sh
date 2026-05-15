#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
echo "--- init buffers ---"
nl -ba envs/t1.py | sed -n '230,330p'
echo "--- step post physics compute reward ---"
nl -ba envs/t1.py | sed -n '760,920p'
echo "--- curriculum update full ---"
nl -ba envs/t1.py | sed -n '430,635p'
echo "--- runner tail save ---"
nl -ba utils/shoulder_runner.py | sed -n '169,340p'
