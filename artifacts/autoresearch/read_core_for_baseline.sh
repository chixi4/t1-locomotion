#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

echo "--- t1 reward/curriculum relevant ---"
nl -ba envs/t1.py | sed -n '1,120p'
nl -ba envs/t1.py | sed -n '330,430p'
nl -ba envs/t1.py | sed -n '520,760p'
nl -ba envs/t1.py | sed -n '840,920p'
nl -ba envs/t1.py | sed -n '960,1068p'

echo "--- shoulder runner relevant ---"
nl -ba utils/shoulder_runner.py | sed -n '1,260p'

echo "--- train shoulder relevant ---"
nl -ba train_shoulder4_frozen.py | sed -n '1,220p'

echo "--- current pitchroll config ---"
sed -n '1,260p' envs/T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000.yaml
