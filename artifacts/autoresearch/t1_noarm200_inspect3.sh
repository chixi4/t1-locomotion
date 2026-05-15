#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '=== model.py ===\n'
sed -n '1,120p' utils/model.py
printf '=== shoulder_runner init/update ===\n'
sed -n '1,130p' utils/shoulder_runner.py
sed -n '250,380p' utils/shoulder_runner.py
printf '=== train_shoulder script ===\n'
sed -n '1,220p' train_shoulder4_frozen.py
