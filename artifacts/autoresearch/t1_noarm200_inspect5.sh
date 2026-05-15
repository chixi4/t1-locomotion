#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '=== envs init ===\n'
sed -n '1,220p' envs/__init__.py
printf '=== class refs ===\n'
grep -RIn "class T1Shoulder4\|T1Shoulder4AntiSwayBaseline\|T1Shoulder4Pitch" envs *.py utils 2>/dev/null | head -100 || true
