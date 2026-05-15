#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '=== nvidia-smi processes ===\n'
nvidia-smi || true
printf '=== reward config ===\n'
sed -n '260,520p' envs/T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000.yaml
printf '=== launch script ===\n'
sed -n '1,240p' artifacts/autoresearch/run_antisway1000_schtasks_cuda.sh || true
printf '=== train scripts ===\n'
find . -maxdepth 3 -type f \( -name '*shoulder*train*.py' -o -name '*anti*sway*.py' -o -name 'train.py' \) | sort | head -80
printf '=== grep refs ===\n'
grep -RIn "arm_action_high_freq\|anti_sway\|logstd_init\|actor_mean_scale\|arm_action_rate\|arm_torques\|shoulder_neutral\|saturation_frac" envs booster_gym artifacts/autoresearch 2>/dev/null | head -260 || true
