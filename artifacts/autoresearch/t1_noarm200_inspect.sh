#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '=== gpu ===\n'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
printf '=== train processes ===\n'
ps -eo pid,ppid,cmd | grep -E 'train.py|booster|python.*train|T1Shoulder' | grep -v grep | head -30 || true
printf '=== latest logs ===\n'
ls -td logs/2026-* 2>/dev/null | head -8 || true
printf '=== config head ===\n'
sed -n '1,260p' envs/T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000.yaml
printf '=== reward/logstd refs ===\n'
rg -n "arm_action_high_freq|anti_sway|initial_logstd|logstd|actor_mean_scale|shoulder_pair|shoulder_neutral|arm_action_rate|frozen_leg|leg_model|T1Shoulder4" envs booster_gym artifacts/autoresearch -g '*.py' -g '*.yaml' | head -220 || true
