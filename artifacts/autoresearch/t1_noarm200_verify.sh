#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '=== alias ===\n'
grep -n "T1Shoulder4FreeArmNoPenaltyLogstd17" envs/__init__.py
printf '=== config key lines ===\n'
grep -n "run_name\|max_iterations\|num_envs\|logstd_init\|logstd_max\|arm_action_saturation_frac_toler\|anti_sway_vs_fixed_arm\|arm_action_rate\|arm_action_high_freq\|arm_dof_vel\|arm_dof_acc\|arm_torques\|arm_power\|shoulder_neutral_low_speed\|shoulder_roll\|shoulder_pitch_soft_limit\|shoulder_pair_symmetry" envs/T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200.yaml
printf '=== launch ===\n'
sed -n '1,120p' artifacts/autoresearch/run_noarm_penalty_logstd17_200_schtasks_cuda.sh
sed -n '1,80p' artifacts/autoresearch/launch_noarm_penalty_logstd17_200_schtasks.cmd
