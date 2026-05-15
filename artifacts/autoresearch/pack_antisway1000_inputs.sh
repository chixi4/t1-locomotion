#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
tar -czf artifacts/autoresearch/antisway1000_web_inputs.tgz \
  logs/2026-05-07-13-32-44/events.out.tfevents.* \
  artifacts/autoresearch/shoulder4_antisway_baseline_model1000_eval \
  artifacts/autoresearch/T1Shoulder4AntiSwayBaseline_train1000_env32768_20260507_1335_schtasks_cuda.log \
  artifacts/autoresearch/fixed_arm_sway_baseline_model7000.pt.json \
  envs/T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000.yaml
ls -lh artifacts/autoresearch/antisway1000_web_inputs.tgz
