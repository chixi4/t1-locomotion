#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
OUT=artifacts/autoresearch/shoulder4_pitch09roll08_model1000_web_assets.tgz
tar -czf "$OUT" \
  artifacts/autoresearch/shoulder4_pitch09roll08_model1000_metrics.json \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_pitch09roll08_model1000_fixed_eval.json \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_pitch09roll08_model1000_fixed_eval.csv \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_pitch09roll08_model1000_random_eval.json \
  artifacts/autoresearch/shoulder4_model900_eval/shoulder4_pitch09roll08_model1000_random_replay.json
ls -lh "$OUT"
