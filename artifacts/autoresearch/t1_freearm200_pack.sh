#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
LABEL="shoulder4_freearm_logstd17_model200"
TAR="artifacts/autoresearch/${LABEL}_web_inputs.tgz"
rm -f "$TAR"
tar -czf "$TAR" \
  "artifacts/autoresearch/${LABEL}_eval" \
  "artifacts/autoresearch/${LABEL}_metrics.json" \
  "artifacts/autoresearch/${LABEL}_fixed_eval_run.log" \
  "artifacts/autoresearch/${LABEL}_random_eval_run.log" \
  "artifacts/autoresearch/T1Shoulder4FreeArmNoPenaltyLogstd17_train200_env32768_20260507_1857_schtasks_cuda.log" \
  "envs/T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200.yaml" \
  "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
  "artifacts/autoresearch/run_eval_freearm_logstd17_model200.sh"
ls -lh "$TAR"
