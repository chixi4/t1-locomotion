#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
LABEL="shoulder4_freearm_logstd17_model200"
printf '=== gpu ===\n'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
printf '=== eval process ===\n'
ps -eo pid,ppid,cmd | grep -E 'eval_shoulder4_frozen|freearm_logstd17|python -u artifacts/autoresearch/eval' | grep -v grep || true
printf '=== fixed tail ===\n'
tail -60 "artifacts/autoresearch/${LABEL}_fixed_eval_run.log" || true
printf '=== random tail ===\n'
tail -30 "artifacts/autoresearch/${LABEL}_random_eval_run.log" || true
printf '=== out files ===\n'
ls -lh "artifacts/autoresearch/${LABEL}_eval" 2>/dev/null || true
