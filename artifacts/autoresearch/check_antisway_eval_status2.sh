#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
LABEL=shoulder4_antisway_baseline_model1000
OUT_DIR=artifacts/autoresearch/${LABEL}_eval
echo "--- process ---"
pgrep -af "eval_shoulder4_frozen.py.*${LABEL}" || true
echo "--- files ---"
find "$OUT_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort || true
echo "--- random log tail ---"
tail -120 artifacts/autoresearch/${LABEL}_random_eval_run.log 2>/dev/null || true
echo "--- summary ---"
cat "$OUT_DIR/${LABEL}_summary.json" 2>/dev/null || true
