#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
CHAIN=artifacts/autoresearch/run_symfix_foot2_norm400_full_auto.sh

cd "$ROOT"
mkdir -p artifacts/autoresearch

RUN_ID=shoulder4_gaitphase_symfix_foot2_norm400_driver_$(date +%Y%m%d_%H%M%S)
LOG=artifacts/autoresearch/${RUN_ID}.log
PID=artifacts/autoresearch/${RUN_ID}.pid

setsid -f bash -lc "cd '$ROOT' && exec </dev/null > '$LOG' 2>&1 && echo \$\$ > '$PID' && bash '$CHAIN'"

sleep 2
echo "RUN_ID=$RUN_ID"
echo "LOG=$LOG"
printf 'PID='
cat "$PID"
echo
echo "TAIL"
tail -30 "$LOG" || true
