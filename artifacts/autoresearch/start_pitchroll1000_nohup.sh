#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
pkill -f train_shoulder4_frozen.py || true
RUN_ID=T1Shoulder4Pitch09Roll08_train1000_env32768_20260507_0900_nohup
LOG="artifacts/autoresearch/${RUN_ID}.log"
PID_FILE="artifacts/autoresearch/${RUN_ID}.pid"
setsid bash -lc "cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official && export PATH=/opt/conda/bin:\$PATH && /opt/conda/bin/python -u train_shoulder4_frozen.py --task T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000 --num_envs 32768 --max_iterations 1000 --headless True > '$LOG' 2>&1" >/dev/null 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
echo "PID=$pid"
echo "LOG=$LOG"
sleep 2
ps -p "$pid" -o pid,stat,etime,cmd || true
