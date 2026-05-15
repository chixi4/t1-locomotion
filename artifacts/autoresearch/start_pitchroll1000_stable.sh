#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
SESSION=t1_pitchroll1000
RUN_ID=T1Shoulder4Pitch09Roll08_train1000_env32768_20260507_0900
PID_FILE="artifacts/autoresearch/${RUN_ID}.pid"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi
pkill -f train_shoulder4_frozen.py || true
tmux new-session -d -s "$SESSION" "bash -lc /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official/artifacts/autoresearch/run_pitchroll1000_inner.sh"
tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -1 > "$PID_FILE"
echo "SESSION=${SESSION}"
echo "PID=$(cat "$PID_FILE")"
echo "LOG=artifacts/autoresearch/${RUN_ID}.log"
