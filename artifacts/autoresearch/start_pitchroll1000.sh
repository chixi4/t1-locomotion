#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:$PATH
SESSION=t1_pitchroll1000
RUN_ID=T1Shoulder4Pitch09Roll08_train1000_env32768_20260507_$(date +%H%M%S)
LOG="artifacts/autoresearch/${RUN_ID}.log"
PID_FILE="artifacts/autoresearch/${RUN_ID}.pid"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session ${SESSION} already exists"
  tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -1 > "$PID_FILE"
  echo "PID=$(cat "$PID_FILE")"
  echo "LOG=${LOG}"
  exit 0
fi
tmux new-session -d -s "$SESSION" bash -lc "cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official && export PATH=/opt/conda/bin:\$PATH && /opt/conda/bin/python -u train_shoulder4_frozen.py --task T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000 --num_envs 32768 --max_iterations 1000 --headless True 2>&1 | tee '$LOG'"
tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -1 > "$PID_FILE"
echo "SESSION=${SESSION}"
echo "PID=$(cat "$PID_FILE")"
echo "LOG=${LOG}"
