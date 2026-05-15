#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

SESSION=t1_satguard2000
RUN_ID=T1Shoulder4SwayMin_satguard2000_from7000_env24576_20260506_1400
LOG="artifacts/autoresearch/${RUN_ID}.log"
PID_FILE="artifacts/autoresearch/${RUN_ID}.pid"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} already exists"
  tmux list-panes -t "${SESSION}" -F '#{pane_pid}' | head -1 > "${PID_FILE}"
  echo "LOG=${LOG}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" bash -lc \
  "cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official && /opt/conda/bin/python -u train_shoulder4_frozen.py --task T1Shoulder4SwayMin_from7000LegFrozen_satguard2000 --num_envs 24576 --max_iterations 2000 --headless True 2>&1 | tee '${LOG}'"

tmux list-panes -t "${SESSION}" -F '#{pane_pid}' | head -1 > "${PID_FILE}"
echo "SESSION=${SESSION}"
echo "PID=$(cat "${PID_FILE}")"
echo "LOG=${LOG}"
