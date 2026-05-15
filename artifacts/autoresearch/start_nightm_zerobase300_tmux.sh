#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

SESSION="nightm_zerobase300"
LOG="artifacts/autoresearch/nightm_zerobase300_tmux.out"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "SESSION_EXISTS ${SESSION}"
  tmux list-sessions | grep "${SESSION}" || true
  exit 0
fi

rm -f \
  artifacts/autoresearch/shoulder4_night_m_zerobase_dynamic300_full_auto_done.json \
  artifacts/autoresearch/shoulder4_night_m_zerobase_dynamic300_full_auto_failed.json \
  artifacts/autoresearch/shoulder4_night_m_zerobase_dynamic300_web_inputs.tgz

tmux new-session -d -s "${SESSION}" \
  "cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official && bash artifacts/autoresearch/run_nightm_zerobase300_full_auto.sh > ${LOG} 2>&1"

echo "STARTED ${SESSION}"
tmux list-sessions | grep "${SESSION}" || true
