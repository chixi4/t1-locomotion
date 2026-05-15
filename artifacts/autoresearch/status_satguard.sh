#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

echo "--- session ---"
tmux has-session -t t1_satguard2000 2>/dev/null && echo tmux_running || echo tmux_missing
ps -eo pid,ppid,stat,etime,cmd | grep -E 'train_shoulder4_frozen|t1_satguard2000|python' | grep -v grep || true

echo "--- log_tail ---"
tail -35 artifacts/autoresearch/T1Shoulder4SwayMin_satguard2000_from7000_env24576_20260506_1400.log || true

echo "--- snapshot ---"
/opt/conda/bin/python artifacts/autoresearch/snapshot_satguard.py || true

echo "--- checkpoints ---"
ls -lh logs/2026-05-06-13-58-17/nn | tail -12 || true

echo "--- gpu ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || true
