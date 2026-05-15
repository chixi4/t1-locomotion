#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
RUN_ID="T1Shoulder4Pitch09Roll08_train1000_env32768_20260507_0915_winproc"
echo "---proc---"
ps -eo pid,ppid,stat,etime,cmd | grep -E "train_shoulder4_frozen|${RUN_ID}|python" | grep -v grep || true
echo "---files---"
ls -lh artifacts/autoresearch/${RUN_ID}.* 2>/dev/null || true
echo "---tail---"
tail -120 artifacts/autoresearch/${RUN_ID}.log 2>/dev/null || true
echo "---windows proc---"
powershell.exe -NoProfile -Command "Get-Process -Id 65684 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime,Responding | Format-List" 2>/dev/null || true
echo "---latest logs---"
ls -td logs/2026-05-07-* 2>/dev/null | head -12 || true
echo "---gpu---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
