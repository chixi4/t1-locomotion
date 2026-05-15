#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
out="artifacts/autoresearch/test_wsl_detach_$(date +%Y%m%d_%H%M%S).log"
echo "hello from wsl pid=$$ date=$(date '+%Y-%m-%d %H:%M:%S')" > "$out"
sleep 5
echo "done date=$(date '+%Y-%m-%d %H:%M:%S')" >> "$out"
