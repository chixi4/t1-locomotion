#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
ls -lt artifacts/autoresearch/test_wsl_detach_*.log 2>/dev/null | head -5 || true
latest="$(ls -t artifacts/autoresearch/test_wsl_detach_*.log 2>/dev/null | head -1 || true)"
if [[ -n "$latest" ]]; then
  echo "---$latest---"
  cat "$latest"
fi
