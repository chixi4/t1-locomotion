#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
echo "---autoload eval/replay files---"
find artifacts/autoresearch -maxdepth 3 -type f | grep -Ei "eval|replay|player|web|shoulder|Pitch09|SatGuard" | sort | sed -n '1,360p'
echo "---root eval scripts---"
find . -maxdepth 3 -type f | grep -Ei "eval|replay|record|player|webgl|shoulder" | sort | sed -n '1,360p'
