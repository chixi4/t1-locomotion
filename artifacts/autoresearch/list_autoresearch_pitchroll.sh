#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
echo "---files---"
find artifacts/autoresearch -maxdepth 3 -iname "*pitchroll*" -o -iname "*Pitch09*" | sort | sed -n '1,240p'
echo "---benchmark scripts---"
for f in artifacts/autoresearch/benchmark_pitchroll_envs.py artifacts/autoresearch/benchmark_pitchroll_envs_fine.py artifacts/autoresearch/benchmark_pitchroll_envs_select.py; do
  echo "### $f"
  [[ -f "$f" ]] && sed -n '1,260p' "$f" || echo missing
done
