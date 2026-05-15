#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '=== launch cmd examples ===\n'
find artifacts/autoresearch -maxdepth 1 -type f -name 'launch_*schtasks*.cmd' -printf '%f\n' | sort | tail -8
for f in $(find artifacts/autoresearch -maxdepth 1 -type f -name 'launch_*schtasks*.cmd' | sort | tail -3); do
  printf '\n--- %s ---\n' "$f"
  sed -n '1,160p' "$f"
done
