#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '=== existing export script ===\n'
sed -n '1,220p' artifacts/autoresearch/export_antisway1000_metrics.py
printf '=== event files ===\n'
find logs/2026-05-07-18-57-33 -type f -name 'events.out.tfevents*' -printf '%p %s\n'
