#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
find logs/2026-05-07-13-32-44 -type f -printf '%s %p\n' | sort -nr | head -100
