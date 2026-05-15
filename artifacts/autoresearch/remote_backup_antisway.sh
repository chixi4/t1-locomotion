#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
backup_dir="artifacts/autoresearch/backups/20260507_antisway_132948"
mkdir -p "$backup_dir/envs" "$backup_dir/utils"
cp envs/t1.py "$backup_dir/envs/t1.py"
cp envs/__init__.py "$backup_dir/envs/__init__.py"
cp utils/shoulder_runner.py "$backup_dir/utils/shoulder_runner.py"
echo "$backup_dir"
