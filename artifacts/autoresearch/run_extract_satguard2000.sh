#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:$PATH
/opt/conda/bin/python artifacts/autoresearch/extract_satguard2000_metrics.py
