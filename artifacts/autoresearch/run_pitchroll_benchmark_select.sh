set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:$PATH
pkill -f train_shoulder4_frozen.py || true
/opt/conda/bin/python -u artifacts/autoresearch/benchmark_pitchroll_envs_select.py
