set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:$PATH
/opt/conda/bin/python -u artifacts/autoresearch/benchmark_pitchroll_envs_fine.py
