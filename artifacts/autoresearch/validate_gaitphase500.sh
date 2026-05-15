#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
chmod +x artifacts/autoresearch/run_gaitphase500_schtasks_cuda.sh artifacts/autoresearch/status_gaitphase500.sh
/opt/conda/bin/python -m py_compile envs/t1.py
/opt/conda/bin/python - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('envs/T1Shoulder4GaitPhaseSoft_from7000LegFrozen_train500.yaml').read_text())
print(cfg['basic']['run_name'], cfg['algorithm']['logstd_init'], cfg['rewards']['scales']['shoulder_gait_phase_pitch'], cfg['rewards']['shoulder_pitch_soft_limit'])
PY
bash -n artifacts/autoresearch/run_gaitphase500_schtasks_cuda.sh
bash -n artifacts/autoresearch/status_gaitphase500.sh
