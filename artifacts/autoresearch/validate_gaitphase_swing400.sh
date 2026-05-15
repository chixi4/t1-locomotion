#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
chmod +x artifacts/autoresearch/run_gaitphase_swing400_schtasks_cuda.sh artifacts/autoresearch/status_gaitphase_swing400.sh
/opt/conda/bin/python -m py_compile envs/t1.py
/opt/conda/bin/python - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('envs/T1Shoulder4GaitPhaseSwing_from7000LegFrozen_train400.yaml').read_text())
print(cfg['basic']['run_name'], cfg['basic']['max_iterations'], cfg['algorithm']['logstd_init'])
print(cfg['rewards']['scales']['shoulder_gait_phase_pitch'], cfg['rewards']['shoulder_pitch_phase_amp'], cfg['rewards']['shoulder_roll_soft_limit'])
PY
bash -n artifacts/autoresearch/run_gaitphase_swing400_schtasks_cuda.sh
bash -n artifacts/autoresearch/status_gaitphase_swing400.sh
