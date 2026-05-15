#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200"
SRC="envs/T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000.yaml"
CFG="envs/${TASK}.yaml"
RUN_ID="T1Shoulder4FreeArmNoPenaltyLogstd17_train200_env32768_20260507_1857"
RUN_SH="artifacts/autoresearch/run_noarm_penalty_logstd17_200_schtasks_cuda.sh"
LAUNCH_CMD="artifacts/autoresearch/launch_noarm_penalty_logstd17_200_schtasks.cmd"

/opt/conda/bin/python - <<'PY'
from pathlib import Path
import yaml

repo = Path('/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official')
task = 'T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200'
src = repo / 'envs/T1Shoulder4AntiSwayBaseline_from7000LegFrozen_train1000.yaml'
dst = repo / f'envs/{task}.yaml'
with src.open('r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg['basic']['checkpoint'] = None
cfg['basic']['leg_checkpoint'] = 'logs/2026-05-05-11-09-07/nn/model_4000.pth'
cfg['basic']['run_name'] = 'T1Shoulder4FreeArmNoPenaltyLogstd17_from7000_train200'
cfg['basic']['max_iterations'] = 200
cfg['basic']['seed'] = 42
cfg['basic']['description'] = (
    'Diagnostic ablation: freeze model7000 leg policy and train 4D shoulders for 200 iterations; '
    'remove all arm-motion penalties, keep anti-sway/tracking/safety rewards, logstd_init=-1.7, logstd_max=-1.0.'
)
cfg['env']['num_envs'] = 32768

alg = cfg['algorithm']
alg['logstd_init'] = -1.7
alg['logstd_min'] = -5.0
alg['logstd_max'] = -1.0
alg['actor_mean_scale_by_dof'] = {
    'Left_Shoulder_Pitch': 0.9,
    'Left_Shoulder_Roll': 0.7,
    'Right_Shoulder_Pitch': 0.9,
    'Right_Shoulder_Roll': 0.7,
}

cmd = cfg['commands']
cmd['arm_action_saturation_frac_toler'] = 1.01

scales = cfg['rewards']['scales']
for name in [
    'arm_action_rate',
    'arm_action_high_freq',
    'arm_dof_vel',
    'arm_dof_acc',
    'arm_torques',
    'arm_power',
    'shoulder_neutral_low_speed',
    'shoulder_roll',
    'shoulder_pitch_soft_limit',
    'shoulder_pair_symmetry',
]:
    scales[name] = 0.0
scales['anti_sway_vs_fixed_arm'] = 8.0

with dst.open('w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

init_path = repo / 'envs/__init__.py'
text = init_path.read_text(encoding='utf-8')
line = f'{task} = T1\n'
if line not in text:
    init_path.write_text(text.rstrip() + '\n' + line, encoding='utf-8')
PY

cat > "${RUN_SH}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

RUN_ID="T1Shoulder4FreeArmNoPenaltyLogstd17_train200_env32768_20260507_1857_schtasks_cuda"
LOG="artifacts/autoresearch/${RUN_ID}.log"
PID_FILE="artifacts/autoresearch/${RUN_ID}.wslpid"

echo "START $(date '+%Y-%m-%d %H:%M:%S') RUN_ID=${RUN_ID}" > "${LOG}"
echo "$$" > "${PID_FILE}"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

exec /opt/conda/bin/python -u train_shoulder4_frozen.py \
  --task T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200 \
  --num_envs 32768 \
  --max_iterations 200 \
  --headless True >> "${LOG}" 2>&1
SH
chmod +x "${RUN_SH}"

cat > "${LAUNCH_CMD}" <<'CMD'
@echo off
set TASK=T1NoArmPenaltyLogstd17_200_20260507_1857
set SCRIPT=/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official/artifacts/autoresearch/run_noarm_penalty_logstd17_200_schtasks_cuda.sh
schtasks /Create /TN "%TASK%" /SC ONCE /ST 23:59 /TR "wsl.exe -e bash %SCRIPT%" /F /RL HIGHEST
schtasks /Run /TN "%TASK%"
CMD

printf 'CONFIG %s\n' "${CFG}"
printf 'RUN_SH %s\n' "${RUN_SH}"
printf 'LAUNCH_CMD %s\n' "${LAUNCH_CMD}"
printf 'INIT_ALIAS '\ngrep -n "${TASK}" envs/__init__.py
printf 'CHECK_CONFIG '\n/opt/conda/bin/python - <<'PY'
import yaml
cfg = yaml.safe_load(open('envs/T1Shoulder4FreeArmNoPenaltyLogstd17_from7000LegFrozen_train200.yaml', encoding='utf-8'))
print(cfg['algorithm']['logstd_init'], cfg['algorithm']['logstd_max'], cfg['commands']['arm_action_saturation_frac_toler'])
print({k: cfg['rewards']['scales'][k] for k in ['anti_sway_vs_fixed_arm','arm_action_rate','arm_action_high_freq','arm_dof_vel','arm_dof_acc','arm_torques','arm_power','shoulder_neutral_low_speed','shoulder_roll','shoulder_pitch_soft_limit','shoulder_pair_symmetry']})
PY
