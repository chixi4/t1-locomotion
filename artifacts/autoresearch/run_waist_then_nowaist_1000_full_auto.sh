#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"
FINAL_ITER="1000"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/autoresearch

run_one() {
  local task="$1"
  local label="$2"
  local display_name="$3"
  local note="$4"

  local run_id="${task}_$(date +%Y%m%d_%H%M%S)_full_auto"
  local train_log="artifacts/autoresearch/${run_id}.log"
  local out_dir="artifacts/autoresearch/${label}_eval"
  local metrics_json="artifacts/autoresearch/${label}_metrics.json"
  local web_tar="artifacts/autoresearch/${label}_web_inputs.tgz"
  local done_json="artifacts/autoresearch/${label}_full_auto_done.json"

  rm -f "${web_tar}" "${done_json}"

  echo "START train $(date '+%Y-%m-%d %H:%M:%S') task=${task}" | tee "${train_log}"
  stdbuf -oL -eL python3 train_shoulder4_frozen.py --task "${task}" 2>&1 | tee -a "${train_log}"

  local run_dir
  run_dir="$(
    TASK_FOR_PY="${task}" FINAL_ITER_FOR_PY="${FINAL_ITER}" python3 - <<'PY'
from pathlib import Path
import os
import yaml

task = os.environ["TASK_FOR_PY"]
final_iter = os.environ["FINAL_ITER_FOR_PY"]
candidates = []
for cfg_path in Path("logs").glob("*/config.yaml"):
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    ckpt = cfg_path.parent / "nn" / f"model_{final_iter}.pth"
    if cfg.get("basic", {}).get("task") == task and ckpt.exists():
        candidates.append(cfg_path.parent)
if not candidates:
    raise SystemExit(f"no completed run dir found for {task}")
print(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
PY
  )"
  local arm_ckpt="${run_dir}/nn/model_${FINAL_ITER}.pth"
  echo "RUN_DIR=${run_dir}" | tee -a "${train_log}"
  echo "ARM_CKPT=${arm_ckpt}" | tee -a "${train_log}"

  echo "START eval $(date '+%Y-%m-%d %H:%M:%S') label=${label}" | tee -a "${train_log}"
  bash artifacts/autoresearch/run_shoulder4_eval_pair.sh \
    --task "${task}" \
    --label "${label}" \
    --arm_checkpoint "${arm_ckpt}" \
    --leg_checkpoint "${LEG_CKPT}" \
    --out_dir "${out_dir}" \
    --num_envs 256 \
    --fixed_duration_s 8 \
    --warmup_s 2 \
    --random_seconds 60 \
    --fps 50 \
    --clean 2>&1 | tee -a "${train_log}"

  echo "START metrics $(date '+%Y-%m-%d %H:%M:%S') label=${label}" | tee -a "${train_log}"
  python3 artifacts/autoresearch/export_shoulder_metrics.py \
    --run-dir "${run_dir}" \
    --out "${metrics_json}" \
    --run-name "${display_name}" \
    --checkpoint "${arm_ckpt}" \
    --note "${note}" 2>&1 | tee -a "${train_log}"

  echo "START pack $(date '+%Y-%m-%d %H:%M:%S') label=${label}" | tee -a "${train_log}"
  tar -czf "${web_tar}" \
    "${out_dir}" \
    "${metrics_json}" \
    "${train_log}" \
    "envs/${task}.yaml" \
    "envs/t1.py" \
    "envs/__init__.py" \
    "utils/shoulder_runner.py" \
    "utils/model.py" \
    "resources/T1/T1_locomotion_shoulder4_waist_yaw_softlimit.urdf" \
    "artifacts/autoresearch/run_waist_then_nowaist_1000_full_auto.sh" \
    "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
    "artifacts/autoresearch/eval_shoulder4_frozen.py" \
    "artifacts/autoresearch/export_shoulder_metrics.py"

  LABEL_FOR_PY="${label}" \
  DISPLAY_NAME_FOR_PY="${display_name}" \
  TASK_FOR_PY="${task}" \
  RUN_DIR_FOR_PY="${run_dir}" \
  ARM_CKPT_FOR_PY="${arm_ckpt}" \
  LEG_CKPT_FOR_PY="${LEG_CKPT}" \
  OUT_DIR_FOR_PY="${out_dir}" \
  METRICS_JSON_FOR_PY="${metrics_json}" \
  WEB_TAR_FOR_PY="${web_tar}" \
  TRAIN_LOG_FOR_PY="${train_log}" \
  DONE_JSON_FOR_PY="${done_json}" \
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "label": os.environ["LABEL_FOR_PY"],
    "display_name": os.environ["DISPLAY_NAME_FOR_PY"],
    "task": os.environ["TASK_FOR_PY"],
    "run_dir": os.environ["RUN_DIR_FOR_PY"],
    "arm_checkpoint": os.environ["ARM_CKPT_FOR_PY"],
    "leg_checkpoint": os.environ["LEG_CKPT_FOR_PY"],
    "eval_dir": os.environ["OUT_DIR_FOR_PY"],
    "metrics_json": os.environ["METRICS_JSON_FOR_PY"],
    "web_inputs": os.environ["WEB_TAR_FOR_PY"],
    "train_log": os.environ["TRAIN_LOG_FOR_PY"],
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
Path(os.environ["DONE_JSON_FOR_PY"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

  ls -lh "${web_tar}" "${done_json}" | tee -a "${train_log}"
  echo "DONE $(date '+%Y-%m-%d %H:%M:%S') label=${label}" | tee -a "${train_log}"
}

run_one \
  "T1Shoulder4GaitPhaseSignFixAmp4LowDownWaist_from7000LegFrozen_train1000" \
  "shoulder4_gaitphase_signfix_amp4_lowdown_waist1000" \
  "T1Shoulder4 SignFix Amp4LowDown Waist model_1000" \
  "腰部 yaw 解锁：hard range ±0.22 rad，action clip ±0.16 rad，soft limit 0.12 rad；shoulder_foot_phase_pitch=-2.0，amp=0.72，sign=-1.0，shoulder_gait_phase_pitch=0，静止同角度下垂范围 0.015-0.05。"

run_one \
  "T1Shoulder4GaitPhaseSignFixAmp4LowDown_from7000LegFrozen_train1000" \
  "shoulder4_gaitphase_signfix_amp4_lowdown_nowaist1000" \
  "T1Shoulder4 SignFix Amp4LowDown NoWaist model_1000" \
  "不解锁腰部对照组；shoulder_foot_phase_pitch=-2.0，amp=0.72，sign=-1.0，shoulder_gait_phase_pitch=0，shoulder_pitch_soft_limit=0.38，静止同角度下垂范围 0.015-0.05。"
