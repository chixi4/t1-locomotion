#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Shoulder4GaitPhaseLateralOutwardLightFoot_from7000LegFrozen_train400"
LABEL="shoulder4_gaitphase_lateral_outward_lightfoot400"
DISPLAY_NAME="T1Shoulder4 LateralOutward LightFoot model_400"
LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"
FINAL_ITER="400"
RUN_ID="${TASK}_$(date +%Y%m%d_%H%M%S)_full_auto"
TRAIN_LOG="artifacts/autoresearch/${RUN_ID}.log"
OUT_DIR="artifacts/autoresearch/${LABEL}_eval"
METRICS_JSON="artifacts/autoresearch/${LABEL}_metrics.json"
WEB_TAR="artifacts/autoresearch/${LABEL}_web_inputs.tgz"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/autoresearch
rm -f "${WEB_TAR}" "${DONE_JSON}"

phase_start() {
  PHASE_NAME="$1"
  PHASE_T0=$(date +%s)
  echo "==== START ${PHASE_NAME} $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ====" | tee -a "${TRAIN_LOG}"
}

phase_done() {
  local now
  now=$(date +%s)
  echo "==== DONE ${PHASE_NAME} $(date '+%Y-%m-%d %H:%M:%S') elapsed=$((now - PHASE_T0))s label=${LABEL} ====" | tee -a "${TRAIN_LOG}"
}

echo "START full_auto $(date '+%Y-%m-%d %H:%M:%S') task=${TASK}" | tee "${TRAIN_LOG}"
phase_start train
stdbuf -oL -eL python3 train_shoulder4_frozen.py --task "${TASK}" 2>&1 | tee -a "${TRAIN_LOG}"
phase_done

RUN_DIR="$(python3 - <<'PY'
from pathlib import Path
import yaml

task = "T1Shoulder4GaitPhaseLateralOutwardLightFoot_from7000LegFrozen_train400"
final_iter = "400"
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
ARM_CKPT="${RUN_DIR}/nn/model_${FINAL_ITER}.pth"
echo "RUN_DIR=${RUN_DIR}" | tee -a "${TRAIN_LOG}"
echo "ARM_CKPT=${ARM_CKPT}" | tee -a "${TRAIN_LOG}"

phase_start eval
bash artifacts/autoresearch/run_shoulder4_eval_pair.sh \
  --task "${TASK}" \
  --label "${LABEL}" \
  --arm_checkpoint "${ARM_CKPT}" \
  --leg_checkpoint "${LEG_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --num_envs 256 \
  --fixed_duration_s 8 \
  --warmup_s 2 \
  --random_seconds 60 \
  --fps 50 \
  --clean 2>&1 | tee -a "${TRAIN_LOG}"
phase_done

phase_start metrics
python3 artifacts/autoresearch/export_shoulder_metrics.py \
  --run-dir "${RUN_DIR}" \
  --out "${METRICS_JSON}" \
  --run-name "${DISPLAY_NAME}" \
  --checkpoint "${ARM_CKPT}" \
  --note "Lateral outward light-foot arm design: sagittal foot-x pitch swing only for vx/yaw, lateral roll outward with left_sign=+1 right_sign=-1, base amp 0.09, foot-side extra 0.20, roll max 0.30, asymmetric roll target/hard limits inward 0.14 rad and outward 0.30 rad, and lateral pitch-down same-angle constraint." 2>&1 | tee -a "${TRAIN_LOG}"
phase_done

phase_start pack
if command -v pigz >/dev/null 2>&1; then
  TAR_COMPRESS=(--use-compress-program "pigz -1")
else
  TAR_COMPRESS=(-I "gzip -1")
fi
tar "${TAR_COMPRESS[@]}" -cf "${WEB_TAR}" \
  "${OUT_DIR}" \
  "${METRICS_JSON}" \
  "${TRAIN_LOG}" \
  "envs/${TASK}.yaml" \
  "envs/t1.py" \
  "envs/__init__.py" \
  "utils/shoulder_runner.py" \
  "utils/model.py" \
  "resources/T1/T1_locomotion_shoulder4_lateral_roll_lightfoot_limit.urdf" \
  "artifacts/autoresearch/run_lateral_outward_lightfoot400_full_auto.sh" \
  "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
  "artifacts/autoresearch/eval_shoulder4_frozen.py" \
  "artifacts/autoresearch/export_shoulder_metrics.py"
phase_done

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "label": "${LABEL}",
    "display_name": "${DISPLAY_NAME}",
    "task": "${TASK}",
    "run_dir": "${RUN_DIR}",
    "arm_checkpoint": "${ARM_CKPT}",
    "leg_checkpoint": "${LEG_CKPT}",
    "eval_dir": "${OUT_DIR}",
    "metrics_json": "${METRICS_JSON}",
    "web_inputs": "${WEB_TAR}",
    "train_log": "${TRAIN_LOG}",
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
Path("${DONE_JSON}").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

ls -lh "${WEB_TAR}" "${DONE_JSON}" | tee -a "${TRAIN_LOG}"
echo "DONE $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL}" | tee -a "${TRAIN_LOG}"
