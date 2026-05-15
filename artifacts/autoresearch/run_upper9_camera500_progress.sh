#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Upper9CameraStableOfficial_from7000LegFrozen_train500"
LABEL="upper9_camera_stable_frozen500"
DISPLAY_NAME="T1 Upper9 CameraStable Frozen model_500"
FINAL_ITER="500"
LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/isaacgym/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/autoresearch
RUN_ID="${TASK}_$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="artifacts/autoresearch/${RUN_ID}.log"
OUT_DIR="artifacts/autoresearch/${LABEL}_eval"
METRICS_JSON="artifacts/autoresearch/${LABEL}_metrics.json"
WEB_TAR="artifacts/autoresearch/${LABEL}_web_inputs.tgz"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"

rm -f "${WEB_TAR}" "${DONE_JSON}"

echo "==== START train $(date '+%Y-%m-%d %H:%M:%S') task=${TASK} label=${LABEL} ====" | tee "${TRAIN_LOG}"
(
  stdbuf -oL -eL python3 -u train_shoulder4_frozen.py \
    --task "${TASK}" \
    --leg_checkpoint "${LEG_CKPT}" \
    --max_iterations "${FINAL_ITER}" \
    --headless True
  echo "==== DONE train $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ===="
) 2>&1 | tee -a "${TRAIN_LOG}" &
TRAIN_PIPE_PID=$!

while kill -0 "${TRAIN_PIPE_PID}" 2>/dev/null; do
  python3 artifacts/autoresearch/show_upper9_progress.py \
    --task "${TASK}" \
    --label "${LABEL}" \
    --final-iter "${FINAL_ITER}" \
    --train-log "${TRAIN_LOG}" \
    --phase "train"
  sleep 5
done
wait "${TRAIN_PIPE_PID}"

RUN_DIR="$(
  TASK_FOR_PY="${TASK}" FINAL_ITER_FOR_PY="${FINAL_ITER}" python3 - <<'PY'
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
ARM_CKPT="${RUN_DIR}/nn/model_${FINAL_ITER}.pth"
echo "RUN_DIR=${RUN_DIR}" | tee -a "${TRAIN_LOG}"
echo "ARM_CKPT=${ARM_CKPT}" | tee -a "${TRAIN_LOG}"

echo "==== START eval $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ====" | tee -a "${TRAIN_LOG}"
FAST_REPLAY_ONLY=0 bash artifacts/autoresearch/run_shoulder4_eval_pair.sh \
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

echo "==== START metrics $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ====" | tee -a "${TRAIN_LOG}"
python3 artifacts/autoresearch/export_shoulder_metrics.py \
  --run-dir "${RUN_DIR}" \
  --out "${METRICS_JSON}" \
  --run-name "${DISPLAY_NAME}" \
  --checkpoint "${ARM_CKPT}" \
  --note "Frozen 7000 leg, official locomotion rewards, shoulders plus elbow pitch/yaw and waist yaw unlocked; camera-stability reward uses Trunk/head roll-pitch and angular smoothness." 2>&1 | tee -a "${TRAIN_LOG}"

echo "==== START pack $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ====" | tee -a "${TRAIN_LOG}"
tar -czf "${WEB_TAR}" \
  "${OUT_DIR}" \
  "${METRICS_JSON}" \
  "${TRAIN_LOG}" \
  "envs/${TASK}.yaml" \
  "envs/t1.py" \
  "envs/__init__.py" \
  "utils/shoulder_runner.py" \
  "utils/model.py" \
  "resources/T1/T1_locomotion_shoulder4_waist_elbow_softlimit.urdf" \
  "artifacts/autoresearch/run_upper9_camera500_progress.sh" \
  "artifacts/autoresearch/show_upper9_progress.py" \
  "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
  "artifacts/autoresearch/eval_shoulder4_frozen.py" \
  "artifacts/autoresearch/export_shoulder_metrics.py"

LABEL_FOR_PY="${LABEL}" \
DISPLAY_NAME_FOR_PY="${DISPLAY_NAME}" \
TASK_FOR_PY="${TASK}" \
RUN_DIR_FOR_PY="${RUN_DIR}" \
ARM_CKPT_FOR_PY="${ARM_CKPT}" \
LEG_CKPT_FOR_PY="${LEG_CKPT}" \
OUT_DIR_FOR_PY="${OUT_DIR}" \
METRICS_JSON_FOR_PY="${METRICS_JSON}" \
WEB_TAR_FOR_PY="${WEB_TAR}" \
TRAIN_LOG_FOR_PY="${TRAIN_LOG}" \
DONE_JSON_FOR_PY="${DONE_JSON}" \
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

ls -lh "${WEB_TAR}" "${DONE_JSON}" | tee -a "${TRAIN_LOG}"
echo "==== DONE run $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ====" | tee -a "${TRAIN_LOG}"
read -r -p "Training/eval packed. Press Enter to close this window..." _
