#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Shoulder4GaitPhaseNightLFourGroupClean_from7000LegFrozen_train400"
RUN_DIR="logs/2026-05-12-17-50-29"
ARM_CKPT="${RUN_DIR}/nn/model_300.pth"
LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"
LABEL="shoulder4_night_l_4group_clean300"
DISPLAY_NAME="T1Shoulder4 NightL FourGroupClean model_300"
COMMAND_TEXT="Night L 300：四组肩部奖励中途结果。静止下垂、动态侧移伸手、防贴边偷懒、左右平衡平滑；用于提前查看 model_300 的侧移伸手幅度。"
OUT_DIR="artifacts/autoresearch/${LABEL}_eval"
METRICS_JSON="artifacts/autoresearch/${LABEL}_metrics.json"
WEB_TAR="artifacts/autoresearch/${LABEL}_web_inputs.tgz"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"
EVAL_LOG="artifacts/autoresearch/${LABEL}_eval_now.log"
TRAIN_LOG="artifacts/autoresearch/T1Shoulder4GaitPhaseNightLFourGroupClean_from7000LegFrozen_train400_20260512_174938_full_auto.log"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${ARM_CKPT}" ]]; then
  echo "missing checkpoint ${ARM_CKPT}" >&2
  exit 2
fi

TRAIN_PID="$(pgrep -f "train_shoulder4_frozen.py --task ${TASK}" | head -1 || true)"
resume_train() {
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    kill -CONT "${TRAIN_PID}" >/dev/null 2>&1 || true
    echo "resumed training pid=${TRAIN_PID}" | tee -a "${EVAL_LOG}" >/dev/null
  fi
}
trap resume_train EXIT

mkdir -p artifacts/autoresearch
rm -f "${WEB_TAR}" "${DONE_JSON}"
echo "START NightL model_300 eval $(date '+%Y-%m-%d %H:%M:%S') ckpt=${ARM_CKPT}" | tee "${EVAL_LOG}"

if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
  echo "pausing training pid=${TRAIN_PID}" | tee -a "${EVAL_LOG}"
  kill -STOP "${TRAIN_PID}"
fi

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
  --clean 2>&1 | tee -a "${EVAL_LOG}"

python3 artifacts/autoresearch/export_shoulder_metrics.py \
  --run-dir "${RUN_DIR}" \
  --out "${METRICS_JSON}" \
  --run-name "${DISPLAY_NAME}" \
  --checkpoint "${ARM_CKPT}" \
  --note "${COMMAND_TEXT}" 2>&1 | tee -a "${EVAL_LOG}"

URDF_FILE="$(python3 - "${TASK}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("envs", f"{sys.argv[1]}.yaml").read_text(encoding="utf-8"))
print(cfg["asset"]["file"])
PY
)"

if command -v pigz >/dev/null 2>&1; then
  TAR_COMPRESS=(--use-compress-program "pigz -1")
else
  TAR_COMPRESS=(-I "gzip -1")
fi
tar "${TAR_COMPRESS[@]}" -cf "${WEB_TAR}" \
  "${OUT_DIR}" \
  "${METRICS_JSON}" \
  "${EVAL_LOG}" \
  "${TRAIN_LOG}" \
  "envs/${TASK}.yaml" \
  "envs/t1.py" \
  "envs/__init__.py" \
  "utils/shoulder_runner.py" \
  "utils/model.py" \
  "${URDF_FILE}" \
  "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
  "artifacts/autoresearch/eval_shoulder4_frozen.py" \
  "artifacts/autoresearch/export_shoulder_metrics.py"

python3 - "${LABEL}" "${DISPLAY_NAME}" "${TASK}" "${RUN_DIR}" "${ARM_CKPT}" "${LEG_CKPT}" "${OUT_DIR}" "${METRICS_JSON}" "${WEB_TAR}" "${EVAL_LOG}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

keys = [
    "label",
    "display_name",
    "task",
    "run_dir",
    "arm_checkpoint",
    "leg_checkpoint",
    "eval_dir",
    "metrics_json",
    "web_inputs",
    "eval_log",
]
payload = dict(zip(keys, sys.argv[1:]))
payload["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
Path(f"artifacts/autoresearch/{payload['label']}_full_auto_done.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

ls -lh "${WEB_TAR}" "${DONE_JSON}" | tee -a "${EVAL_LOG}"
echo "DONE NightL model_300 eval $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${EVAL_LOG}"
