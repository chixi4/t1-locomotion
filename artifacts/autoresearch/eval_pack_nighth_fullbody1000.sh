#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK=T1Shoulder4NightHFullBodyGrid15Scratch_train5000
ITER="${ITER:-1000}"
LABEL=shoulder4_nighth_fullbody_grid15_scratch1000
DISPLAY_NAME="T1Shoulder4 NightH FullBody Grid15 Scratch model_${ITER}"
NOTE="Manual paused checkpoint preview at model_${ITER}; full-body scratch fusion of T1CircleGridFace6Tight15 grid3d_circle face6 curriculum with NightH NoInwardBoundary shoulder rewards."
COMMAND_TEXT="从零训练到 model_${ITER} 的中途预览；腿部使用 grid3d_circle 1.5m/s、face6 逐格解锁课程，肩部使用 NightH NoInwardBoundary：roll target_clip/URDF 禁止内收，侧移时最小外展 margin，低速保持下垂 pitch。"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/autoresearch
EVAL_LOG="artifacts/autoresearch/${LABEL}_manual_eval_$(date +%Y%m%d_%H%M%S).log"
OUT_DIR="artifacts/autoresearch/${LABEL}_eval"
METRICS_JSON="artifacts/autoresearch/${LABEL}_metrics.json"
WEB_TAR="artifacts/autoresearch/${LABEL}_web_inputs.tgz"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"

find_run_dir() {
  python3 - "$TASK" "$ITER" <<'PY'
from pathlib import Path
import sys
import yaml

task = sys.argv[1]
iteration = sys.argv[2]
candidates = []
for cfg_path in Path("logs").glob("*/config.yaml"):
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    ckpt = cfg_path.parent / "nn" / f"model_{iteration}.pth"
    if cfg.get("basic", {}).get("task") == task and ckpt.exists():
        candidates.append(cfg_path.parent)
if not candidates:
    raise SystemExit(f"no run dir found for {task} model_{iteration}")
print(sorted(candidates, key=lambda p: (p / "nn" / f"model_{iteration}.pth").stat().st_mtime)[-1])
PY
}

asset_file() {
  python3 - "$TASK" <<'PY'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("envs", f"{sys.argv[1]}.yaml").read_text(encoding="utf-8"))
print(cfg["asset"]["file"])
PY
}

RUN_DIR="$(find_run_dir)"
CKPT="${RUN_DIR}/nn/model_${ITER}.pth"
URDF_FILE="$(asset_file)"
MAX_STEP=$((ITER - 1))

rm -rf "${OUT_DIR}"
rm -f "${METRICS_JSON}" "${WEB_TAR}" "${DONE_JSON}"

{
  echo "START manual eval label=${LABEL} task=${TASK} iter=${ITER} $(date '+%Y-%m-%d %H:%M:%S')"
  echo "RUN_DIR=${RUN_DIR}"
  echo "CHECKPOINT=${CKPT}"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true

  python3 artifacts/autoresearch/eval_fullbody.py \
    --task "${TASK}" \
    --label "${LABEL}" \
    --checkpoint "${CKPT}" \
    --out_dir "${OUT_DIR}" \
    --num_envs 256 \
    --fixed_duration_s 8 \
    --warmup_s 2 \
    --random_seconds 60 \
    --fps 50 \
    --clean

  python3 artifacts/autoresearch/export_shoulder_metrics.py \
    --run-dir "${RUN_DIR}" \
    --out "${METRICS_JSON}" \
    --run-name "${DISPLAY_NAME}" \
    --checkpoint "${CKPT}" \
    --max-step "${MAX_STEP}" \
    --note "${NOTE} Command: ${COMMAND_TEXT}"

  if command -v pigz >/dev/null 2>&1; then
    TAR_COMPRESS=(--use-compress-program "pigz -1")
  else
    TAR_COMPRESS=(-I "gzip -1")
  fi
  tar "${TAR_COMPRESS[@]}" -cf "${WEB_TAR}" \
    "${OUT_DIR}" \
    "${METRICS_JSON}" \
    "${EVAL_LOG}" \
    "envs/${TASK}.yaml" \
    "envs/t1.py" \
    "envs/__init__.py" \
    "utils/model.py" \
    "${URDF_FILE}" \
    "artifacts/autoresearch/eval_fullbody.py" \
    "artifacts/autoresearch/export_shoulder_metrics.py"

  python3 - "$LABEL" "$DISPLAY_NAME" "$TASK" "$RUN_DIR" "$CKPT" "$OUT_DIR" "$METRICS_JSON" "$WEB_TAR" "$EVAL_LOG" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

keys = ["label", "display_name", "task", "run_dir", "checkpoint", "eval_dir", "metrics_json", "web_inputs", "eval_log"]
payload = dict(zip(keys, sys.argv[1:]))
payload["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
Path(f"artifacts/autoresearch/{payload['label']}_full_auto_done.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

  ls -lh "${WEB_TAR}" "${DONE_JSON}"
  echo "DONE manual eval label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${EVAL_LOG}"
