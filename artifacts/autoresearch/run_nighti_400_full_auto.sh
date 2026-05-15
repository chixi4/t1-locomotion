#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"
FINAL_ITER="400"
META_JSON="artifacts/autoresearch/nighti_runs.json"
MASTER_LOG="artifacts/autoresearch/nighti_400_full_auto_$(date +%Y%m%d_%H%M%S).log"
RUN_CODES=(i)

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/autoresearch
touch "${MASTER_LOG}"

start_gpu_audit() {
  if [[ -x artifacts/autoresearch/nighti_gpu_audit_wsl.sh ]]; then
    mkdir -p artifacts/autoresearch/gpu_train_audit
    if ! pgrep -f "nighti_gpu_audit_wsl.sh" >/dev/null 2>&1; then
      nohup bash artifacts/autoresearch/nighti_gpu_audit_wsl.sh \
        > artifacts/autoresearch/gpu_train_audit/nighti_wsl_logger.out 2>&1 &
      echo "$!" > artifacts/autoresearch/gpu_train_audit/nighti_wsl_logger.pid
      echo "==== GPU audit started pid=$(cat artifacts/autoresearch/gpu_train_audit/nighti_wsl_logger.pid) ====" | tee -a "${MASTER_LOG}"
    else
      echo "==== GPU audit already running ====" | tee -a "${MASTER_LOG}"
    fi
  fi
}

meta_field() {
  local code="$1"
  local field="$2"
  python3 - "$META_JSON" "$code" "$field" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
code, field = sys.argv[2], sys.argv[3]
for item in meta:
    if item["code"] == code:
        print(item[field])
        raise SystemExit(0)
raise SystemExit(f"missing run code={code}")
PY
}

phase_start() {
  PHASE_NAME="$1"
  PHASE_T0=$(date +%s)
  echo "==== START ${PHASE_NAME} $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ====" | tee -a "${TRAIN_LOG:-/dev/null}"
}

phase_done() {
  local now
  now=$(date +%s)
  echo "==== DONE ${PHASE_NAME} $(date '+%Y-%m-%d %H:%M:%S') elapsed=$((now - PHASE_T0))s label=${LABEL} ====" | tee -a "${TRAIN_LOG:-/dev/null}"
}

find_run_dir() {
  python3 - "$TASK" "$FINAL_ITER" <<'PY'
from pathlib import Path
import sys
import yaml

task = sys.argv[1]
final_iter = sys.argv[2]
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

write_failed_json() {
  local status="$1"
  python3 - "$LABEL" "$DISPLAY_NAME" "$TASK" "$TRAIN_LOG" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

label, display_name, task, train_log, status = sys.argv[1:6]
payload = {
    "label": label,
    "display_name": display_name,
    "task": task,
    "train_log": train_log,
    "status": int(status),
    "failed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
Path(f"artifacts/autoresearch/{label}_full_auto_failed.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

run_one() (
  set -euo pipefail
  CODE="$1"
  TASK="$(meta_field "$CODE" task)"
  LABEL="$(meta_field "$CODE" label)"
  DISPLAY_NAME="$(meta_field "$CODE" display)"
  NOTE="$(meta_field "$CODE" note)"
  COMMAND_TEXT="$(meta_field "$CODE" command)"
  RUN_ID="${TASK}_$(date +%Y%m%d_%H%M%S)_full_auto"
  TRAIN_LOG="artifacts/autoresearch/${RUN_ID}.log"
  OUT_DIR="artifacts/autoresearch/${LABEL}_eval"
  METRICS_JSON="artifacts/autoresearch/${LABEL}_metrics.json"
  WEB_TAR="artifacts/autoresearch/${LABEL}_web_inputs.tgz"
  DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"
  FAIL_JSON="artifacts/autoresearch/${LABEL}_full_auto_failed.json"

  rm -f "${WEB_TAR}" "${DONE_JSON}" "${FAIL_JSON}"
  echo "START run code=${CODE} label=${LABEL} task=${TASK} $(date '+%Y-%m-%d %H:%M:%S')" | tee "${TRAIN_LOG}"

  phase_start train
  stdbuf -oL -eL python3 train_shoulder4_frozen.py --task "${TASK}" 2>&1 | tee -a "${TRAIN_LOG}"
  phase_done

  RUN_DIR="$(find_run_dir)"
  ARM_CKPT="${RUN_DIR}/nn/model_${FINAL_ITER}.pth"
  URDF_FILE="$(asset_file)"
  echo "RUN_DIR=${RUN_DIR}" | tee -a "${TRAIN_LOG}"
  echo "ARM_CKPT=${ARM_CKPT}" | tee -a "${TRAIN_LOG}"
  echo "URDF_FILE=${URDF_FILE}" | tee -a "${TRAIN_LOG}"

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
    --note "${NOTE} Command: ${COMMAND_TEXT}" 2>&1 | tee -a "${TRAIN_LOG}"
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
    "${URDF_FILE}" \
    "${META_JSON}" \
    "artifacts/autoresearch/run_nighti_400_full_auto.sh" \
    "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
    "artifacts/autoresearch/eval_shoulder4_frozen.py" \
    "artifacts/autoresearch/export_shoulder_metrics.py"
  phase_done

  python3 - "$LABEL" "$DISPLAY_NAME" "$TASK" "$RUN_DIR" "$ARM_CKPT" "$LEG_CKPT" "$OUT_DIR" "$METRICS_JSON" "$WEB_TAR" "$TRAIN_LOG" <<'PY'
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
    "train_log",
]
payload = dict(zip(keys, sys.argv[1:]))
payload["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
Path(f"artifacts/autoresearch/{payload['label']}_full_auto_done.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

  ls -lh "${WEB_TAR}" "${DONE_JSON}" | tee -a "${TRAIN_LOG}"
  echo "DONE run label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${TRAIN_LOG}"
)

echo "==== NIGHTI START $(date '+%Y-%m-%d %H:%M:%S') order=${RUN_CODES[*]} ====" | tee -a "${MASTER_LOG}"
start_gpu_audit
for idx in "${!RUN_CODES[@]}"; do
  CODE="${RUN_CODES[$idx]}"
  LABEL="$(meta_field "$CODE" label)"
  TASK="$(meta_field "$CODE" task)"
  DISPLAY_NAME="$(meta_field "$CODE" display)"
  TRAIN_LOG="artifacts/autoresearch/${TASK}_pending.log"
  echo "==== RUN $((idx + 1))/${#RUN_CODES[@]} code=${CODE} label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
  if run_one "$CODE" 2>&1 | tee -a "${MASTER_LOG}"; then
    echo "==== RUN OK label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
  else
    status=${PIPESTATUS[0]}
    echo "==== RUN FAILED label=${LABEL} status=${status} $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
    write_failed_json "$status" 2>&1 | tee -a "${MASTER_LOG}" || true
  fi
done
echo "==== NIGHTI DONE $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
