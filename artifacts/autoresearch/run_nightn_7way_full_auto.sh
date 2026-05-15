#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

PYTHON_BIN="${PYTHON_BIN:-python3}"
LEG_CKPT="${LEG_CKPT:-logs/2026-05-05-11-09-07/nn/model_4000.pth}"
FINAL_ITER="${FINAL_ITER:-300}"
TRAIN_MAX_ITER="${TRAIN_MAX_ITER:-${FINAL_ITER}}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-28672}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-256}"
EVAL_FIXED_DURATION_S="${EVAL_FIXED_DURATION_S:-8}"
EVAL_WARMUP_S="${EVAL_WARMUP_S:-2}"
EVAL_RANDOM_SECONDS="${EVAL_RANDOM_SECONDS:-60}"
EVAL_FPS="${EVAL_FPS:-50}"
META_JSON="${META_JSON:-artifacts/autoresearch/nightn_runs.json}"
MASTER_LOG="${MASTER_LOG:-artifacts/autoresearch/nightn_7way_full_auto_$(date +%Y%m%d_%H%M%S).log}"

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/isaacgym/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/autoresearch
touch "${MASTER_LOG}"

preflight() {
  "${PYTHON_BIN}" - <<'PY'
import sys
import isaacgym  # noqa: F401
import tensorboard  # noqa: F401
import yaml  # noqa: F401
print(f"PRECHECK python={sys.executable}")
PY
}

if [[ -n "${RUN_CODES_OVERRIDE:-}" ]]; then
  read -r -a RUN_CODES <<< "${RUN_CODES_OVERRIDE//,/ }"
else
  mapfile -t RUN_CODES < <("${PYTHON_BIN}" - "${META_JSON}" <<'PY'
import json
import sys
from pathlib import Path
for item in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    print(item["code"])
PY
  )
fi

meta_field() {
  local code="$1"
  local field="$2"
  "${PYTHON_BIN}" - "$META_JSON" "$code" "$field" <<'PY'
import json
import sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in meta:
    if item["code"] == sys.argv[2]:
        print(item[sys.argv[3]])
        raise SystemExit(0)
raise SystemExit(f"missing run code={sys.argv[2]}")
PY
}

find_run_dir() {
  "${PYTHON_BIN}" - "$TASK" "$FINAL_ITER" <<'PY'
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
  "${PYTHON_BIN}" - "$TASK" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("envs", f"{sys.argv[1]}.yaml").read_text(encoding="utf-8"))
print(cfg["asset"]["file"])
PY
}

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

phase_fail() {
  local now
  now=$(date +%s)
  echo "==== FAIL ${PHASE_NAME} $(date '+%Y-%m-%d %H:%M:%S') elapsed=$((now - PHASE_T0))s label=${LABEL} ====" | tee -a "${TRAIN_LOG}"
}

write_status_json() {
  local target="$1"
  local status="$2"
  local failure_phase="${3:-}"
  local failure_reason="${4:-}"
  "${PYTHON_BIN}" - "$target" "$status" "$failure_phase" "$failure_reason" "$LABEL" "$DISPLAY_NAME" "$TASK" "${RUN_DIR:-}" "${ARM_CKPT:-}" "$LEG_CKPT" "$OUT_DIR" "$METRICS_JSON" "$WEB_TAR" "$TRAIN_LOG" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

keys = [
    "target",
    "status",
    "failure_phase",
    "failure_reason",
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
target = Path(payload.pop("target"))
payload["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

fail_run() {
  local failure_phase="$1"
  local failure_reason="$2"
  echo "RUN_FAILED phase=${failure_phase} reason=${failure_reason}" | tee -a "${TRAIN_LOG}"
  echo "LAST_LOG_LINES_BEGIN" | tee -a "${TRAIN_LOG}"
  tail -n 40 "${TRAIN_LOG}" || true
  echo "LAST_LOG_LINES_END" | tee -a "${TRAIN_LOG}"
  write_status_json "${FAIL_JSON}" failed "${failure_phase}" "${failure_reason}" | tee -a "${TRAIN_LOG}"
  return 1
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
  RUN_DIR=""
  ARM_CKPT=""
  URDF_FILE=""

  rm -rf "${OUT_DIR}"
  rm -f "${WEB_TAR}" "${DONE_JSON}" "${FAIL_JSON}" "${METRICS_JSON}"
  echo "START run code=${CODE} label=${LABEL} task=${TASK} $(date '+%Y-%m-%d %H:%M:%S')" | tee "${TRAIN_LOG}"
  echo "CONFIG train_num_envs=${TRAIN_NUM_ENVS} train_max_iterations=${TRAIN_MAX_ITER} final_iter=${FINAL_ITER} eval_num_envs=${EVAL_NUM_ENVS}" | tee -a "${TRAIN_LOG}"

  phase_start train
  if ! stdbuf -oL -eL "${PYTHON_BIN}" train_shoulder4_frozen.py \
    --task "${TASK}" \
    --num_envs "${TRAIN_NUM_ENVS}" \
    --max_iterations "${TRAIN_MAX_ITER}" >> "${TRAIN_LOG}" 2>&1; then
    phase_fail
    fail_run train "train_shoulder4_frozen.py failed"
  fi
  phase_done

  if ! RUN_DIR="$(find_run_dir)"; then
    fail_run resolve_run_dir "no completed run dir found"
  fi
  ARM_CKPT="${RUN_DIR}/nn/model_${FINAL_ITER}.pth"
  URDF_FILE="$(asset_file)"
  echo "RUN_DIR=${RUN_DIR}" | tee -a "${TRAIN_LOG}"
  echo "ARM_CKPT=${ARM_CKPT}" | tee -a "${TRAIN_LOG}"
  echo "URDF_FILE=${URDF_FILE}" | tee -a "${TRAIN_LOG}"

  phase_start eval
  if ! bash artifacts/autoresearch/run_shoulder4_eval_pair.sh \
    --task "${TASK}" \
    --label "${LABEL}" \
    --arm_checkpoint "${ARM_CKPT}" \
    --leg_checkpoint "${LEG_CKPT}" \
    --out_dir "${OUT_DIR}" \
    --num_envs "${EVAL_NUM_ENVS}" \
    --fixed_duration_s "${EVAL_FIXED_DURATION_S}" \
    --warmup_s "${EVAL_WARMUP_S}" \
    --random_seconds "${EVAL_RANDOM_SECONDS}" \
    --fps "${EVAL_FPS}" \
    --clean >> "${TRAIN_LOG}" 2>&1; then
    phase_fail
    fail_run eval "run_shoulder4_eval_pair.sh failed"
  fi
  phase_done

  phase_start metrics
  if ! "${PYTHON_BIN}" artifacts/autoresearch/export_shoulder_metrics.py \
    --run-dir "${RUN_DIR}" \
    --out "${METRICS_JSON}" \
    --run-name "${DISPLAY_NAME}" \
    --checkpoint "${ARM_CKPT}" \
    --note "${NOTE} Command: ${COMMAND_TEXT}" >> "${TRAIN_LOG}" 2>&1; then
    phase_fail
    fail_run metrics "export_shoulder_metrics.py failed"
  fi
  phase_done

  phase_start pack
  if command -v pigz >/dev/null 2>&1; then
    TAR_COMPRESS=(--use-compress-program "pigz -1")
  else
    TAR_COMPRESS=(-I "gzip -1")
  fi
  if ! tar "${TAR_COMPRESS[@]}" -cf "${WEB_TAR}" \
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
    "artifacts/autoresearch/run_nightn_7way_full_auto.sh" \
    "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
    "artifacts/autoresearch/eval_shoulder4_frozen.py" \
    "artifacts/autoresearch/export_shoulder_metrics.py"; then
    phase_fail
    fail_run pack "tar packaging failed"
  fi
  phase_done

  write_status_json "${DONE_JSON}" done "" "" | tee -a "${TRAIN_LOG}"

  ls -lh "${WEB_TAR}" "${DONE_JSON}" | tee -a "${TRAIN_LOG}"
  echo "DONE run label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${TRAIN_LOG}"
)

preflight | tee -a "${MASTER_LOG}"
echo "==== NIGHTN 7WAY START $(date '+%Y-%m-%d %H:%M:%S') order=${RUN_CODES[*]} ====" | tee -a "${MASTER_LOG}"
for idx in "${!RUN_CODES[@]}"; do
  CODE="${RUN_CODES[$idx]}"
  LABEL="$(meta_field "$CODE" label)"
  echo "==== RUN $((idx + 1))/${#RUN_CODES[@]} code=${CODE} label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
  if run_one "$CODE" 2>&1 | tee -a "${MASTER_LOG}"; then
    echo "==== RUN OK label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
  else
    status=${PIPESTATUS[0]}
    echo "==== RUN FAILED label=${LABEL} status=${status} $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
  fi
done
echo "==== NIGHTN 7WAY DONE $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "${MASTER_LOG}"
