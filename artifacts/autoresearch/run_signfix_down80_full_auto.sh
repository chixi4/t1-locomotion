#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK="T1Shoulder4GaitPhaseSignFixDown_from7000LegFrozen_train80"
LABEL="shoulder4_gaitphase_signfix_down80"
DISPLAY_NAME="T1Shoulder4 SignFix Down model_80"
LEG_CKPT="logs/2026-05-05-11-09-07/nn/model_4000.pth"
FINAL_ITER="80"
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
rm -f "$WEB_TAR" "$DONE_JSON"

echo "START train $(date '+%Y-%m-%d %H:%M:%S') task=${TASK}" | tee "$TRAIN_LOG"
stdbuf -oL -eL python3 train_shoulder4_frozen.py --task "$TASK" 2>&1 | tee -a "$TRAIN_LOG"

RUN_DIR="$(python3 - <<'PY'
from pathlib import Path
import yaml

task = "T1Shoulder4GaitPhaseSignFixDown_from7000LegFrozen_train80"
final_iter = "80"
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
echo "RUN_DIR=${RUN_DIR}" | tee -a "$TRAIN_LOG"
echo "ARM_CKPT=${ARM_CKPT}" | tee -a "$TRAIN_LOG"

echo "START eval $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL}" | tee -a "$TRAIN_LOG"
bash artifacts/autoresearch/run_shoulder4_eval_pair.sh \
  --task "$TASK" \
  --label "$LABEL" \
  --arm_checkpoint "$ARM_CKPT" \
  --leg_checkpoint "$LEG_CKPT" \
  --out_dir "$OUT_DIR" \
  --num_envs 256 \
  --fixed_duration_s 8 \
  --warmup_s 2 \
  --random_seconds 60 \
  --fps 50 \
  --clean 2>&1 | tee -a "$TRAIN_LOG"

echo "START metrics $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL}" | tee -a "$TRAIN_LOG"
python3 artifacts/autoresearch/export_shoulder_metrics.py \
  --run-dir "$RUN_DIR" \
  --out "$METRICS_JSON" \
  --run-name "$DISPLAY_NAME" \
  --checkpoint "$ARM_CKPT" \
  --note "Foot-phase sign -1.0, foot-phase pitch weight -4.0, low-speed same-down constraint." 2>&1 | tee -a "$TRAIN_LOG"

echo "START pack $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL}" | tee -a "$TRAIN_LOG"
tar -czf "$WEB_TAR" \
  "$OUT_DIR" \
  "$METRICS_JSON" \
  "$TRAIN_LOG" \
  "envs/${TASK}.yaml" \
  "artifacts/autoresearch/run_signfix_down80_full_auto.sh" \
  "artifacts/autoresearch/run_shoulder4_eval_pair.sh" \
  "artifacts/autoresearch/eval_shoulder4_frozen.py" \
  "artifacts/autoresearch/export_shoulder_metrics.py"

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

ls -lh "$WEB_TAR" "$DONE_JSON" | tee -a "$TRAIN_LOG"
echo "DONE $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL}" | tee -a "$TRAIN_LOG"
