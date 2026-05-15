#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/isaacgym/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3}"
TASK="T1Shoulder4OfficialZeroSwingVelOpp_from7000LegFrozen_train400"
LABEL="shoulder4_official_zero_swing_velopp400"
DISPLAY_NAME="T1Shoulder4 Official ZeroSwing VelOpp model_400"
NOTE="Official-leg-like shoulder training from scratch: frozen 7000 leg model, official terrain/randomization/commands/body rewards, zero standing arm target, normal contralateral pitch swing, and commanded outward foot velocity side-lift for the opposite arm."
COMMAND_TEXT="OfficialZeroSwingVelOpp: checkpoint=null; official leg rewards/settings restored; shoulder_static_posture targets exact zero at stand; pitch target from abs(vx_cmd)*left-right foot x distance; side roll uses commanded outward foot velocity so right foot moving outward on right sidestep raises left arm and left foot moving outward on left sidestep raises right arm."
LEG_CKPT="${LEG_CKPT:-logs/2026-05-05-11-09-07/nn/model_4000.pth}"
FINAL_ITER="${FINAL_ITER:-400}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-24576}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-256}"
EVAL_FIXED_DURATION_S="${EVAL_FIXED_DURATION_S:-8}"
EVAL_WARMUP_S="${EVAL_WARMUP_S:-2}"
EVAL_RANDOM_SECONDS="${EVAL_RANDOM_SECONDS:-60}"
EVAL_FPS="${EVAL_FPS:-50}"
META_JSON="artifacts/autoresearch/official_zero_swing_velopp_runs.json"
RUN_ID="${TASK}_$(date +%Y%m%d_%H%M%S)_progress"
TRAIN_LOG="artifacts/autoresearch/${RUN_ID}.log"
LATEST_LOG_LINK="artifacts/autoresearch/${LABEL}_latest.log"
OUT_DIR="artifacts/autoresearch/${LABEL}_eval"
METRICS_JSON="artifacts/autoresearch/${LABEL}_metrics.json"
WEB_TAR="artifacts/autoresearch/${LABEL}_web_inputs.tgz"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"
FAIL_JSON="artifacts/autoresearch/${LABEL}_full_auto_failed.json"
mkdir -p artifacts/autoresearch
rm -rf "${OUT_DIR}"
rm -f "${WEB_TAR}" "${DONE_JSON}" "${FAIL_JSON}" "${METRICS_JSON}" "${LATEST_LOG_LINK}"
ln -s "$(basename "${TRAIN_LOG}")" "${LATEST_LOG_LINK}"
phase_start(){ PHASE_NAME="$1"; PHASE_T0=$(date +%s); echo "==== START ${PHASE_NAME} $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ====" | tee -a "${TRAIN_LOG}"; }
phase_done(){ local now; now=$(date +%s); echo "==== DONE ${PHASE_NAME} $(date '+%Y-%m-%d %H:%M:%S') elapsed=$((now-PHASE_T0))s label=${LABEL} ====" | tee -a "${TRAIN_LOG}"; }
write_status_json(){
  local target="$1" status="$2" failure_phase="${3:-}" failure_reason="${4:-}"
  "${PYTHON_BIN}" - "$target" "$status" "$failure_phase" "$failure_reason" "$LABEL" "$DISPLAY_NAME" "$TASK" "${RUN_DIR:-}" "${ARM_CKPT:-}" "$LEG_CKPT" "$OUT_DIR" "$METRICS_JSON" "$WEB_TAR" "$TRAIN_LOG" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
keys=["target","status","failure_phase","failure_reason","label","display_name","task","run_dir","arm_checkpoint","leg_checkpoint","eval_dir","metrics_json","web_inputs","train_log"]
payload=dict(zip(keys, sys.argv[1:])); target=Path(payload.pop('target'))
payload['created_at']=datetime.now(timezone.utc).replace(microsecond=0).isoformat()
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}
find_run_dir(){
  "${PYTHON_BIN}" - "$TASK" "$FINAL_ITER" <<'PY'
from pathlib import Path
import sys, yaml
task=sys.argv[1]; final_iter=sys.argv[2]; c=[]
for cfg_path in Path('logs').glob('*/config.yaml'):
    try: cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    except Exception: continue
    ckpt=cfg_path.parent/'nn'/f'model_{final_iter}.pth'
    if cfg.get('basic',{}).get('task')==task and ckpt.exists(): c.append(cfg_path.parent)
if not c: raise SystemExit(f'no completed run dir found for {task}')
print(sorted(c,key=lambda p:p.stat().st_mtime)[-1])
PY
}
asset_file(){ "${PYTHON_BIN}" - "$TASK" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path('envs',f'{sys.argv[1]}.yaml').read_text(encoding='utf-8'))
print(cfg['asset']['file'])
PY
}
fail_run(){ local failure_phase="$1" failure_reason="$2"; echo "RUN_FAILED phase=${failure_phase} reason=${failure_reason}" | tee -a "${TRAIN_LOG}"; write_status_json "${FAIL_JSON}" failed "${failure_phase}" "${failure_reason}" | tee -a "${TRAIN_LOG}" || true; exit 1; }
echo "START run label=${LABEL} task=${TASK} envs=${TRAIN_NUM_ENVS} final_iter=${FINAL_ITER} $(date '+%Y-%m-%d %H:%M:%S')" | tee "${TRAIN_LOG}"
echo "CONFIG scratch arm checkpoint=null; leg=${LEG_CKPT}; official leg rewards/settings restored" | tee -a "${TRAIN_LOG}"
echo "CONFIG reward: zero stand target; pitch abs(cmd_x)*foot_sag; commanded outward foot velocity side roll right-foot->left-arm left-foot->right-arm" | tee -a "${TRAIN_LOG}"
phase_start train
if ! stdbuf -oL -eL "${PYTHON_BIN}" train_shoulder4_frozen.py --task "${TASK}" --leg_checkpoint "${LEG_CKPT}" --num_envs "${TRAIN_NUM_ENVS}" --max_iterations "${FINAL_ITER}" 2>&1 | tee -a "${TRAIN_LOG}"; then fail_run train "train_shoulder4_frozen.py failed"; fi
phase_done train
if ! RUN_DIR="$(find_run_dir)"; then fail_run resolve_run_dir "no completed run dir found"; fi
ARM_CKPT="${RUN_DIR}/nn/model_${FINAL_ITER}.pth"; URDF_FILE="$(asset_file)"
echo "RUN_DIR=${RUN_DIR}" | tee -a "${TRAIN_LOG}"; echo "ARM_CKPT=${ARM_CKPT}" | tee -a "${TRAIN_LOG}"; echo "URDF_FILE=${URDF_FILE}" | tee -a "${TRAIN_LOG}"
phase_start eval
if ! bash artifacts/autoresearch/run_shoulder4_eval_pair.sh --task "${TASK}" --label "${LABEL}" --arm_checkpoint "${ARM_CKPT}" --leg_checkpoint "${LEG_CKPT}" --out_dir "${OUT_DIR}" --num_envs "${EVAL_NUM_ENVS}" --fixed_duration_s "${EVAL_FIXED_DURATION_S}" --warmup_s "${EVAL_WARMUP_S}" --random_seconds "${EVAL_RANDOM_SECONDS}" --fps "${EVAL_FPS}" --clean 2>&1 | tee -a "${TRAIN_LOG}"; then fail_run eval "run_shoulder4_eval_pair.sh failed"; fi
phase_done eval
phase_start metrics
if ! "${PYTHON_BIN}" artifacts/autoresearch/export_shoulder_metrics.py --run-dir "${RUN_DIR}" --out "${METRICS_JSON}" --run-name "${DISPLAY_NAME}" --checkpoint "${ARM_CKPT}" --note "${NOTE} Command: ${COMMAND_TEXT}" 2>&1 | tee -a "${TRAIN_LOG}"; then fail_run metrics "export_shoulder_metrics.py failed"; fi
phase_done metrics
phase_start pack
printf '{"label":"%s","display_name":"%s","task":"%s","note":"%s"}\n' "${LABEL}" "${DISPLAY_NAME}" "${TASK}" "${NOTE}" > "${META_JSON}"
if command -v pigz >/dev/null 2>&1; then TAR_COMPRESS=(--use-compress-program "pigz -1"); else TAR_COMPRESS=(-I "gzip -1"); fi
if ! tar "${TAR_COMPRESS[@]}" -cf "${WEB_TAR}" "${OUT_DIR}" "${METRICS_JSON}" "${TRAIN_LOG}" "envs/${TASK}.yaml" "envs/t1.py" "envs/__init__.py" "utils/shoulder_runner.py" "utils/model.py" "${URDF_FILE}" "${META_JSON}" "artifacts/autoresearch/run_official_zero_swing_velopp400_progress.sh" "artifacts/autoresearch/run_shoulder4_eval_pair.sh" "artifacts/autoresearch/eval_shoulder4_frozen.py" "artifacts/autoresearch/export_shoulder_metrics.py"; then fail_run pack "tar packaging failed"; fi
phase_done pack
write_status_json "${DONE_JSON}" done "" "" | tee -a "${TRAIN_LOG}"
ls -lh "${WEB_TAR}" "${DONE_JSON}" | tee -a "${TRAIN_LOG}"
echo "DONE run label=${LABEL} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${TRAIN_LOG}"
