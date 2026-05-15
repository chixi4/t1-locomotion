#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH=/opt/conda/bin:/usr/lib/wsl/lib:${PATH}
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

TASK="T1Upper9CameraStableOfficialOpenLeg18_fromBC_train500"
LABEL="upper9_camera_stable_openleg18_dagger500"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="artifacts/autoresearch/${TASK}_${STAMP}_progress.log"
BC_OUT="logs/warmstarts/upper9_camera_stable_openleg18_bc_dagger_model.pth"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"
FAILED_JSON="artifacts/autoresearch/${LABEL}_full_auto_failed.json"

mkdir -p artifacts/autoresearch logs/warmstarts
rm -f "${DONE_JSON}" "${FAILED_JSON}"

{
  echo "==== START fullbody warmstart ${STAMP} label=${LABEL} ===="
  echo "Task: ${TASK}"
  echo "BC:   ${BC_OUT}"
  if [[ ! -f "${BC_OUT}" ]]; then
    python3 -u artifacts/autoresearch/bootstrap_fullbody_bc.py \
      --task "${TASK}" \
      --leg-checkpoint logs/2026-05-05-11-09-07/nn/model_4000.pth \
      --upper-checkpoint logs/2026-05-14-08-59-29/nn/model_500.pth \
      --out "${BC_OUT}" \
      --num-envs 8192 \
      --steps 2400 \
      --lr 2.0e-4 \
      --dagger-start-frac 0.35 \
      --dagger-final-blend 0.75
  else
    echo "BC warmstart already exists; reusing ${BC_OUT}"
  fi
  PROBE_JSON="artifacts/autoresearch/${LABEL}_bc_probe.json"
  echo "==== START BC probe $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ===="
  python3 -u artifacts/autoresearch/probe_fullbody_checkpoint.py \
    --task "${TASK}" \
    --checkpoint "${BC_OUT}" \
    --num-envs 192 \
    --seconds 4 \
    --warmup-s 1 \
    --out "${PROBE_JSON}"
  python3 - "${PROBE_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
bad = []
for item in data["results"]:
    if item["reset_events_per_env"] > 0.05:
        bad.append(f"{item['case']} reset={item['reset_events_per_env']:.3f}")
if bad:
    raise SystemExit("BC probe failed: " + ", ".join(bad))
print("BC probe accepted")
PY
  echo "==== START PPO $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ===="
  python3 -u train.py --task "${TASK}"
  status=$?
  if [[ ${status} -eq 0 ]]; then
    latest_run="$(python3 - <<'PY'
from pathlib import Path
import yaml
task = "T1Upper9CameraStableOfficialOpenLeg18_fromBC_train500"
candidates = []
for cfg_path in Path("logs").glob("*/config.yaml"):
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if cfg.get("basic", {}).get("task") == task:
        candidates.append(cfg_path.parent)
if candidates:
    print(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
PY
)"
    checkpoint="${latest_run}/nn/model_500.pth"
    printf '{"label":"%s","task":"%s","run_dir":"%s","checkpoint":"%s","completed_at":"%s"}\n' \
      "${LABEL}" "${TASK}" "${latest_run}" "${checkpoint}" "$(date '+%Y-%m-%d %H:%M:%S')" > "${DONE_JSON}"
  else
    printf '{"label":"%s","task":"%s","status":%s,"failed_at":"%s"}\n' \
      "${LABEL}" "${TASK}" "${status}" "$(date '+%Y-%m-%d %H:%M:%S')" > "${FAILED_JSON}"
  fi
  echo "==== DONE status=${status} $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ===="
  exit ${status}
} 2>&1 | tee "${LOG}"
