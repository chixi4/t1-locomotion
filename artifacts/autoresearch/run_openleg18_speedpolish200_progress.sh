#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH=/opt/conda/bin:/usr/lib/wsl/lib:${PATH}
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

TASK="T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedPolish_train200"
LABEL="upper9_camera_stable_openleg18_legresidual_speedpolish200"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="artifacts/autoresearch/${TASK}_${STAMP}_progress.log"
DONE_JSON="artifacts/autoresearch/${LABEL}_full_auto_done.json"
FAILED_JSON="artifacts/autoresearch/${LABEL}_full_auto_failed.json"

mkdir -p artifacts/autoresearch
rm -f "${DONE_JSON}" "${FAILED_JSON}"

{
  echo "==== START residual PPO ${STAMP} label=${LABEL} ===="
  echo "Task: ${TASK}"
  python3 -u train_fullbody_residual.py --task "${TASK}"
  status=$?
  if [[ ${status} -eq 0 ]]; then
    latest_run="$(python3 - <<'PY'
from pathlib import Path
import yaml
task = "T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedPolish_train200"
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
    checkpoint="${latest_run}/nn/model_700.pth"
    printf '{"label":"%s","task":"%s","run_dir":"%s","checkpoint":"%s","completed_at":"%s"}\n' \
      "${LABEL}" "${TASK}" "${latest_run}" "${checkpoint}" "$(date '+%Y-%m-%d %H:%M:%S')" > "${DONE_JSON}"
  else
    printf '{"label":"%s","task":"%s","status":%s,"failed_at":"%s"}\n' \
      "${LABEL}" "${TASK}" "${status}" "$(date '+%Y-%m-%d %H:%M:%S')" > "${FAILED_JSON}"
  fi
  echo "==== DONE status=${status} $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ===="
  exit ${status}
} 2>&1 | tee "${LOG}"
