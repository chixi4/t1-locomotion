#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH=/opt/conda/bin:/usr/lib/wsl/lib:${PATH}
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

TASK="T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedUnlock_train300"
LABEL="upper9_camera_stable_openleg18_legresidual_speedunlock300"
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
    python3 - <<'PY' "${TASK}" "${LABEL}" "${DONE_JSON}"
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

task, label, done_json = sys.argv[1:4]
candidates = []
for cfg_path in Path("logs").glob("*/config.yaml"):
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if cfg.get("basic", {}).get("task") == task:
        candidates.append(cfg_path.parent)
if not candidates:
    raise SystemExit(f"no run dir found for {task}")
run_dir = sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]
checkpoints = sorted(
    run_dir.joinpath("nn").glob("model_*.pth"),
    key=lambda path: int(re.search(r"model_(\d+)\.pth$", path.name).group(1)),
)
if not checkpoints:
    raise SystemExit(f"no checkpoints found under {run_dir / 'nn'}")
payload = {
    "label": label,
    "task": task,
    "run_dir": str(run_dir),
    "checkpoint": str(checkpoints[-1]),
    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
Path(done_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  else
    printf '{"label":"%s","task":"%s","status":%s,"failed_at":"%s"}\n' \
      "${LABEL}" "${TASK}" "${status}" "$(date '+%Y-%m-%d %H:%M:%S')" > "${FAILED_JSON}"
  fi
  echo "==== DONE status=${status} $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} ===="
  exit ${status}
} 2>&1 | tee "${LOG}"


