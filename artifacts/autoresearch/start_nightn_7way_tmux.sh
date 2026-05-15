#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

SESSION="${SESSION:-nightn_7way}"
LOG="${LOG:-artifacts/autoresearch/nightn_7way_tmux.out}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "SESSION_EXISTS ${SESSION}"
  tmux list-sessions | grep "${SESSION}" || true
  exit 0
fi

python3 - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("artifacts/autoresearch/nightn_runs.json").read_text(encoding="utf-8"))
for item in meta:
    label = item["label"]
    for suffix in ("_full_auto_done.json", "_full_auto_failed.json", "_web_inputs.tgz"):
        p = Path("artifacts/autoresearch") / f"{label}{suffix}"
        if p.exists():
            p.unlink()
PY

TMUX_CMD="cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official &&"
for var in \
  PYTHON_BIN \
  LEG_CKPT \
  FINAL_ITER \
  TRAIN_MAX_ITER \
  TRAIN_NUM_ENVS \
  EVAL_NUM_ENVS \
  EVAL_FIXED_DURATION_S \
  EVAL_WARMUP_S \
  EVAL_RANDOM_SECONDS \
  EVAL_FPS \
  META_JSON \
  MASTER_LOG \
  RUN_CODES_OVERRIDE \
  FAST_REPLAY_ONLY
do
  if [[ "${!var+x}" == "x" ]]; then
    printf -v quoted_value '%q' "${!var}"
    TMUX_CMD+=" ${var}=${quoted_value}"
  fi
done
printf -v quoted_log '%q' "${LOG}"
TMUX_CMD+=" bash artifacts/autoresearch/run_nightn_7way_full_auto.sh > ${quoted_log} 2>&1"

tmux new-session -d -s "${SESSION}" "${TMUX_CMD}"

echo "STARTED ${SESSION}"
tmux list-sessions | grep "${SESSION}" || true
