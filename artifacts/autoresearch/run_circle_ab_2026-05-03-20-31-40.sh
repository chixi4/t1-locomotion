#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:/opt/conda/lib/python3.8/site-packages/ninja/data/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/conda/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export WANDB_MODE=disabled
run_id="2026-05-03-20-31-40"
run_root="/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official/artifacts/autoresearch/circle_ab_2026-05-03-20-31-40"
mkdir -p "$run_root"
gpu_log="$run_root/gpu.csv"
summary_tsv="artifacts/autoresearch/results.tsv"
echo "CIRCLE_AB_START $(date -Is) run_id=${run_id} cwd=$PWD"
echo "RUN_ROOT=$run_root"
echo "GPU_LOG=$gpu_log"
(while true; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> "$gpu_log" 2>/dev/null || true; sleep 5; done) &
mon=$!
cleanup() { kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true; }
trap cleanup EXIT
run_one() {
  local task="$1"
  local label="$2"
  local num_envs=28672
  local max_iterations=1000
  local start_iso finish_iso commit latest_run latest_ckpt rc log manifest
  log="$run_root/${label}.log"
  manifest="$run_root/${label}.json"
  start_iso=$(date -Is)
  commit=$(git rev-parse --short HEAD 2>/dev/null || true)
  echo "START $label $start_iso task=$task num_envs=$num_envs max_iterations=$max_iterations commit=$commit"
  echo "COMMAND=/opt/conda/bin/python -u train.py --task $task --headless true --num_envs $num_envs --max_iterations $max_iterations"
  set +e
  /opt/conda/bin/python -u train.py --task "$task" --headless true --num_envs "$num_envs" --max_iterations "$max_iterations" > "$log" 2>&1
  rc=$?
  set -e
  finish_iso=$(date -Is)
  latest_run=$(find logs -maxdepth 1 -type d -printf '%T@ %p
' | sort -nr | awk 'NR==1{print $2}')
  latest_ckpt=""
  if [ -n "${latest_run:-}" ]; then
    latest_ckpt=$(find "$latest_run/nn" -maxdepth 1 -name 'model_*.pth' 2>/dev/null | sort -V | tail -1 || true)
  fi
  START_ISO="$start_iso" FINISH_ISO="$finish_iso" RC="$rc" COMMIT="$commit" TASK="$task" LABEL="$label" NUM_ENVS="$num_envs" MAX_ITERATIONS="$max_iterations" LATEST_RUN="$latest_run" LATEST_CKPT="$latest_ckpt" LOG_PATH="$log" MANIFEST="$manifest" RUN_ROOT="$run_root" /opt/conda/bin/python - <<'PY'
import json, os
manifest = {
    "kind": "circle_command_ab_from_scratch",
    "started_at": os.environ["START_ISO"],
    "finished_at": os.environ["FINISH_ISO"],
    "returncode": int(os.environ["RC"]),
    "cwd": "/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official",
    "commit": os.environ.get("COMMIT", ""),
    "task": os.environ["TASK"],
    "label": os.environ["LABEL"],
    "command": ["/opt/conda/bin/python", "-u", "train.py", "--task", os.environ["TASK"], "--headless", "true", "--num_envs", os.environ["NUM_ENVS"], "--max_iterations", os.environ["MAX_ITERATIONS"]],
    "num_envs": int(os.environ["NUM_ENVS"]),
    "max_iterations": int(os.environ["MAX_ITERATIONS"]),
    "checkpoint": None,
    "from_scratch": True,
    "sampler": "uniform_area_circle_vmax_1",
    "duration_s": [8.0, 12.0],
    "wandb_mode": "disabled",
    "latest_run": os.environ.get("LATEST_RUN", ""),
    "latest_checkpoint": os.environ.get("LATEST_CKPT", ""),
    "log": os.environ.get("LOG_PATH", ""),
    "run_root": os.environ.get("RUN_ROOT", ""),
}
print(json.dumps(manifest, indent=2))
Path = __import__('pathlib').Path
Path(os.environ["MANIFEST"]).write_text(json.dumps(manifest, indent=2) + "
", encoding="utf-8")
PY
  printf '%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s	%s
'     "$finish_iso" "$label" "$latest_run" "$max_iterations" "$latest_ckpt" "launch_complete" "rc=$rc" "" "circle_sampler_vmax1_curriculum=${task#T1Circle}" "$commit" "" "" "log=$log" >> "$summary_tsv"
  echo "DONE $label $finish_iso rc=$rc latest_run=$latest_run latest_ckpt=$latest_ckpt"
  if [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
}
run_one T1Circle circle_full
run_one T1CircleCurriculum circle_curriculum
echo "CIRCLE_AB_DONE $(date -Is) run_id=${run_id}"
