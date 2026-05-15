#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:/opt/conda/lib/python3.8/site-packages/ninja/data/bin:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/conda/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export WANDB_MODE=disabled
mkdir -p artifacts/autoresearch
run_id="2026-05-02-12-15-00"
log="artifacts/autoresearch/official_highthroughput_${run_id}.log"
manifest="artifacts/autoresearch/official_highthroughput_${run_id}.json"
gpu_log="artifacts/autoresearch/official_highthroughput_${run_id}_gpu.csv"
exec > "$log" 2>&1
start_iso=$(date -Is)
commit=$(git rev-parse --short HEAD 2>/dev/null || true)
echo "START ${start_iso}"
echo "PWD=$PWD"
echo "OFFICIAL_COMMIT=${commit}"
echo "WANDB_MODE=$WANDB_MODE"
echo "COMMAND=/opt/conda/bin/python -u train.py --task T1 --headless true --num_envs 28672 --max_iterations 500"
(while true; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> "$gpu_log" 2>/dev/null || true; sleep 5; done) &
mon=$!
set +e
/opt/conda/bin/python -u train.py --task T1 --headless true --num_envs 28672 --max_iterations 500
rc=$?
set -e
kill "$mon" 2>/dev/null || true
wait "$mon" 2>/dev/null || true
finish_iso=$(date -Is)
latest_run=$(find logs -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk 'NR==1{print $2}')
latest_ckpt=""
if [ -n "${latest_run:-}" ]; then
  latest_ckpt=$(find "$latest_run/nn" -maxdepth 1 -name 'model_*.pth' 2>/dev/null | sort -V | tail -1 || true)
fi
START_ISO="$start_iso" FINISH_ISO="$finish_iso" RC="$rc" COMMIT="$commit" LATEST_RUN="$latest_run" LATEST_CKPT="$latest_ckpt" LOG_PATH="$log" GPU_LOG="$gpu_log" MANIFEST="$manifest" /opt/conda/bin/python - <<'PY'
import json, os
manifest = {
  "kind": "official_highthroughput_baseline",
  "started_at": os.environ["START_ISO"],
  "finished_at": os.environ["FINISH_ISO"],
  "returncode": int(os.environ["RC"]),
  "cwd": "/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official",
  "official_commit": os.environ.get("COMMIT", ""),
  "command": ["/opt/conda/bin/python", "-u", "train.py", "--task", "T1", "--headless", "true", "--num_envs", "28672", "--max_iterations", "500"],
  "num_envs": 28672,
  "max_iterations": 500,
  "wandb_mode": "disabled",
  "selection_reason": "best measured short-run throughput below PhysX 64K material limit",
  "latest_run": os.environ.get("LATEST_RUN", ""),
  "latest_checkpoint": os.environ.get("LATEST_CKPT", ""),
  "log": os.environ.get("LOG_PATH", ""),
  "gpu_log": os.environ.get("GPU_LOG", ""),
}
print(json.dumps(manifest, indent=2))
open(os.environ["MANIFEST"], "w", encoding="utf-8").write(json.dumps(manifest, indent=2) + "\n")
PY
echo "DONE ${finish_iso} rc=${rc} latest_run=${latest_run} latest_ckpt=${latest_ckpt}"
exit "$rc"
