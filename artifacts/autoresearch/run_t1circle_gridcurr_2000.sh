#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PATH=/opt/conda/bin:/opt/conda/lib/python3.8/site-packages/ninja/data/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/conda/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}

run_id="grid3d_curriculum_$(date +%Y-%m-%d-%H-%M-%S)"
out_dir="artifacts/autoresearch/${run_id}"
mkdir -p "$out_dir"
ln -sfn "$run_id" artifacts/autoresearch/grid3d_curriculum_latest

{
  echo "run_id=${run_id}"
  echo "task=T1CircleGridCurriculum"
  echo "checkpoint=from_scratch"
  echo "num_envs=24576"
  echo "max_iterations=2000"
  echo "started_at=$(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} | tee "$out_dir/meta.txt"

/opt/conda/bin/python train.py \
  --task T1CircleGridCurriculum \
  --num_envs 24576 \
  --max_iterations 2000 \
  --headless True \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  2>&1 | tee "$out_dir/train.log"
