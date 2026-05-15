#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

export PYTHONPATH=.

/opt/conda/bin/python artifacts/autoresearch/probe_residual_checkpoint.py \
  --task T1Upper9CameraStableOfficialOpenLeg18S8ZeroResidual_eval \
  --checkpoints logs/2026-05-14-15-08-00/nn/model_500.pth \
  --num-envs 128 \
  --seconds 5.0 \
  --warmup-s 1.5 \
  --out artifacts/autoresearch/s8_zero_residual_probe.json
