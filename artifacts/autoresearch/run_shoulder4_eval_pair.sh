#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 --task TASK --label LABEL --arm_checkpoint PATH --leg_checkpoint PATH [options]

Options:
  --out_dir DIR              Default: artifacts/autoresearch/LABEL_eval
  --num_envs N               Default: 256
  --fixed_duration_s S       Default: 10
  --warmup_s S               Default: 1
  --random_seconds S         Default: 60
  --fps N                    Default: 50
  --clean                    Pass --clean to eval script

Runs the random replay needed for WebGL animation and writes LABEL_summary.json
in OUT_DIR. NightK uses this fast replay-only path to avoid the slow 31-command
fixed eval during visual iteration.
USAGE
}

TASK=""
LABEL=""
ARM_CKPT=""
LEG_CKPT=""
OUT_DIR=""
NUM_ENVS=256
FIXED_DURATION_S=10
WARMUP_S=1
RANDOM_SECONDS=60
FPS=50
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --arm_checkpoint) ARM_CKPT="$2"; shift 2 ;;
    --leg_checkpoint) LEG_CKPT="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --fixed_duration_s) FIXED_DURATION_S="$2"; shift 2 ;;
    --warmup_s) WARMUP_S="$2"; shift 2 ;;
    --random_seconds) RANDOM_SECONDS="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$TASK" || -z "$LABEL" || -z "$ARM_CKPT" || -z "$LEG_CKPT" ]]; then
  usage >&2
  exit 2
fi

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
OUT_DIR="${OUT_DIR:-artifacts/autoresearch/${LABEL}_eval}"
LOG_RANDOM="artifacts/autoresearch/${LABEL}_random_eval_run.log"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${FAST_REPLAY_ONLY:-1}" == "1" ]]; then
  RANDOM_SECONDS=60
  FPS=50
fi

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/isaacgym/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
CLEAN_ARGS=()
if [[ "$CLEAN" == "1" ]]; then CLEAN_ARGS+=(--clean); fi

echo "SKIP fixed $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} reason=replay_only_visual_iteration"

echo "START random $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL}" | tee "$LOG_RANDOM"
RANDOM_START=$(date +%s)
"${PYTHON_BIN}" -u artifacts/autoresearch/eval_shoulder4_frozen.py \
  --task "$TASK" \
  --arm_checkpoint "$ARM_CKPT" \
  --leg_checkpoint "$LEG_CKPT" \
  --out_dir "$OUT_DIR" \
  --label "$LABEL" \
  --num_envs "$NUM_ENVS" \
  --fixed_duration_s "$FIXED_DURATION_S" \
  --warmup_s "$WARMUP_S" \
  --random_seconds "$RANDOM_SECONDS" \
  --fps "$FPS" \
  --mode random \
  "${CLEAN_ARGS[@]}" 2>&1 | tee -a "$LOG_RANDOM"
RANDOM_END=$(date +%s)
echo "DONE random $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL} elapsed=$((RANDOM_END - RANDOM_START))s" | tee -a "$LOG_RANDOM"

echo "DONE $(date '+%Y-%m-%d %H:%M:%S') label=${LABEL}"
cat "${OUT_DIR}/${LABEL}_summary.json"
