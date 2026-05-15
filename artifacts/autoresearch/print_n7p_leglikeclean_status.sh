#!/usr/bin/env bash
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official || exit 0

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3}"
TASK="T1Shoulder4GaitPhaseNightN7PitchBoostSideAmpLegLikeClean_from7000LegFrozen_train300"
LABEL="shoulder4_night_n7p_sideamp_leglikeclean300"
LOG="$(ls -t artifacts/autoresearch/${TASK}_*_progress.log 2>/dev/null | head -1)"
EPOCH=""
PHASE=""
EVAL_PROGRESS=""
if [ -n "$LOG" ]; then
  EPOCH="$(grep -E '^epoch:' "$LOG" | tail -1)"
  PHASE="$(grep -E '==== START|==== DONE|RUN_FAILED|DONE run' "$LOG" | tail -1)"
  EVAL_PROGRESS="$(grep -E 'random replay progress|fixed eval progress|fixed replay progress' "$LOG" | tail -1)"
fi

CUR=0
TOTAL=300
ELAPSED="?"
ETA="?"
PCT="0.0"
if [ -n "$EPOCH" ]; then
  CUR="$(printf '%s\n' "$EPOCH" | sed -n 's/^epoch: \([0-9][0-9]*\)\/\([0-9][0-9]*\).*elapsed=\([0-9.][0-9.]*\)m eta=\([0-9.][0-9.]*\)m.*/\1/p')"
  TOTAL="$(printf '%s\n' "$EPOCH" | sed -n 's/^epoch: \([0-9][0-9]*\)\/\([0-9][0-9]*\).*elapsed=\([0-9.][0-9.]*\)m eta=\([0-9.][0-9.]*\)m.*/\2/p')"
  ELAPSED="$(printf '%s\n' "$EPOCH" | sed -n 's/^epoch: \([0-9][0-9]*\)\/\([0-9][0-9]*\).*elapsed=\([0-9.][0-9.]*\)m eta=\([0-9.][0-9.]*\)m.*/\3/p')"
  ETA="$(printf '%s\n' "$EPOCH" | sed -n 's/^epoch: \([0-9][0-9]*\)\/\([0-9][0-9]*\).*elapsed=\([0-9.][0-9.]*\)m eta=\([0-9.][0-9.]*\)m.*/\4/p')"
  [ -z "$CUR" ] && CUR=0
  [ -z "$TOTAL" ] && TOTAL=300
  PCT="$(awk -v c="$CUR" -v t="$TOTAL" 'BEGIN { if (t > 0) printf "%.1f", c * 100.0 / t; else printf "0.0" }')"
fi

BAR_WIDTH=40
FILLED="$(awk -v c="$CUR" -v t="$TOTAL" -v w="$BAR_WIDTH" 'BEGIN { if (t > 0) printf "%d", c * w / t; else printf "0" }')"
BAR=""
i=0
while [ "$i" -lt "$BAR_WIDTH" ]; do
  if [ "$i" -lt "$FILLED" ]; then BAR="${BAR}#"; else BAR="${BAR}."; fi
  i=$((i + 1))
done

GPU="$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
PROC="$(ps -eo pid,etime,cmd | grep -E 'train_shoulder4_frozen.py.*LegLikeClean|run_n7p_leglikeclean300_progress.sh' | grep -v grep | sed -n '1,3p')"
SCALARS=""
if [ -n "$LOG" ]; then
  SCALARS="$($PYTHON_BIN - "$TASK" <<'PY' 2>/dev/null
from pathlib import Path
import sys
import yaml
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    raise SystemExit(0)

task = sys.argv[1]
candidates = []
for cfg_path in Path("logs").glob("*/config.yaml"):
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if cfg.get("basic", {}).get("task") == task:
        candidates.append(cfg_path.parent)
if not candidates:
    raise SystemExit(0)
run_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
summary_dir = run_dir / "summaries"
if not summary_dir.exists():
    raise SystemExit(0)
ea = EventAccumulator(str(summary_dir), size_guidance={"scalars": 0})
ea.Reload()
tags = set(ea.Tags().get("scalars", []))
want = [
    "reward", "steps", "value_loss", "kl_mean", "lr", "policy/mean_abs_max",
    "curriculum/active_tilt_rms_mean", "curriculum/active_arm_saturation_frac_mean",
    "episode/shoulder_static_posture", "episode/shoulder_dynamic_target",
    "episode/shoulder_no_lazy_boundary", "episode/shoulder_balance_smoothness",
]
print(f"RunDir: {run_dir}")
for tag in want:
    if tag not in tags:
        continue
    vals = ea.Scalars(tag)
    if not vals:
        continue
    ev = vals[-1]
    print(f"{tag}: step {ev.step} value {ev.value:.6g}")
PY
)"
fi

DONE=""
[ -f "artifacts/autoresearch/${LABEL}_full_auto_done.json" ] && DONE="done"
FAILED=""
[ -f "artifacts/autoresearch/${LABEL}_full_auto_failed.json" ] && FAILED="failed"

echo "T1 LegLikeClean Progress Dashboard"
echo "======================================================================"
echo "Run:   T1Shoulder4 NightN7P SideAmp LegLikeClean model_300"
echo "Start: SideAmp model_300 + frozen 7000 leg, optimizer reset"
echo "Patch: episode 10s -> 30s | command resample 100s -> 8-12s"
echo "Reward: survival/tracking direct rewards -> 0; shoulder style as guardrails"
echo "Arm:   pitch_amp 0.84, roll_base 0.0, roll_extra 0.46, roll_max 0.34, min_out 0.10"
echo
echo "Epoch: ${CUR}/${TOTAL}   ${PCT}%   elapsed ${ELAPSED}m   ETA ${ETA}m"
echo "[${BAR}]"
if [ -n "$EVAL_PROGRESS" ]; then
  echo
  echo "Eval:  ${EVAL_PROGRESS}"
fi
echo
[ -n "$PHASE" ] && echo "Phase: $PHASE"
[ -n "$GPU" ] && echo "GPU:   $GPU"
[ -n "$LOG" ] && echo "Log:   $LOG"
if [ -n "$DONE" ]; then echo "State: DONE"; fi
if [ -n "$FAILED" ]; then echo "State: FAILED"; fi
echo
echo "Processes:"
[ -n "$PROC" ] && printf '%s\n' "$PROC" || echo "No training process found"
echo
echo "Latest scalars:"
echo "----------------------------------------------------------------------"
[ -n "$SCALARS" ] && printf '%s\n' "$SCALARS" | sed -n '1,14p' || echo "Waiting for TensorBoard scalars..."
echo
echo "Recent progress:"
echo "----------------------------------------------------------------------"
if [ -n "$LOG" ]; then
  grep -E 'epoch:|random replay progress|fixed eval progress|fixed replay progress|==== START|==== DONE|RUN_FAILED|DONE run|CONFIG ' "$LOG" | tail -18
else
  echo "Waiting for training log..."
fi
echo
echo "Outputs:"
echo "----------------------------------------------------------------------"
ls -lh artifacts/autoresearch/${LABEL}_* 2>/dev/null | sed -n '1,12p' || true
echo
echo "Monitor only. Closing this window will not stop training."
