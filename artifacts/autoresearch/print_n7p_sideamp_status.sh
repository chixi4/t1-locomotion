#!/usr/bin/env bash
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official || exit 0

LOG="$(ls -t artifacts/autoresearch/T1Shoulder4GaitPhaseNightN7PitchBoostSideAmpConservative*_progress.log 2>/dev/null | head -1)"
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
PROC="$(ps -eo pid,etime,cmd | grep -E 'train_shoulder4_frozen.py.*SideAmp|run_n7p_sideamp300_progress.sh' | grep -v grep | sed -n '1,2p')"

echo "T1 SideAmp Progress Dashboard"
echo "============================================================"
echo "Run:   T1Shoulder4 NightN7 PitchBoost SideAmp model_300"
echo "Patch: side roll extra 0.38 -> 0.46 | max 0.30 -> 0.34 | min 0.08 -> 0.10"
echo "Base:  roll base stays 0.0, pitch amp stays 0.84"
echo
echo "Epoch: ${CUR}/${TOTAL}   ${PCT}%   elapsed ${ELAPSED}m   ETA ${ETA}m"
echo "[${BAR}]"
if [ -n "$EVAL_PROGRESS" ] && printf '%s\n' "$PHASE" | grep -q 'START eval'; then
  echo
  echo "Eval:  ${EVAL_PROGRESS}"
elif [ -n "$EVAL_PROGRESS" ] && ! printf '%s\n' "$PHASE" | grep -q 'DONE run'; then
  echo
  echo "Eval:  ${EVAL_PROGRESS}"
fi
echo
[ -n "$PHASE" ] && echo "Phase: $PHASE"
[ -n "$GPU" ] && echo "GPU:   $GPU"
[ -n "$LOG" ] && echo "Log:   $LOG"
echo
echo "Processes:"
[ -n "$PROC" ] && printf '%s\n' "$PROC" || echo "No training process found"
echo
echo "Recent progress:"
echo "------------------------------------------------------------"
if [ -n "$LOG" ]; then
  grep -E 'epoch:|random replay progress|fixed eval progress|fixed replay progress|==== START|==== DONE|RUN_FAILED|DONE run' "$LOG" | tail -16
else
  echo "Waiting for training log..."
fi
echo
echo "Outputs:"
echo "------------------------------------------------------------"
ls -lh artifacts/autoresearch/shoulder4_night_n7_pitchboost_sideamp300_* 2>/dev/null | sed -n '1,12p' || true
echo
echo "Monitor only. Closing this window will not stop training."
