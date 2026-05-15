#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

echo "---benchmark results---"
for f in \
  artifacts/autoresearch/pitchroll_env_benchmark/results.tsv \
  artifacts/autoresearch/pitchroll_env_benchmark/fine_results.tsv \
  artifacts/autoresearch/pitchroll_env_benchmark/select_results.tsv
do
  echo "### $f"
  [[ -f "$f" ]] && cat "$f" || echo missing
done

echo "---launch scripts---"
for f in artifacts/autoresearch/start_pitchroll1000_nohup.sh /tmp/start_pitchroll1000_nohup.sh /tmp/run_pitchroll1000_inner.sh /tmp/start_pitchroll1000_stable.sh; do
  echo "### $f"
  [[ -f "$f" ]] && sed -n '1,180p' "$f" || echo missing
done

echo "---pitchroll logs---"
ls -lh artifacts/autoresearch/T1Shoulder4Pitch09Roll08_train1000_env32768_* 2>/dev/null || true
for f in artifacts/autoresearch/T1Shoulder4Pitch09Roll08_train1000_env32768_*; do
  [[ -f "$f" ]] || continue
  echo "### tail $f"
  tail -80 "$f"
done

echo "---latest run train logs---"
for d in $(ls -td logs/2026-05-07-* 2>/dev/null | head -8); do
  echo "### $d"
  find "$d" -maxdepth 2 -type f | sort
  [[ -f "$d/train.log" ]] && tail -60 "$d/train.log" || true
done
