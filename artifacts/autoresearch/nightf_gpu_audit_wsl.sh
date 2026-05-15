#!/usr/bin/env bash
set -uo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official || exit 1

OUT_ROOT="artifacts/autoresearch/gpu_train_audit/nightf_wsl_$(date +%Y%m%d_%H%M%S)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-2}"
MAX_SECONDS="${MAX_SECONDS:-36000}"
QUIET_SECONDS="${QUIET_SECONDS:-900}"

mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/gpu_summary.csv"
COMPUTE_LOG="${OUT_ROOT}/nvidia_compute_apps.log"
PROCESS_LOG="${OUT_ROOT}/process_snapshots.log"
MARKER="${OUT_ROOT}/logger_running.txt"

echo "started=$(date -Is) pid=$$ interval=${INTERVAL_SECONDS}s max=${MAX_SECONDS}s" | tee "${MARKER}"
echo "timestamp,gpu_timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,power_w,temp_c" > "${SUMMARY_CSV}"

start_ts=$(date +%s)
last_busy_ts="${start_ts}"

while true; do
  now_ts=$(date +%s)
  if (( now_ts - start_ts >= MAX_SECONDS )); then
    echo "finished=$(date -Is) reason=max_seconds last_busy=$(date -Is -d "@${last_busy_ts}" 2>/dev/null || date -Is)" >> "${MARKER}"
    exit 0
  fi

  line=$(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "${line}" ]]; then
    IFS=',' read -r gpu_time util mem_used mem_total power temp <<< "${line}"
    util=${util// /}
    mem_used=${mem_used// /}
    echo "$(date -Is),${gpu_time},${util},${mem_used},${mem_total// /},${power// /},${temp// /}" >> "${SUMMARY_CSV}"
    if [[ "${mem_used}" =~ ^[0-9]+$ ]] && (( mem_used >= 1024 )); then
      last_busy_ts="${now_ts}"
    fi
    if [[ "${util}" =~ ^[0-9]+$ ]] && (( util >= 5 )); then
      last_busy_ts="${now_ts}"
    fi
  fi

  if (( (now_ts - start_ts) % 10 < INTERVAL_SECONDS )); then
    {
      echo "===== $(date -Is) nvidia-smi compute apps ====="
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
    } >> "${COMPUTE_LOG}"
    {
      echo "===== $(date -Is) ps ====="
      ps -eo pid,ppid,stat,pcpu,pmem,comm,args --sort=-pcpu | head -80
    } >> "${PROCESS_LOG}"
  fi

  if (( now_ts - last_busy_ts >= QUIET_SECONDS )); then
    echo "finished=$(date -Is) reason=quiet last_busy=$(date -Is -d "@${last_busy_ts}" 2>/dev/null || date -Is)" >> "${MARKER}"
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done
