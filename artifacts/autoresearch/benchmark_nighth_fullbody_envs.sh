#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official

TASK=T1Shoulder4NightHFullBodyGrid15Scratch_train5000
ITERATIONS="${ITERATIONS:-6}"
OUT_TSV="artifacts/autoresearch/nighth_fullbody_env_benchmark_$(date +%Y%m%d_%H%M%S).tsv"
ENVS=("$@")
if [[ ${#ENVS[@]} -eq 0 ]]; then
  ENVS=(28672 30720)
fi

export PATH="/opt/conda/bin:/usr/lib/wsl/lib:${PATH}"
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/wsl/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/autoresearch
printf "timestamp\tnum_envs\titerations\trc\tepochs_seen\tepoch_sec_tail\ttransitions_per_sec_tail\tlog\n" > "${OUT_TSV}"

for envs in "${ENVS[@]}"; do
  run_name="${TASK}_bench_env${envs}_$(date +%Y%m%d_%H%M%S)"
  log="artifacts/autoresearch/${run_name}.log"
  echo "BENCH_START envs=${envs} iterations=${ITERATIONS} run=${run_name}"
  set +e
  /opt/conda/bin/python -u artifacts/autoresearch/train_fullbody.py \
    --task "${TASK}" \
    --num_envs "${envs}" \
    --max_iterations "${ITERATIONS}" \
    --run_name "${run_name}" \
    --headless true 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  python3 - "${envs}" "${ITERATIONS}" "${rc}" "${log}" "${OUT_TSV}" <<'PY'
import re
import sys
import time
from pathlib import Path

envs = int(sys.argv[1])
iterations = int(sys.argv[2])
rc = int(sys.argv[3])
log = Path(sys.argv[4])
out_tsv = Path(sys.argv[5])
epochs = []
for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"epoch:\s+(\d+)/\d+\s+elapsed_s=([0-9.]+)", line)
    if match:
        epochs.append((int(match.group(1)), float(match.group(2))))

tail = 0.0
transitions = 0.0
if len(epochs) >= 3:
    recent = epochs[-3:]
    tail = (recent[-1][1] - recent[0][1]) / max(1, recent[-1][0] - recent[0][0])
elif len(epochs) >= 2:
    tail = (epochs[-1][1] - epochs[0][1]) / max(1, epochs[-1][0] - epochs[0][0])
if tail > 0:
    transitions = envs * 24 / tail

row = [
    time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    str(envs),
    str(iterations),
    str(rc),
    str(len(epochs)),
    f"{tail:.6f}",
    f"{transitions:.3f}",
    str(log),
]
with out_tsv.open("a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
print("\t".join(row), flush=True)
PY
  echo "BENCH_DONE envs=${envs} rc=${rc} log=${log}"
  if [[ ${rc} -ne 0 ]]; then
    exit "${rc}"
  fi
done

echo "BENCH_TSV=${OUT_TSV}"
cat "${OUT_TSV}"
