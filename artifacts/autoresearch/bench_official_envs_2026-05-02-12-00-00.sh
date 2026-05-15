#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
export PATH=/opt/conda/bin:/opt/conda/lib/python3.8/site-packages/ninja/data/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/conda/lib:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH:-}
export WANDB_MODE=disabled
mkdir -p artifacts/autoresearch
out="artifacts/autoresearch/official_env_benchmark_2026-05-02-12-00-00.tsv"
printf 'timestamp\tnum_envs\titerations\trc\tepochs_seen\tepoch_sec_mean\tepoch_sec_tail\ttransitions_per_sec_tail\tmax_mem_mb\tmax_gpu_util\tlog\tnotes\n' > "$out"
run_one() {
  local n="$1"
  local iters="$2"
  local log="artifacts/autoresearch/bench_env${n}_$(date +%Y%m%d_%H%M%S).log"
  local gpu_log="${log%.log}_gpu.csv"
  echo "== benchmark num_envs=${n} iters=${iters} log=${log} =="
  (while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$gpu_log" 2>/dev/null || true; sleep 1; done) &
  local mon=$!
  set +e
  timeout 900 bash -lc "/opt/conda/bin/python -u train.py --task T1 --headless true --num_envs ${n} --max_iterations ${iters} 2>&1 | while IFS= read -r line; do printf '%s %s\\n' \"\$(date +%s.%N)\" \"\$line\"; done" > "$log"
  local rc=$?
  set -e
  kill "$mon" 2>/dev/null || true
  wait "$mon" 2>/dev/null || true
  /opt/conda/bin/python - <<PY
from pathlib import Path
import re, statistics, csv
n=$n; iters=$iters; rc=$rc; log=Path('$log'); gpu=Path('$gpu_log')
pat=re.compile(r'^(\d+\.\d+) epoch: (\d+)/(\d+)')
t=[]
for line in log.read_text(errors='ignore').splitlines():
    m=pat.match(line)
    if m:
        t.append((float(m.group(1)), int(m.group(2))))
deltas=[t[i][0]-t[i-1][0] for i in range(1,len(t))]
# skip the first two observed epoch gaps to reduce init/cache noise
stable=deltas[2:] if len(deltas)>4 else deltas
tail=stable[-5:] if len(stable)>=5 else stable
mean=sum(stable)/len(stable) if stable else 0.0
tail_mean=sum(tail)/len(tail) if tail else 0.0
tps=(n*24/tail_mean) if tail_mean>0 else 0.0
max_mem=0; max_util=0
if gpu.exists():
    for row in csv.reader(gpu.open(errors='ignore')):
        if len(row)>=3:
            try:
                max_mem=max(max_mem, int(float(row[1].strip())))
                max_util=max(max_util, int(float(row[2].strip())))
            except Exception:
                pass
notes='ok' if rc==0 else ('timeout' if rc==124 else 'failed')
print(f"{n}\tepochs={len(t)}\trc={rc}\ttail_sec={tail_mean:.4f}\ttps={tps:.0f}\tmem={max_mem}\tutil={max_util}\t{notes}")
with open('$out','a',encoding='utf-8') as f:
    f.write(f"$(date -Is)\t{n}\t{iters}\t{rc}\t{len(t)}\t{mean:.6f}\t{tail_mean:.6f}\t{tps:.3f}\t{max_mem}\t{max_util}\t{log}\t{notes}\n")
PY
}
for n in 8192 16384 24576 32768 40960; do
  run_one "$n" 12
  sleep 5
  pkill -f '/opt/conda/bin/python -u train.py --task T1' || true
  sleep 5
done
printf '\n== benchmark table ==\n'
cat "$out"
