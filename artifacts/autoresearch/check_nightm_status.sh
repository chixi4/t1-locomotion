#!/usr/bin/env bash
set -uo pipefail

cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official || exit 1

echo "=== time ==="
date '+%Y-%m-%d %H:%M:%S'

echo "=== tmux sessions ==="
tmux list-sessions 2>&1 || true

echo "=== tmux panes ==="
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} dead=#{pane_dead} cmd=#{pane_current_command}' 2>&1 || true

echo "=== tmux capture ==="
tmux capture-pane -t nightm_zerobase300 -p -S -120 2>&1 || true

echo "=== process matches ==="
python3 - <<'PY'
import subprocess

out = subprocess.run(["ps", "-eo", "pid,ppid,stat,etime,args"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
needles = ("train_shoulder4_frozen", "run_nightm", "NightMZeroBase", "nightm_zerobase300", "eval_shoulder4")
for line in out.splitlines():
    if any(n in line for n in needles):
        print(line)
PY

echo "=== nvidia-smi summary ==="
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>&1 || true

echo "=== nvidia-smi compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true

echo "=== nightm files ==="
python3 - <<'PY'
from pathlib import Path
patterns = [
    "artifacts/autoresearch/nightm_zerobase300*",
    "artifacts/autoresearch/T1Shoulder4GaitPhaseNightMZeroBaseDynamic_from7000LegFrozen_train300*",
    "artifacts/autoresearch/shoulder4_night_m_zerobase_dynamic300*",
]
files = []
for pattern in patterns:
    files.extend(Path(".").glob(pattern))
for p in sorted(set(files), key=lambda x: x.stat().st_mtime, reverse=True)[:80]:
    st = p.stat()
    print(f"{st.st_mtime:.0f} {st.st_size:>10} {p}")
PY

echo "=== latest logs dirs ==="
python3 - <<'PY'
from pathlib import Path
for p in sorted(Path("logs").iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
    print(p, p.stat().st_mtime)
PY

echo "=== tmux log tail ==="
tail -n 160 artifacts/autoresearch/nightm_zerobase300_tmux.out 2>&1 || true

echo "=== train log tail ==="
latest=$(ls -t artifacts/autoresearch/T1Shoulder4GaitPhaseNightMZeroBaseDynamic_from7000LegFrozen_train300_*_full_auto.log 2>/dev/null | head -n 1)
if [[ -n "${latest:-}" ]]; then
  echo "--- ${latest} ---"
  tail -n 160 "${latest}" 2>&1 || true
fi
