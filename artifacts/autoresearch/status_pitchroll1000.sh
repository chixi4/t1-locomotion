cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
printf '%s\n' '--- session ---'
tmux has-session -t t1_pitchroll1000 2>/dev/null && echo tmux_running || echo tmux_missing
ps -eo pid,ppid,stat,etime,cmd | grep -E 'train_shoulder4_frozen|t1_pitchroll1000|python' | grep -v grep || true
printf '%s\n' '--- log_tail ---'
tail -50 artifacts/autoresearch/T1Shoulder4Pitch09Roll08_train1000_env32768_20260507_084952.log || true
printf '%s\n' '--- latest logs ---'
ls -td logs/2026-05-07-* 2>/dev/null | head -8 || true
printf '%s\n' '--- gpu ---'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || true
