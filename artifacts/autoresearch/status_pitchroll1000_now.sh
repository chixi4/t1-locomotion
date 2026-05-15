cd /mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official
echo ---session---
tmux has-session -t t1_pitchroll1000 2>/dev/null && echo tmux_running || echo tmux_missing
tmux capture-pane -pt t1_pitchroll1000 -S -80 2>/dev/null || true
echo ---tail---
tail -80 artifacts/autoresearch/T1Shoulder4Pitch09Roll08_train1000_env32768_20260507_0900.log 2>/dev/null || true
echo ---proc---
ps -eo pid,ppid,stat,etime,cmd | grep -E 'train_shoulder4_frozen|t1_pitchroll1000|python' | grep -v grep || true
echo ---gpu---
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || true
