#!/usr/bin/env bash
set -euo pipefail
pkill -f "T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000" || true
pkill -f "train_shoulder4_frozen.py" || true
echo "remaining:"
ps -eo pid,ppid,stat,etime,cmd | grep -E "train_shoulder4_frozen|T1Shoulder4Pitch09Roll08" | grep -v grep || true
