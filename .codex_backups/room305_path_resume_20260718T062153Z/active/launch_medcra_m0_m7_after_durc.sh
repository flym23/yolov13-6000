#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rtx6000/ZZF/yolov13-6000
STATE=$ROOT/runs/urpc2020_medcra_m0_m7
RUN_ID=${MEDCRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=$STATE/launcher_${RUN_ID}.log

mkdir -p "$STATE"
cd "$ROOT"
MEDCRA_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_medcra_m0_m7_after_durc.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}.pid"
echo "MEDCRA_RUN_ID=$RUN_ID"
echo "MEDCRA_LAUNCHER_PID=$PID"
echo "MEDCRA_LAUNCHER_LOG=$LOG"
