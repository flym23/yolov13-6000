#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rom305/zzf/yolov13-305
STATE=$ROOT/runs/wsdr_ablation
RUN_ID=${WSDR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=$STATE/launcher_${RUN_ID}.log

mkdir -p "$STATE"
cd "$ROOT"
WSDR_RUN_ID="$RUN_ID" nohup setsid "$ROOT/run_wsdr_ablation_parallel.sh" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
echo "WSDR_RUN_ID=$RUN_ID"
echo "WSDR_LAUNCHER_PID=$PID"
echo "WSDR_LAUNCHER_LOG=$LOG"
