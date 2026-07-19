#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rom305/zzf/yolov13-305
STATE=$ROOT/runs/dcra_ablation
RUN_ID=${DCRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=$STATE/launcher_${RUN_ID}.log

mkdir -p "$STATE"
cd "$ROOT"
DCRA_RUN_ID="$RUN_ID" nohup setsid "$ROOT/run_dcra_ablation_parallel.sh" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
echo "DCRA_RUN_ID=$RUN_ID"
echo "DCRA_LAUNCHER_PID=$PID"
echo "DCRA_LAUNCHER_LOG=$LOG"
