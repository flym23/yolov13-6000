#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/rom305/zzf/yolov13-305"
STATE_DIR="$ROOT/runs/aarg_ablation"
RUN_ID="${AARG_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG="$STATE_DIR/launcher_${RUN_ID}.log"

mkdir -p "$STATE_DIR"
cd "$ROOT"
AARG_RUN_ID="$RUN_ID" nohup setsid "$ROOT/run_aarg_ablation_parallel.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE_DIR/nohup_launcher_${RUN_ID}.pid"
echo "$RUN_ID" > "$STATE_DIR/run_id.txt"
echo "AARG_RUN_ID=$RUN_ID"
echo "AARG_LAUNCHER_PID=$PID"
echo "AARG_LAUNCHER_LOG=$LOG"
