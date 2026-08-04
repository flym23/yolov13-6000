#!/usr/bin/env bash
# Detach the DAD post-GSD waiter and retain the launcher log and PID.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
STATE="$ROOT/runs/dad_yolov13_a1_a7"
RUN_ID="${DAD_YOLOV13_RUN_ID:-dad_yolov13_$(date -u +%Y%m%d_%H%M%S)}"
LAUNCH_TAG="${DAD_YOLOV13_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$STATE/launcher_${RUN_ID}_${LAUNCH_TAG}.log"

mkdir -p "$STATE"
DAD_YOLOV13_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_dad_yolov13_a1_a7_after_gsd.sh" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$STATE/nohup_waiter_${RUN_ID}_${LAUNCH_TAG}.pid"
echo "DAD_YOLOV13_RUN_ID=$RUN_ID"
echo "DAD_YOLOV13_WAITER_PID=$PID"
echo "DAD_YOLOV13_WAITER_LOG=$LOG"
