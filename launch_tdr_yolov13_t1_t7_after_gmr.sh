#!/usr/bin/env bash
# Detach the TDR post-GMR waiter and preserve its PID and launcher log.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
STATE="$ROOT/runs/tdr_yolov13_t1_t7"
RUN_ID=${TDR_YOLOV13_RUN_ID:-tdr_yolov13_$(date -u +%Y%m%d_%H%M%S)}
LOG="$STATE/launcher_${RUN_ID}.log"

mkdir -p "$STATE"
TDR_YOLOV13_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_tdr_yolov13_t1_t7_after_gmr.sh" < /dev/null >"$LOG" 2>&1 &
PID=$!
echo "$PID" >"$STATE/nohup_waiter_${RUN_ID}.pid"
echo "TDR_YOLOV13_RUN_ID=$RUN_ID"
echo "TDR_YOLOV13_WAITER_PID=$PID"
echo "TDR_YOLOV13_WAITER_LOG=$LOG"
