#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/room305/ZZF/yolov13-6000
STATE=$ROOT/runs/urpc2020_baseline_a4
RUN_ID=${URPC_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=$STATE/launcher_${RUN_ID}.log
RESUME_FROM=${URPC_RESUME_FROM:-}
SKIP_SMOKE=${URPC_SKIP_SMOKE:-false}

mkdir -p "$STATE"
cd "$ROOT"
URPC_RUN_ID="$RUN_ID" URPC_RESUME_FROM="$RESUME_FROM" URPC_SKIP_SMOKE="$SKIP_SMOKE" \
  nohup setsid /bin/bash "$ROOT/run_dcra_ablation_parallel.sh" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
echo "URPC_RUN_ID=$RUN_ID"
echo "URPC_LAUNCHER_PID=$PID"
echo "URPC_LAUNCHER_LOG=$LOG"
echo "URPC_RESUME_FROM=${RESUME_FROM:-none}"
echo "URPC_SKIP_SMOKE=$SKIP_SMOKE"
