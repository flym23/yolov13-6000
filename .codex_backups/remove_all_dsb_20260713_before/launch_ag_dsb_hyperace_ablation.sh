#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rom305/zzf/yolov13-305
STATE=$ROOT/runs/ag_dsb_hyperace_ablation
RUN_ID=${AGDSB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$STATE"
cd "$ROOT"
AGDSB_RUN_ID="$RUN_ID" nohup setsid "$ROOT/run_ag_dsb_hyperace_ablation_parallel.sh" \
  > "$STATE/launcher_${RUN_ID}.log" 2>&1 < /dev/null &
echo $! > "$STATE/launcher_bootstrap.pid"
echo "$RUN_ID" > "$STATE/launch_run_id.txt"
echo "AGDSB_RUN_ID=$RUN_ID"
echo "LAUNCHER_BOOTSTRAP_PID=$!"
echo "LOG=$STATE/launcher_${RUN_ID}.log"
