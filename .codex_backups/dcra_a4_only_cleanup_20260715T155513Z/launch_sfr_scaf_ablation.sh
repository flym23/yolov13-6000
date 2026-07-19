#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/rom305/zzf/yolov13-305"
STATE_DIR="$ROOT/runs/sfr_scaf_ablation"
RUN_ID="${SFR_SCAF_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG="$STATE_DIR/launcher_${RUN_ID}.log"

mkdir -p "$STATE_DIR"
cd "$ROOT"
SFR_SCAF_RUN_ID="$RUN_ID" nohup setsid "$ROOT/run_sfr_scaf_ablation_parallel.sh" > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$STATE_DIR/launcher_bootstrap.pid"
echo "$RUN_ID" > "$STATE_DIR/launch_run_id.txt"
echo "SFR_SCAF_RUN_ID=$RUN_ID"
echo "LAUNCHER_BOOTSTRAP_PID=$!"
echo "LOG=$LOG"
