#!/usr/bin/env bash
# Detach the simple TDR D0 launcher so it continues after SSH disconnects.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/runs/tdr_urpc2019"
RUN_ID="${TDR_URPC2019_RUN_ID:-tdr_urpc2019_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$STATE"
LOG="$STATE/launcher_${RUN_ID}.log"
TDR_URPC2019_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_tdr_urpc2019_after_yolov26.sh" < /dev/null > "$LOG" 2>&1 &
PID="$!"
printf '%s\n' "$PID" > "$STATE/launcher_${RUN_ID}.pid"
printf 'run_id=%s\nlauncher_pid=%s\nlauncher_log=%s\n' "$RUN_ID" "$PID" "$LOG"
