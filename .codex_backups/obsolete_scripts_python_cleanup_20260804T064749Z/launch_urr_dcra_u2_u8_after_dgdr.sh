#!/usr/bin/env bash
# Detach the SCPG-dependent U2--U8 launcher while preserving state and logs.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/runs/urr_dcra_u2_u8"
RUN_ID="${URR_DCRA_RUN_ID:-urr_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LAUNCH_TAG="${URR_DCRA_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$STATE/launcher_${RUN_ID}_${LAUNCH_TAG}.log"

mkdir -p "$STATE"
URR_DCRA_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_urr_dcra_u2_u8_after_dgdr.sh" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}_${LAUNCH_TAG}.pid"
echo "URR_DCRA_RUN_ID=$RUN_ID"
echo "URR_DCRA_LAUNCHER_PID=$PID"
echo "URR_DCRA_LAUNCHER_LOG=$LOG"
