#!/usr/bin/env bash
# Detach the DGMR-LCER-DCRA D1--D4 post-SBT chain and preserve its launcher log.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/runs/dgmr_lcer_dcra_d1_d4"
RUN_ID="${DGMR_LCER_DCRA_RUN_ID:-dgmr_lcer_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LAUNCH_TAG="${DGMR_LCER_DCRA_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$STATE/launcher_${RUN_ID}_${LAUNCH_TAG}.log"

mkdir -p "$STATE"
DGMR_LCER_DCRA_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_dgmr_lcer_dcra_d1_d4_after_sbt.sh" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}_${LAUNCH_TAG}.pid"
echo "DGMR_LCER_DCRA_RUN_ID=$RUN_ID"
echo "DGMR_LCER_DCRA_LAUNCHER_PID=$PID"
echo "DGMR_LCER_DCRA_LAUNCHER_LOG=$LOG"
