#!/usr/bin/env bash
# Detach the LSMR-LCER-DCRA R1--R3 chain while preserving PID and launcher log.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/runs/lsmr_lcer_dcra_r1_r3"
RUN_ID="${LSMR_LCER_DCRA_RUN_ID:-lsmr_lcer_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LAUNCH_TAG="${LSMR_LCER_DCRA_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$STATE/launcher_${RUN_ID}_${LAUNCH_TAG}.log"

mkdir -p "$STATE"
LSMR_LCER_DCRA_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_lsmr_lcer_dcra_r1_r3.sh" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}_${LAUNCH_TAG}.pid"
echo "LSMR_LCER_DCRA_RUN_ID=$RUN_ID"
echo "LSMR_LCER_DCRA_LAUNCHER_PID=$PID"
echo "LSMR_LCER_DCRA_LAUNCHER_LOG=$LOG"
