#!/usr/bin/env bash
# Detach the CBER-dependent SAMR-LCER-DCRA chain while preserving its PID and log.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/runs/samr_lcer_dcra_s1_s3"
RUN_ID="${SAMR_LCER_DCRA_RUN_ID:-samr_lcer_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LAUNCH_TAG="${SAMR_LCER_DCRA_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$STATE/launcher_${RUN_ID}_${LAUNCH_TAG}.log"

mkdir -p "$STATE"
SAMR_LCER_DCRA_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_samr_lcer_dcra_s1_s3_after_cber.sh" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}_${LAUNCH_TAG}.pid"
echo "SAMR_LCER_DCRA_RUN_ID=$RUN_ID"
echo "SAMR_LCER_DCRA_LAUNCHER_PID=$PID"
echo "SAMR_LCER_DCRA_LAUNCHER_LOG=$LOG"
