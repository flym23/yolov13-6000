#!/usr/bin/env bash
# Detach the CPCR-dependent SPC-LCER-DCRA chain while preserving its PID and log.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/runs/spc_lcer_dcra_p0_p3"
RUN_ID="${SPC_LCER_DCRA_RUN_ID:-spc_lcer_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LAUNCH_TAG="${SPC_LCER_DCRA_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$STATE/launcher_${RUN_ID}_${LAUNCH_TAG}.log"

mkdir -p "$STATE"
SPC_LCER_DCRA_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_spc_lcer_dcra_p0_p3_after_cpcr.sh" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}_${LAUNCH_TAG}.pid"
echo "SPC_LCER_DCRA_RUN_ID=$RUN_ID"
echo "SPC_LCER_DCRA_LAUNCHER_PID=$PID"
echo "SPC_LCER_DCRA_LAUNCHER_LOG=$LOG"
