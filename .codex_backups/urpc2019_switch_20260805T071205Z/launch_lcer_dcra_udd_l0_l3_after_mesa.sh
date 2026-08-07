#!/usr/bin/env bash
# Detach the UDD L0/L3 chain; it waits internally for the MESA chain to exit.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/runs/lcer_dcra_udd_l0_l3"
RUN_ID="${LCER_DCRA_UDD_RUN_ID:-lcer_dcra_udd_$(date -u +%Y%m%d_%H%M%S)}"
LAUNCH_TAG="${LCER_DCRA_UDD_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$STATE/launcher_${RUN_ID}_${LAUNCH_TAG}.log"

mkdir -p "$STATE"
LCER_DCRA_UDD_RUN_ID="$RUN_ID" nohup setsid /bin/bash "$ROOT/run_lcer_dcra_udd_l0_l3_after_mesa.sh" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}_${LAUNCH_TAG}.pid"
echo "LCER_DCRA_UDD_RUN_ID=$RUN_ID"
echo "LCER_DCRA_UDD_LAUNCHER_PID=$PID"
echo "LCER_DCRA_UDD_LAUNCHER_LOG=$LOG"
