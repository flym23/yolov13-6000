#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/room305/ZZF/yolov13-6000
STATE=$ROOT/runs/urpc2020_medcra_m0_m7
RUN_ID=${MEDCRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
START_STAGE=${MEDCRA_START_STAGE:-m0_original}
RESUME_STAGE=${MEDCRA_RESUME_STAGE:-}
LAUNCH_TAG=${MEDCRA_LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG=$STATE/launcher_${RUN_ID}_${START_STAGE}_${LAUNCH_TAG}.log

mkdir -p "$STATE"
cd "$ROOT"
MEDCRA_RUN_ID="$RUN_ID" MEDCRA_START_STAGE="$START_STAGE" MEDCRA_RESUME_STAGE="$RESUME_STAGE" nohup setsid /bin/bash "$ROOT/run_medcra_m0_m7_after_durc.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}_${START_STAGE}_${LAUNCH_TAG}.pid"
echo "MEDCRA_RUN_ID=$RUN_ID"
echo "MEDCRA_START_STAGE=$START_STAGE"
echo "MEDCRA_RESUME_STAGE=${RESUME_STAGE:-none}"
echo "MEDCRA_LAUNCH_TAG=$LAUNCH_TAG"
echo "MEDCRA_LAUNCHER_PID=$PID"
echo "MEDCRA_LAUNCHER_LOG=$LOG"
