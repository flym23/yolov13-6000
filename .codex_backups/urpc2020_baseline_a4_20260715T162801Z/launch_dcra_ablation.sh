#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rtx6000/ZZF/yolov13-6000
PY=/home/rtx6000/.conda/envs/yolov13/bin/python
SOURCE_ROOT=${DCRA_WAIT_SOURCE:-/home/rtx6000/ZZF/yolov13yuan-6000}
EXPECTED_ROOTS=${DCRA_EXPECTED_ROOTS:-3}
STATE=$ROOT/runs/dcra_ablation
RUN_ID=${DCRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=$STATE/launcher_${RUN_ID}.log
RESUME_FROM=${DCRA_RESUME_FROM:-}
SKIP_SMOKE=${DCRA_SKIP_SMOKE:-false}

mkdir -p "$STATE"
cd "$ROOT"
DCRA_RUN_ID="$RUN_ID" DCRA_RESUME_FROM="$RESUME_FROM" DCRA_SKIP_SMOKE="$SKIP_SMOKE" nohup setsid "$PY" "$ROOT/tools/wait_for_training_processes.py" \
  --source-root "$SOURCE_ROOT" \
  --target-root "$ROOT" \
  --run-script "$ROOT/run_dcra_ablation_parallel.sh" \
  --state "$STATE/wait_state.json" \
  --run-id "$RUN_ID" \
  --expected-roots "$EXPECTED_ROOTS" \
  --poll-seconds 30 \
  --settle-polls 2 \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE/nohup_launcher_${RUN_ID}.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
echo "DCRA_RUN_ID=$RUN_ID"
echo "DCRA_LAUNCHER_PID=$PID"
echo "DCRA_LAUNCHER_LOG=$LOG"
echo "DCRA_WAIT_SOURCE=$SOURCE_ROOT"
echo "DCRA_EXPECTED_ROOTS=$EXPECTED_ROOTS"
echo "DCRA_RESUME_FROM=${RESUME_FROM:-none}"
echo "DCRA_SKIP_SMOKE=$SKIP_SMOKE"
