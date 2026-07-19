#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/rom305/zzf/yolov13-305"
PY="/home/rom305/miniconda3/envs/yolov13/bin/python"
STATE_DIR="$ROOT/runs/ufcr_ablation"
RUN_ID="${UFCR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG="$STATE_DIR/launcher_${RUN_ID}.log"

cd "$ROOT"
mkdir -p "$STATE_DIR"

if [[ -s "$STATE_DIR/state.json" ]]; then
  read -r status pid < <("$PY" - "$STATE_DIR/state.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(state.get("status", ""), state.get("launcher_pid", 0))
PY
)
  if [[ "$status" == "running" || "$status" == "stage_complete" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "UFCR chain already active: pid=$pid"
    exit 0
  fi
fi

if [[ -d "$STATE_DIR/chain.lock" ]]; then
  rmdir "$STATE_DIR/chain.lock" 2>/dev/null || {
    echo "Non-empty or active chain lock: $STATE_DIR/chain.lock" >&2
    exit 73
  }
fi

UFCR_RUN_ID="$RUN_ID" nohup setsid "$ROOT/run_ufcr_ablation_parallel.sh" > "$LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$STATE_DIR/launch.pid"
sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
  echo "UFCR launcher exited during startup. See $LOG" >&2
  tail -n 80 "$LOG" >&2 || true
  exit 1
fi
printf '{"run_id":"%s","launch_pid":%s,"log":"%s"}\n' "$RUN_ID" "$pid" "$LOG" > "$STATE_DIR/launched.json"
echo "UFCR_RUN_ID=$RUN_ID"
echo "UFCR_LAUNCH_PID=$pid"
echo "UFCR_LOG=$LOG"
