#!/usr/bin/env bash
# Wait for the active GMR chain to finish successfully, then start TDR T1--T7 once.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
GMR_STATE=${GMR_STATE:-/home/room305/ZZF/yolov13yuan-6000/runs/gmr_ablation/gmr_20260727_233907.state.json}
PY=/home/room305/.conda/envs/yolov13/bin/python
STATE_DIR="$ROOT/runs/tdr_yolov13_t1_t7"
LOCK="$STATE_DIR/.wait_for_gmr.lock"

mkdir -p "$STATE_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "TDR GMR waiter is already running" >&2; exit 73; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
[[ -x "$PY" && -f "$GMR_STATE" ]] || { echo "Missing GMR state or Python runtime" >&2; exit 78; }

while true; do
  status=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))' "$GMR_STATE")
  [[ "$status" == "complete" ]] && break
  [[ "$status" == "failed" ]] && { echo "GMR ended with failed state" >&2; exit 82; }
  printf '%s %s\n' "$(date -u +%FT%TZ)" "waiting_gmr_status=$status" >"$STATE_DIR/wait_for_gmr.log"
  sleep 60
done
exec /usr/bin/env bash "$ROOT/run_tdr_yolov13_t1_t7.sh"
