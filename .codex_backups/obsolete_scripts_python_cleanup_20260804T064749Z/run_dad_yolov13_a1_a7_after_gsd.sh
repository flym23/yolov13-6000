#!/usr/bin/env bash
# Start DAD only after the active GSD chain exits successfully; no score gate is applied.
set -Eeuo pipefail

DAD_ROOT=$(cd "$(dirname "$0")" && pwd)
GSD_ROOT=/home/room305/ZZF/yolov13yuan-6000
PY=/home/room305/.conda/envs/yolov13/bin/python
STATE_DIR="$DAD_ROOT/runs/dad_yolov13_a1_a7"
LOCK="$STATE_DIR/.wait_for_gsd.lock"
PID=$$

mkdir -p "$STATE_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "DAD GSD waiter is already running" >&2; exit 73; }

write_wait_state() {
  DAD_WAIT_DIR="$STATE_DIR" DAD_WAIT_STATUS="$1" DAD_WAIT_PID="$PID" DAD_GSD_ROOT="$GSD_ROOT" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["DAD_WAIT_DIR"]) / "wait_for_gsd.json"
payload = {
    "status": os.environ["DAD_WAIT_STATUS"], "launcher_pid": int(os.environ["DAD_WAIT_PID"]),
    "gsd_root": os.environ["DAD_GSD_ROOT"],
    "policy": "start DAD A1-A7 after GSD launcher and workers exit with state=complete; no metric threshold",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temp.replace(path)
'
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT
trap 'write_wait_state interrupted; exit 130' INT
trap 'write_wait_state interrupted; exit 143' TERM
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }

while pgrep -f "[r]un_gsd_ablation\.sh|[t]rain_gsd_worker\.py|[t]est\.py.*gsd_" >/dev/null; do
  write_wait_state waiting_gsd_running
  sleep 60
done

state_file=$(ls -t "$GSD_ROOT"/runs/gsd_ablation/gsd_*.state.json 2>/dev/null | head -n 1 || true)
[[ -f "$state_file" ]] || { write_wait_state missing_gsd_state; echo "No GSD state file found" >&2; exit 81; }
status=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))' "$state_file")
[[ "$status" == "complete" ]] || { write_wait_state "gsd_not_complete_$status"; echo "GSD did not complete successfully: $state_file ($status)" >&2; exit 82; }
write_wait_state gsd_complete_starting_dad
rmdir "$LOCK" 2>/dev/null || true
exec /usr/bin/env bash "$DAD_ROOT/run_dad_yolov13_a1_a7.sh"
