#!/usr/bin/env bash
# Wait only for DRC's persisted terminal state, then hand off once to DOR.
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <absolute-project-root> <run-id> <absolute-upstream-state> <absolute-resume-d0-root> <absolute-pod-controls-root>" >&2
  exit 64
fi

PROJECT_ROOT="$1"
RUN_ID="$2"
UPSTREAM_STATE="$3"
RESUME_D0_ROOT="$4"
POD_CONTROLS_ROOT="$5"
PYTHON3_BIN="${PYTHON3_BIN:-python3}"
SCHEME_ROOT="$PROJECT_ROOT/runs/dor_${RUN_ID}"
STATE="$SCHEME_ROOT/state.json"

[[ "$PROJECT_ROOT" = /* && "$UPSTREAM_STATE" = /* && "$RESUME_D0_ROOT" = /* && "$POD_CONTROLS_ROOT" = /* ]] || { echo "all filesystem arguments must be absolute" >&2; exit 65; }
mkdir -p "$SCHEME_ROOT/train" "$SCHEME_ROOT/test"

write_wait_state() {
  "$PYTHON3_BIN" - "$STATE" "$PROJECT_ROOT" "$SCHEME_ROOT" "$RUN_ID" "$UPSTREAM_STATE" "$RESUME_D0_ROOT" "$POD_CONTROLS_ROOT" "$1" "$2" "$3" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "project_root": sys.argv[2], "scheme_root": sys.argv[3], "run_id": sys.argv[4],
    "upstream_state": sys.argv[5], "resume_d0_root": sys.argv[6], "pod_controls_root": sys.argv[7],
    "status": sys.argv[8], "phase": "waiting_upstream", "upstream_status": sys.argv[9],
    "upstream_failure_reason": sys.argv[10], "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temporary.replace(path)
PY
}

echo "[$(date -u +%FT%TZ)] waiting for DRC state: $UPSTREAM_STATE"
write_wait_state waiting "" ""
while true; do
  state_line="$($PYTHON3_BIN - "$UPSTREAM_STATE" <<'PY'
import json, sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    reason = str(payload.get("failure_reason", payload.get("reason", ""))).replace("\n", " ")
    print(f"{status}|{reason}")
except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
    print("invalid|")
PY
)"
  upstream_status="${state_line%%|*}"
  upstream_reason="${state_line#*|}"
  case "$upstream_status" in
    completed|failed|cancelled)
      echo "[$(date -u +%FT%TZ)] detected DRC terminal state=$upstream_status reason=$upstream_reason"
      write_wait_state waiting "$upstream_status" "$upstream_reason"
      exec bash "$PROJECT_ROOT/scripts/run_dor.sh" "$PROJECT_ROOT" "$RUN_ID" "$UPSTREAM_STATE" "$upstream_status" "$upstream_reason" "$RESUME_D0_ROOT" "$POD_CONTROLS_ROOT"
      ;;
    *)
      echo "[$(date -u +%FT%TZ)] DRC status=${upstream_status:-missing_or_invalid}; waiting 30 seconds"
      write_wait_state waiting "$upstream_status" "$upstream_reason"
      sleep 30
      ;;
  esac
done
