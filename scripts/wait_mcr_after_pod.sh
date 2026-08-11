#!/usr/bin/env bash
# Wait for POD's persisted terminal state, then hand off exactly once to the MCR matrix runner.
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <absolute-project-root> <run-id> <absolute-upstream-state> <absolute-resume-d0-root>" >&2
  exit 64
fi

PROJECT_ROOT="$1"
RUN_ID="$2"
UPSTREAM_STATE="$3"
RESUME_D0_ROOT="$4"
PYTHON3_BIN="${PYTHON3_BIN:-python3}"
SCHEME_ROOT="$PROJECT_ROOT/runs/mcr_${RUN_ID}"
STATE="$SCHEME_ROOT/state.json"

[[ "$PROJECT_ROOT" = /* && "$UPSTREAM_STATE" = /* && "$RESUME_D0_ROOT" = /* ]] || { echo "all filesystem arguments must be absolute" >&2; exit 65; }
mkdir -p "$SCHEME_ROOT/train"

write_wait_state() {
  "$PYTHON3_BIN" - "$STATE" "$RUN_ID" "$UPSTREAM_STATE" "$RESUME_D0_ROOT" "$1" "$2" "$3" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2], "status": sys.argv[5], "phase": "waiting_upstream",
    "upstream_state": sys.argv[3], "upstream_status": sys.argv[6], "upstream_failure_reason": sys.argv[7],
    "resume_d0_root": sys.argv[4], "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temporary.replace(path)
PY
}

echo "[$(date -u +%FT%TZ)] waiting for POD state: $UPSTREAM_STATE"
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
      echo "[$(date -u +%FT%TZ)] detected POD terminal state=$upstream_status reason=$upstream_reason"
      write_wait_state waiting "$upstream_status" "$upstream_reason"
      exec bash "$PROJECT_ROOT/scripts/run_mcr.sh" "$PROJECT_ROOT" "$RUN_ID" "$UPSTREAM_STATE" "$upstream_status" "$upstream_reason" "$RESUME_D0_ROOT"
      ;;
    *)
      echo "[$(date -u +%FT%TZ)] POD status=${upstream_status:-missing_or_invalid}; waiting 30 seconds"
      write_wait_state waiting "$upstream_status" "$upstream_reason"
      sleep 30
      ;;
  esac
done
