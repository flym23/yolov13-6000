#!/usr/bin/env bash
# Execute the immutable DOR 2^3 matrix after the recorded DRC terminal state.
set -Eeuo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <absolute-project-root> <run-id> <absolute-upstream-state> <upstream-status> <upstream-reason> <absolute-resume-d0-root> <absolute-pod-controls-root>" >&2
  exit 64
fi

PROJECT_ROOT="$1"
RUN_ID="$2"
UPSTREAM_STATE="$3"
UPSTREAM_STATUS="$4"
UPSTREAM_REASON="$5"
RESUME_D0_ROOT="$6"
POD_CONTROLS_ROOT="$7"
PYTHON_BIN="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
SCHEME_ROOT="$PROJECT_ROOT/runs/dor_${RUN_ID}"
STATE="$SCHEME_ROOT/state.json"

[[ "$PROJECT_ROOT" = /* && "$UPSTREAM_STATE" = /* && "$RESUME_D0_ROOT" = /* && "$POD_CONTROLS_ROOT" = /* ]] || { echo "all filesystem arguments must be absolute" >&2; exit 65; }
case "$UPSTREAM_STATUS" in completed|failed|cancelled) ;; *) echo "invalid upstream status" >&2; exit 66;; esac
[[ -x "$PYTHON_BIN" && -f "$PROJECT_ROOT/tools/run_dor_matrix.py" && -d "$RESUME_D0_ROOT" && -d "$POD_CONTROLS_ROOT" ]] || { echo "DOR prerequisites unavailable" >&2; exit 67; }

ACTUAL_STATUS="$($PYTHON_BIN - "$UPSTREAM_STATE" <<'PY'
import json, sys
from pathlib import Path
try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status", ""))
except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
    print("")
PY
)"
[[ "$ACTUAL_STATUS" == "$UPSTREAM_STATUS" ]] || { echo "upstream state mismatch: ${ACTUAL_STATUS:-invalid}" >&2; exit 68; }
if [[ -f "$STATE" ]]; then
  CURRENT_STATUS="$($PYTHON_BIN - "$STATE" <<'PY'
import json, sys
from pathlib import Path
try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status", ""))
except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
    print("")
PY
)"
  [[ "$CURRENT_STATUS" == "waiting" ]] || { echo "DOR run already has state=${CURRENT_STATUS:-invalid}" >&2; exit 69; }
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/tools/run_dor_matrix.py" \
  --project-root "$PROJECT_ROOT" --scheme-root "$SCHEME_ROOT" --run-id "$RUN_ID" \
  --upstream-state "$UPSTREAM_STATE" --upstream-status "$UPSTREAM_STATUS" \
  --upstream-reason "$UPSTREAM_REASON" --resume-d0-root "$RESUME_D0_ROOT" \
  --pod-controls-root "$POD_CONTROLS_ROOT"
