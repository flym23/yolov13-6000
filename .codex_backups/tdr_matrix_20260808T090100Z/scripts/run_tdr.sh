#!/usr/bin/env bash
# Execute the complete TDR factorial matrix after a recorded upstream terminal state.
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <absolute-project-root> <run-id> <absolute-upstream-state> <upstream-status> <upstream-reason> <absolute-resume-d0-root>" >&2
  exit 64
fi

PROJECT_ROOT="$1"
RUN_ID="$2"
UPSTREAM_STATE="$3"
UPSTREAM_STATUS="$4"
UPSTREAM_REASON="$5"
RESUME_D0_ROOT="$6"
PYTHON_BIN="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
CHAIN_ROOT="$PROJECT_ROOT/runs/chain/tdr_${RUN_ID}"

[[ "$PROJECT_ROOT" = /* && "$UPSTREAM_STATE" = /* && "$RESUME_D0_ROOT" = /* ]] || {
  echo "all filesystem arguments must be absolute paths" >&2
  exit 65
}
case "$UPSTREAM_STATUS" in completed|failed|cancelled) ;; *) echo "invalid upstream terminal status: $UPSTREAM_STATUS" >&2; exit 66 ;; esac
[[ -x "$PYTHON_BIN" && -f "$PROJECT_ROOT/tools/run_tdr_matrix.py" && -d "$RESUME_D0_ROOT" ]] || {
  echo "TDR runner prerequisites are unavailable" >&2
  exit 67
}
ACTUAL_UPSTREAM_STATUS="$($PYTHON_BIN - "$UPSTREAM_STATE" <<'PY'
import json
import sys
from pathlib import Path

try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status", ""))
except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
    print("")
PY
)"
[[ "$ACTUAL_UPSTREAM_STATUS" == "$UPSTREAM_STATUS" ]] || {
  echo "upstream state mismatch: expected $UPSTREAM_STATUS, got ${ACTUAL_UPSTREAM_STATUS:-missing_or_invalid}" >&2
  exit 68
}
mkdir -p "$CHAIN_ROOT"

exec "$PYTHON_BIN" "$PROJECT_ROOT/tools/run_tdr_matrix.py" \
  --project-root "$PROJECT_ROOT" \
  --chain-root "$CHAIN_ROOT" \
  --run-id "$RUN_ID" \
  --upstream-state "$UPSTREAM_STATE" \
  --upstream-status "$UPSTREAM_STATUS" \
  --upstream-reason "$UPSTREAM_REASON" \
  --resume-d0-root "$RESUME_D0_ROOT"
