#!/usr/bin/env bash
# Start once; wait for CARM completion, then launch the L3-CRU T0--T7 chain.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
CARM_STATE_FILE=${CARM_STATE_FILE:-$ROOT/../yolov13yuan-6000/runs/carm_ablation/carm_20260730_043000.state.json}
WAIT_LOG="$ROOT/runs/l3cru/after_carm_wait.log"

mkdir -p "$(dirname "$WAIT_LOG")"
[[ -x "$PY" && -f "$CARM_STATE_FILE" ]] || { echo "CARM dependency or Python runtime is unavailable" >&2; exit 78; }
while true; do
  status=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$CARM_STATE_FILE")
  case "$status" in
    complete)
      echo "CARM complete; starting L3-CRU." | tee -a "$WAIT_LOG"
      exec bash "$ROOT/run_l3cru_t0_t7.sh" "$CARM_STATE_FILE"
      ;;
    failed)
      echo "CARM failed; L3-CRU will not start." | tee -a "$WAIT_LOG" >&2
      exit 75
      ;;
    *)
      echo "$(date -u +%FT%TZ) waiting for CARM: $status" >>"$WAIT_LOG"
      sleep 120
      ;;
  esac
done
