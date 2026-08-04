#!/usr/bin/env bash
# DAD A1--A7 full-factorial chain: exactly three seed processes run concurrently within each stage.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE_DIR="$ROOT/runs/dad_yolov13_a1_a7"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=${DAD_YOLOV13_RUN_ID:-dad_yolov13_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
CURRENT_STAGE=initializing
COMPLETED=""
STAGES=(a1 a2 a3 a4 a5 a6 a7)

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "DAD chain already running for $RUN_ID" >&2; exit 73; }
export PIN_MEMORY=false
export WANDB_DISABLED=true

write_state() {
  DAD_STATE_FILE="$STATE_FILE" DAD_RUN_ID="$RUN_ID" DAD_STATUS="$1" DAD_STAGE="$CURRENT_STAGE" DAD_DETAIL="${2:-}" DAD_COMPLETED="$COMPLETED" DAD_PID="$$" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["DAD_STATE_FILE"])
payload = {
    "run_id": os.environ["DAD_RUN_ID"], "status": os.environ["DAD_STATUS"], "stage": os.environ["DAD_STAGE"],
    "detail": os.environ["DAD_DETAIL"], "launcher_pid": int(os.environ["DAD_PID"]),
    "completed_stages": os.environ["DAD_COMPLETED"].split(), "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
    "settings": {"epochs": 300, "patience": 40, "workers": 2, "amp": False, "plots": False, "deterministic": True, "parallel_seed_processes": 3},
    "a0_policy": "reuse designated LCER-L0 and SPC-P0 original-YOLOv13 summaries; do not retrain A0",
    "chain_policy": "A1-A7 execute sequentially; each stage starts exactly three seeds in parallel; any failed seed stops the chain",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temp.replace(path)
'
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
on_error() { code=$?; write_state failed "exit_code=$code" || true; exit "$code"; }
trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_group() {
  local label failed pid other
  label="$1"; shift; failed=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
      for other in "$@"; do
        if [[ "$other" != "$pid" ]] && kill -0 "$other" 2>/dev/null; then kill "$other" 2>/dev/null || true; fi
      done
    fi
  done
  (( failed == 0 )) || { echo "DAD $CURRENT_STAGE $label failed" >&2; return 75; }
}

run_stage() {
  local stage seed name weights
  local -a train_pids=() test_pids=()
  stage="$1"; CURRENT_STAGE="$stage"; write_state training
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_dad_yolov13_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    train_pids+=("$!")
  done
  printf '%s\n' "${train_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.train.pids"
  wait_group training "${train_pids[@]}"
  write_state testing
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    weights="$TRAIN_DIR/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; return 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 >"$TEST_DIR/$name.log" 2>&1 &
    test_pids+=("$!")
  done
  printf '%s\n' "${test_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.test.pids"
  wait_group testing "${test_pids[@]}"
  COMPLETED="$COMPLETED $stage"
  "$PY" "$ROOT/tools/collect_dad_yolov13_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $COMPLETED >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
[[ -f "$ROOT/yolov13n.pt" ]] || { echo "Pretrained weights unavailable: $ROOT/yolov13n.pt" >&2; exit 80; }
echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=preflight
write_state running "launcher_pid=$$"
"$PY" -m py_compile "$ROOT/ultralytics/nn/modules/block.py" "$ROOT/ultralytics/nn/modules/head.py" "$ROOT/ultralytics/nn/modules/__init__.py" "$ROOT/ultralytics/nn/tasks.py" "$ROOT/test_dad_yolov13.py" "$ROOT/tools/train_dad_yolov13_worker.py" "$ROOT/tools/collect_dad_yolov13_ablation.py" >"$STATE_DIR/$RUN_ID.py_compile.log" 2>&1
"$PY" "$ROOT/test_dad_yolov13.py" --yaml --imgsz 640 >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1

for stage in "${STAGES[@]}"; do
  run_stage "$stage"
done
CURRENT_STAGE=complete
write_state complete "a1_a7_finished=true; all_stages_started_without_score_gates=true"
echo "DAD A1--A7 chain completed: run_id=$RUN_ID"
