#!/usr/bin/env bash
# TDR T1--T7: one smoke trial, then exactly three seeds in parallel for every formal stage.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE_DIR="$ROOT/runs/tdr_yolov13_t1_t7"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=${TDR_YOLOV13_RUN_ID:-tdr_yolov13_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
STAGES=(t1 t2 t3 t4 t5 t6 t7)
CURRENT_STAGE=initializing
COMPLETED=""

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "TDR chain already running for $RUN_ID" >&2; exit 73; }
export WANDB_DISABLED=true PIN_MEMORY=false

write_state() {
  TDR_STATE_FILE="$STATE_FILE" TDR_RUN_ID="$RUN_ID" TDR_STATUS="$1" TDR_STAGE="$CURRENT_STAGE" \
    TDR_DETAIL="${2:-}" TDR_COMPLETED="$COMPLETED" TDR_PID="$$" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["TDR_STATE_FILE"])
payload = {
    "run_id": os.environ["TDR_RUN_ID"], "status": os.environ["TDR_STATUS"], "stage": os.environ["TDR_STAGE"],
    "detail": os.environ["TDR_DETAIL"], "launcher_pid": int(os.environ["TDR_PID"]),
    "completed_stages": os.environ["TDR_COMPLETED"].split(),
    "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
    "settings": {"epochs": 300, "patience": 40, "workers": 2, "amp": False, "deterministic": True,
                 "parallel_seed_processes": 3},
    "t0_policy": "reuse the authorized LCER-L0 and SPC-P0 baseline summaries",
    "chain_policy": "T1-T7 run sequentially without result thresholds; every formal stage uses three parallel seeds",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
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
  (( failed == 0 )) || { echo "TDR $CURRENT_STAGE $label failed" >&2; return 75; }
}

run_smoke_trial() {
  local name weights
  CURRENT_STAGE=smoke_t7
  name="${RUN_ID}_smoke_t7_seed0"
  write_state smoke_training "one_epoch_t7"
  "$PY" "$ROOT/tools/train_tdr_yolov13_worker.py" --root "$ROOT" --stage t7 --data "$DATA" --name "$name" \
    --seed 0 --epochs 1 --patience 1 >"$TRAIN_DIR/$name.log" 2>&1
  weights="$TRAIN_DIR/$name/weights/best.pt"
  [[ -f "$weights" ]] || { echo "TDR smoke trial did not create $weights" >&2; return 77; }
  write_state smoke_testing "one_epoch_t7"
  "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 \
    >"$TEST_DIR/$name.log" 2>&1
  [[ -f "$TEST_DIR/$name/summary_metrics.json" ]] || { echo "TDR smoke validation summary missing" >&2; return 77; }
}

run_stage() {
  local stage seed name weights
  local -a train_pids=() test_pids=()
  stage="$1"; CURRENT_STAGE="$stage"; write_state training
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_tdr_yolov13_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" \
      --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    train_pids+=("$!")
  done
  printf '%s\n' "${train_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.train.pids"
  wait_group training "${train_pids[@]}"
  write_state testing
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    weights="$TRAIN_DIR/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; return 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 \
      >"$TEST_DIR/$name.log" 2>&1 &
    test_pids+=("$!")
  done
  printf '%s\n' "${test_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.test.pids"
  wait_group testing "${test_pids[@]}"
  COMPLETED="$COMPLETED $stage"
  "$PY" "$ROOT/tools/collect_tdr_yolov13_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $COMPLETED \
    >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
[[ -f "$ROOT/yolov13n.pt" ]] || { echo "Pretrained weights unavailable: $ROOT/yolov13n.pt" >&2; exit 80; }
echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=preflight
write_state running "launcher_pid=$$"
"$PY" -m py_compile "$ROOT/ultralytics/nn/modules/block.py" "$ROOT/ultralytics/nn/modules/head.py" \
  "$ROOT/ultralytics/nn/modules/__init__.py" "$ROOT/ultralytics/nn/tasks.py" "$ROOT/test_tdr_yolov13_reviewed.py" \
  "$ROOT/tools/train_tdr_yolov13_worker.py" "$ROOT/tools/collect_tdr_yolov13_ablation.py" >"$STATE_DIR/$RUN_ID.py_compile.log" 2>&1
"$PY" "$ROOT/test_tdr_yolov13_reviewed.py" --yaml --imgsz 128 >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1
run_smoke_trial
for stage in "${STAGES[@]}"; do run_stage "$stage"; done
CURRENT_STAGE=complete
write_state complete "t1_t7_finished=true; smoke_t7_passed=true; no_result_gates=true"
echo "TDR T1--T7 chain completed: run_id=$RUN_ID"
