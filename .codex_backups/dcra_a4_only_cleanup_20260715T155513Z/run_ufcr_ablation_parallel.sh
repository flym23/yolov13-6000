#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/rom305/zzf/yolov13-305"
PY="/home/rom305/miniconda3/envs/yolov13/bin/python"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
STATE_DIR="$ROOT/runs/ufcr_ablation"
RUN_ID="${UFCR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOCK_DIR="$STATE_DIR/chain.lock"
CURRENT_STAGE="initialization"

cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir -p "$TRAIN_DIR" "$TEST_DIR" "$STATE_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "UFCR chain lock already exists: $LOCK_DIR" >&2
  exit 73
fi

write_state() {
  local status="$1"
  local stage="$2"
  local code="${3:-0}"
  UFCR_STATUS="$status" UFCR_STAGE="$stage" UFCR_CODE="$code" UFCR_RUN_ID="$RUN_ID" \
    UFCR_LAUNCHER_PID="$$" UFCR_STATE_DIR="$STATE_DIR" "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["UFCR_STATE_DIR"]) / "state.json"
payload = {
    "run_id": os.environ["UFCR_RUN_ID"],
    "status": os.environ["UFCR_STATUS"],
    "stage": os.environ["UFCR_STAGE"],
    "exit_code": int(os.environ["UFCR_CODE"]),
    "launcher_pid": int(os.environ["UFCR_LAUNCHER_PID"]),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if (( rc != 0 )); then
    write_state "failed" "$CURRENT_STAGE" "$rc" || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "$$" > "$STATE_DIR/launcher.pid"
echo "$RUN_ID" > "$STATE_DIR/run_id.txt"
write_state "running" "$CURRENT_STAGE"

run_train() {
  local stage="$1"
  local gpu="$2"
  local seed="$3"
  local name="ufcr_${RUN_ID}_${stage}_seed${seed}"
  local log="$TRAIN_DIR/${name}.log"

  CUDA_VISIBLE_DEVICES="$gpu" WANDB_DISABLED=true "$PY" "$ROOT/tools/train_ufcr_worker.py" \
    --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$STATE_DIR/${stage}_seed${seed}.train.pid"
  echo "$pid" > "$TRAIN_DIR/${name}.pid"
  echo "START train stage=$stage seed=$seed gpu=$gpu pid=$pid name=$name"
  TRAIN_PIDS+=("$pid")
  TRAIN_NAMES+=("$name")
}

wait_train_group() {
  local failed=0
  local i pid name
  for i in "${!TRAIN_PIDS[@]}"; do
    pid="${TRAIN_PIDS[$i]}"
    name="${TRAIN_NAMES[$i]}"
    if wait "$pid"; then
      if [[ ! -s "$TRAIN_DIR/$name/weights/best.pt" ]]; then
        echo "Missing best.pt after successful worker: $name" >&2
        failed=1
      fi
    else
      echo "Training worker failed: $name (pid=$pid)" >&2
      failed=1
    fi
  done
  return "$failed"
}

run_test() {
  local stage="$1"
  local gpu="$2"
  local seed="$3"
  local name="ufcr_${RUN_ID}_${stage}_seed${seed}"
  local weight="$TRAIN_DIR/$name/weights/best.pt"
  local log="$TEST_DIR/${name}.log"

  CUDA_VISIBLE_DEVICES="$gpu" WANDB_DISABLED=true "$PY" "$ROOT/test.py" \
    --weights "$weight" --name "$name" --device 0 --batch 16 --imgsz 640 > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$STATE_DIR/${stage}_seed${seed}.test.pid"
  echo "START test stage=$stage seed=$seed gpu=$gpu pid=$pid name=$name"
  TEST_PIDS+=("$pid")
  TEST_NAMES+=("$name")
}

wait_test_group() {
  local failed=0
  local i pid name
  for i in "${!TEST_PIDS[@]}"; do
    pid="${TEST_PIDS[$i]}"
    name="${TEST_NAMES[$i]}"
    if wait "$pid"; then
      if [[ ! -s "$TEST_DIR/$name/summary_metrics.json" ]]; then
        echo "Missing summary_metrics.json after validation: $name" >&2
        failed=1
      fi
    else
      echo "Validation worker failed: $name (pid=$pid)" >&2
      failed=1
    fi
  done
  return "$failed"
}

run_group() {
  local stage="$1"
  CURRENT_STAGE="$stage.train"
  write_state "running" "$CURRENT_STAGE"
  TRAIN_PIDS=()
  TRAIN_NAMES=()
  run_train "$stage" 0 0
  run_train "$stage" 1 1
  run_train "$stage" 2 2
  wait_train_group

  CURRENT_STAGE="$stage.test"
  write_state "running" "$CURRENT_STAGE"
  TEST_PIDS=()
  TEST_NAMES=()
  run_test "$stage" 0 0
  run_test "$stage" 1 1
  run_test "$stage" 2 2
  wait_test_group

  "$PY" "$ROOT/tools/collect_ufcr_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "$stage"
  write_state "stage_complete" "$stage"
}

STAGES=()
for stage in a1 a2 a3; do
  run_group "$stage"
  STAGES+=("$stage")
done

CURRENT_STAGE="a4.gate"
write_state "running" "$CURRENT_STAGE"
if "$PY" "$ROOT/tools/evaluate_ufcr_a4_gate.py" --root "$ROOT" --run-id "$RUN_ID"; then
  run_group "a4"
  STAGES+=("a4")
else
  rc=$?
  if (( rc != 3 )); then
    exit "$rc"
  fi
  echo "A4 skipped because A3 did not pass the Original-YOLOv13 gate or the A0 baseline summary was unavailable."
fi

CURRENT_STAGE="reporting"
write_state "running" "$CURRENT_STAGE"
"$PY" "$ROOT/tools/collect_ufcr_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${STAGES[@]}"

CURRENT_STAGE="complete"
write_state "complete" "$CURRENT_STAGE"
echo "UFCR ablation chain completed: run_id=$RUN_ID stages=${STAGES[*]}"
