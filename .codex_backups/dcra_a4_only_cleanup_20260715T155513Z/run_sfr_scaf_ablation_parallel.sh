#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/rom305/zzf/yolov13-305"
PY="/home/rom305/miniconda3/envs/yolov13/bin/python"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
STATE_DIR="$ROOT/runs/sfr_scaf_ablation"
RUN_ID="${SFR_SCAF_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOCK_DIR="$STATE_DIR/chain.lock"
CURRENT_STAGE="initialization"
STAGE_ORDER=(f3_sfr_scaf f4_no_semantic_filter f5_fixed_route f6_consistency_only f7_semantic_only)

cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir -p "$TRAIN_DIR" "$TEST_DIR" "$STATE_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "SFR-SCAF chain lock already exists: $LOCK_DIR" >&2
  exit 73
fi

write_state() {
  SFR_SCAF_STATUS="$1" SFR_SCAF_STAGE="$2" SFR_SCAF_CODE="${3:-0}" SFR_SCAF_RUN_ID="$RUN_ID" SFR_SCAF_LAUNCHER_PID="$$" SFR_SCAF_STATE_DIR="$STATE_DIR" "$PY" -c 'import json,os; from datetime import datetime,timezone; from pathlib import Path; p=Path(os.environ["SFR_SCAF_STATE_DIR"])/"state.json"; d={"run_id":os.environ["SFR_SCAF_RUN_ID"],"status":os.environ["SFR_SCAF_STATUS"],"stage":os.environ["SFR_SCAF_STAGE"],"exit_code":int(os.environ["SFR_SCAF_CODE"]),"launcher_pid":int(os.environ["SFR_SCAF_LAUNCHER_PID"]),"updated_at":datetime.now(timezone.utc).isoformat(),"stage_order":["f3_sfr_scaf","f4_no_semantic_filter","f5_fixed_route","f6_consistency_only","f7_semantic_only"]}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  rmdir "$LOCK_DIR" 2>/dev/null || true
  (( rc == 0 )) || write_state "failed" "$CURRENT_STAGE" "$rc" || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "$$" > "$STATE_DIR/launcher.pid"
echo "$RUN_ID" > "$STATE_DIR/run_id.txt"
write_state "running" "$CURRENT_STAGE"

run_smoke() {
  local stage="$1" name="sfr_scaf_${RUN_ID}_smoke_${1}"
  CURRENT_STAGE="smoke.${stage}"
  write_state "running" "$CURRENT_STAGE"
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_sfr_scaf_worker.py" --root "$ROOT" --stage "$stage" --name "$name" --seed 0 --epochs 1 > "$TRAIN_DIR/${name}.log" 2>&1
  test -s "$TRAIN_DIR/$name/weights/last.pt"
}

run_train() {
  local stage="$1" gpu="$2" seed="$3" name="sfr_scaf_${RUN_ID}_${1}_seed${3}"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_DISABLED=true "$PY" "$ROOT/tools/train_sfr_scaf_worker.py" --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" > "$TRAIN_DIR/${name}.log" 2>&1 &
  TRAIN_PIDS+=("$!")
  TRAIN_NAMES+=("$name")
  echo "$!" > "$STATE_DIR/${stage}_seed${seed}.train.pid"
  echo "$!" > "$TRAIN_DIR/${name}.pid"
}

wait_train_group() {
  local index failed=0
  for index in "${!TRAIN_PIDS[@]}"; do
    if ! wait "${TRAIN_PIDS[$index]}" || [[ ! -s "$TRAIN_DIR/${TRAIN_NAMES[$index]}/weights/best.pt" ]]; then failed=1; fi
  done
  return "$failed"
}

run_test() {
  local stage="$1" gpu="$2" seed="$3" name="sfr_scaf_${RUN_ID}_${1}_seed${3}" log="$TEST_DIR/sfr_scaf_${RUN_ID}_${1}_seed${3}.log"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_DISABLED=true "$PY" "$ROOT/test.py" --weights "$TRAIN_DIR/$name/weights/best.pt" --name "$name" --device 0 --batch 16 --imgsz 640 > "$log" 2>&1 &
  TEST_PIDS+=("$!")
  TEST_NAMES+=("$name")
  echo "$!" > "$STATE_DIR/${stage}_seed${seed}.test.pid"
}

wait_test_group() {
  local index failed=0
  for index in "${!TEST_PIDS[@]}"; do
    if ! wait "${TEST_PIDS[$index]}" || [[ ! -s "$TEST_DIR/${TEST_NAMES[$index]}/summary_metrics.json" ]]; then failed=1; fi
  done
  return "$failed"
}

for stage in "${STAGE_ORDER[@]}"; do run_smoke "$stage"; done
COMPLETED=()
for stage in "${STAGE_ORDER[@]}"; do
  CURRENT_STAGE="${stage}.train"; write_state "running" "$CURRENT_STAGE"
  TRAIN_PIDS=(); TRAIN_NAMES=()
  run_train "$stage" 0 0; run_train "$stage" 1 1; run_train "$stage" 2 2
  wait_train_group
  CURRENT_STAGE="${stage}.test"; write_state "running" "$CURRENT_STAGE"
  TEST_PIDS=(); TEST_NAMES=()
  run_test "$stage" 0 0; run_test "$stage" 1 1; run_test "$stage" 2 2
  wait_test_group
  "$PY" "$ROOT/tools/collect_sfr_scaf_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "$stage"
  COMPLETED+=("$stage"); write_state "stage_complete" "$stage"
done
CURRENT_STAGE="reporting"; write_state "running" "$CURRENT_STAGE"
"$PY" "$ROOT/tools/collect_sfr_scaf_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}"
CURRENT_STAGE="complete"; write_state "complete" "$CURRENT_STAGE"
echo "SFR-SCAF ablation chain completed: run_id=$RUN_ID"
