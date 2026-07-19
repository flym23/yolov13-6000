#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rom305/zzf/yolov13-305
PY=/home/rom305/miniconda3/envs/yolov13/bin/python
TRAIN=$ROOT/runs/train
TEST=$ROOT/runs/test
STATE=$ROOT/runs/ag_dsb_hyperace_ablation
RUN_ID=${AGDSB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOCK=$STATE/chain.lock
CURRENT=initialization
STAGES=(a2_dense a3_topk2 a4_topk3 a5_topk2_no_norm a6_topk2_both)

mkdir -p "$TRAIN" "$TEST" "$STATE"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir "$LOCK" 2>/dev/null || { echo "AG-DSB-HyperACE chain already running" >&2; exit 73; }

state() {
  AG_STATUS=$1 AG_STAGE=$2 AG_CODE=${3:-0} AG_RUN_ID=$RUN_ID AG_PID=$$ AG_STATE=$STATE "$PY" -c 'import json,os; from pathlib import Path; from datetime import datetime,timezone; p=Path(os.environ["AG_STATE"])/"state.json"; d={"run_id":os.environ["AG_RUN_ID"],"status":os.environ["AG_STATUS"],"stage":os.environ["AG_STAGE"],"exit_code":int(os.environ["AG_CODE"]),"launcher_pid":int(os.environ["AG_PID"]),"updated_at":datetime.now(timezone.utc).isoformat(),"stage_order":["a2_dense","a3_topk2","a4_topk3","a5_topk2_no_norm","a6_topk2_both"]}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
}

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  rmdir "$LOCK" 2>/dev/null || true
  ((rc == 0)) || state failed "$CURRENT" "$rc" || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo $$ > "$STATE/launcher.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
state running "$CURRENT"

audit_gate() {
  local checkpoint=$1
  "$PY" "$ROOT/tools/audit_ag_dsb_gate.py" "$checkpoint" \
    --require-headwise-moved \
    --min-mean-move 1e-6 \
    --min-max-move 1e-5 \
    --per-head-threshold 1e-6 \
    --min-moved-heads 2 \
    --check-forward
}

preflight() {
  local name=agdsb_${RUN_ID}_preflight_a2_dense_seed0
  CURRENT=preflight.a2_dense
  state running "$CURRENT"
  echo $$ > "$STATE/a2_dense.preflight.pid"
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_ag_dsb_hyperace_worker.py" \
    --root "$ROOT" --stage a2_dense --name "$name" --seed 0 --epochs 5 > "$TRAIN/${name}.log" 2>&1
  test -s "$TRAIN/$name/weights/last.pt"
  "$PY" "$ROOT/tools/audit_ag_dsb_gate.py" "$TRAIN/$name/weights/last.pt" \
    --require-headwise-moved \
    --min-mean-move 1e-7 \
    --min-max-move 1e-6 \
    --per-head-threshold 1e-7 \
    --min-moved-heads 2 \
    --check-forward >> "$TRAIN/${name}.log" 2>&1
  "$PY" -c 'import sys; from pathlib import Path; t=Path(sys.argv[1]).read_text(encoding="utf-8",errors="replace").lower(); assert "traceback" not in t and "runtimeerror" not in t and "nan" not in t; assert "ag-dsb eta optimizer registration: pass" in t; assert "ag-dsb head-aligned gate audit passed" in t; assert "checkpoint fp32 forward passed" in t' "$TRAIN/${name}.log"
}

preflight
state preflight_complete a2_dense

start_train() {
  local stage=$1 gpu=$2 seed=$3 name=agdsb_${RUN_ID}_${1}_seed${3}
  CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/tools/train_ag_dsb_hyperace_worker.py" \
    --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" > "$TRAIN/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"
}

wait_train() {
  local stage=$1 bad=0
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && test -s "$TRAIN/${NAMES[$i]}/weights/best.pt" || bad=1
  done
  ((bad == 0)) || return 1
  for name in "${NAMES[@]}"; do
    audit_gate "$TRAIN/$name/weights/best.pt" >> "$TRAIN/${name}.log" 2>&1 || bad=1
  done
  return "$bad"
}

start_test() {
  local stage=$1 gpu=$2 seed=$3 name=agdsb_${RUN_ID}_${1}_seed${3}
  CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/test.py" \
    --weights "$TRAIN/$name/weights/best.pt" --name "$name" --device 0 --batch 16 --imgsz 640 > "$TEST/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  echo "$!" > "$STATE/${stage}_seed${seed}.test.pid"
}

wait_test() {
  local bad=0
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && test -s "$TEST/${NAMES[$i]}/summary_metrics.json" || bad=1
  done
  return "$bad"
}

DONE=()
for stage in "${STAGES[@]}"; do
  CURRENT=${stage}.train
  state running "$CURRENT"
  PIDS=(); NAMES=()
  start_train "$stage" 0 0
  start_train "$stage" 1 1
  start_train "$stage" 2 2
  wait_train "$stage"
  CURRENT=${stage}.test
  state running "$CURRENT"
  PIDS=(); NAMES=()
  start_test "$stage" 0 0
  start_test "$stage" 1 1
  start_test "$stage" 2 2
  wait_test
  "$PY" "$ROOT/tools/collect_ag_dsb_hyperace_ablation.py" \
    --root "$ROOT" --run-id "$RUN_ID" --stages "$stage"
  DONE+=("$stage")
  state stage_complete "$stage"
done

CURRENT=reporting
state running "$CURRENT"
"$PY" "$ROOT/tools/collect_ag_dsb_hyperace_ablation.py" \
  --root "$ROOT" --run-id "$RUN_ID" --stages "${DONE[@]}"
CURRENT=complete
state complete "$CURRENT"
