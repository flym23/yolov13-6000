#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rom305/zzf/yolov13-305
PY=/home/rom305/miniconda3/envs/yolov13/bin/python
TRAIN=$ROOT/runs/train
TEST=$ROOT/runs/test
STATE=$ROOT/runs/rat_ablation
RUN_ID=${RAT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOCK=$STATE/chain.lock
CURRENT=initialization
STAGES=(t1_rat_initial t4_rat_no_amplitude t5_rat_channel_only t6_rat_amplitude_only t2_rat_late t3_rat_all t7_faar_rat_initial)

mkdir -p "$TRAIN" "$TEST" "$STATE"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir "$LOCK" 2>/dev/null || { echo "RAT chain already running" >&2; exit 73; }
state() {
  RAT_STATUS=$1 RAT_STAGE=$2 RAT_CODE=${3:-0} RAT_RUN_ID=$RUN_ID RAT_PID=$$ RAT_STATE=$STATE "$PY" -c 'import json,os; from pathlib import Path; from datetime import datetime,timezone; p=Path(os.environ["RAT_STATE"])/"state.json"; d={"run_id":os.environ["RAT_RUN_ID"],"status":os.environ["RAT_STATUS"],"stage":os.environ["RAT_STAGE"],"exit_code":int(os.environ["RAT_CODE"]),"launcher_pid":int(os.environ["RAT_PID"]),"updated_at":datetime.now(timezone.utc).isoformat(),"stage_order":["t1_rat_initial","t4_rat_no_amplitude","t5_rat_channel_only","t6_rat_amplitude_only","t2_rat_late","t3_rat_all","t7_faar_rat_initial"]}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
}
cleanup() { rc=$?; trap - EXIT INT TERM; rmdir "$LOCK" 2>/dev/null || true; ((rc == 0)) || state failed "$CURRENT" "$rc" || true; exit "$rc"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
echo $$ > "$STATE/launcher.pid"; echo "$RUN_ID" > "$STATE/run_id.txt"; state running "$CURRENT"

smoke() {
  local stage=$1 name=rat_${RUN_ID}_smoke_${1}
  CURRENT=smoke.${stage}; state running "$CURRENT"
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_rat_worker.py" --root "$ROOT" --stage "$stage" --name "$name" --seed 0 --epochs 1 > "$TRAIN/${name}.log" 2>&1
  test -s "$TRAIN/$name/weights/last.pt"
  "$PY" -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); assert "nan" not in p.read_text(encoding="utf-8",errors="replace").lower()' "$TRAIN/${name}.log"
  rm -rf "$TRAIN/$name"; rm -f "$TRAIN/${name}.log" "$TRAIN/${name}.pid"
}
start_train() { local stage=$1 gpu=$2 seed=$3 name=rat_${RUN_ID}_${1}_seed${3}; CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/tools/train_rat_worker.py" --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" > "$TRAIN/${name}.log" 2>&1 & PIDS+=("$!"); NAMES+=("$name"); echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"; }
wait_train() { local bad=0; for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" && test -s "$TRAIN/${NAMES[$i]}/weights/best.pt" || bad=1; done; return "$bad"; }
start_test() { local stage=$1 gpu=$2 seed=$3 name=rat_${RUN_ID}_${1}_seed${3}; CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/test.py" --weights "$TRAIN/$name/weights/best.pt" --name "$name" --device 0 --batch 16 --imgsz 640 > "$TEST/${name}.log" 2>&1 & PIDS+=("$!"); NAMES+=("$name"); echo "$!" > "$STATE/${stage}_seed${seed}.test.pid"; }
wait_test() { local bad=0; for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" && test -s "$TEST/${NAMES[$i]}/summary_metrics.json" || bad=1; done; return "$bad"; }

DONE=()
for stage in "${STAGES[@]}"; do
  smoke "$stage"
  CURRENT=${stage}.train; state running "$CURRENT"; PIDS=(); NAMES=(); start_train "$stage" 0 0; start_train "$stage" 1 1; start_train "$stage" 2 2; wait_train
  CURRENT=${stage}.test; state running "$CURRENT"; PIDS=(); NAMES=(); start_test "$stage" 0 0; start_test "$stage" 1 1; start_test "$stage" 2 2; wait_test
  "$PY" "$ROOT/tools/collect_rat_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "$stage"
  DONE+=("$stage"); state stage_complete "$stage"
done
CURRENT=reporting; state running "$CURRENT"; "$PY" "$ROOT/tools/collect_rat_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${DONE[@]}"; CURRENT=complete; state complete "$CURRENT"
