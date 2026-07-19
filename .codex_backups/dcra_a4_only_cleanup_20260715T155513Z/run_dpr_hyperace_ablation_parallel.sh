#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rom305/zzf/yolov13-305
PY=/home/rom305/miniconda3/envs/yolov13/bin/python
TRAIN=$ROOT/runs/train
TEST=$ROOT/runs/test
STATE=$ROOT/runs/dpr_hyperace_ablation
RUN_ID=${DPR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOCK=$STATE/chain.lock
CURRENT=initialization
STAGES=(h1_dpr_p3_spd h4_dpr_dwdown h5_dpr_replace h2_dpr_p5 h3_dpr_dual h6_faar_dpr)

mkdir -p "$TRAIN" "$TEST" "$STATE"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir "$LOCK" 2>/dev/null || { echo "DPR-HyperACE chain already running" >&2; exit 73; }
state() {
  DPR_STATUS=$1 DPR_STAGE=$2 DPR_CODE=${3:-0} DPR_RUN_ID=$RUN_ID DPR_PID=$$ DPR_STATE=$STATE "$PY" -c 'import json,os; from pathlib import Path; from datetime import datetime,timezone; p=Path(os.environ["DPR_STATE"])/"state.json"; d={"run_id":os.environ["DPR_RUN_ID"],"status":os.environ["DPR_STATUS"],"stage":os.environ["DPR_STAGE"],"exit_code":int(os.environ["DPR_CODE"]),"launcher_pid":int(os.environ["DPR_PID"]),"updated_at":datetime.now(timezone.utc).isoformat(),"stage_order":["h1_dpr_p3_spd","h4_dpr_dwdown","h5_dpr_replace","h2_dpr_p5","h3_dpr_dual","h6_faar_dpr"]}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
}
cleanup() { rc=$?; trap - EXIT INT TERM; rmdir "$LOCK" 2>/dev/null || true; ((rc == 0)) || state failed "$CURRENT" "$rc" || true; exit "$rc"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo $$ > "$STATE/launcher.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
state running "$CURRENT"

smoke() {
  local stage=$1 gpu=$2 name=dpr_${RUN_ID}_smoke_${1}
  CURRENT=smoke.${stage}
  state running "$CURRENT"
  echo $$ > "$STATE/${stage}.smoke.pid"
  CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/tools/train_dpr_hyperace_worker.py" --root "$ROOT" --stage "$stage" --name "$name" --seed 0 --epochs 1 > "$TRAIN/${name}.log" 2>&1
  test -s "$TRAIN/$name/weights/last.pt"
  "$PY" -c 'import sys; from pathlib import Path; t=Path(sys.argv[1]).read_text(encoding="utf-8",errors="replace").lower(); assert "traceback" not in t and "runtimeerror" not in t and "nan" not in t; assert "dpr alpha optimizer registration: pass" in t; assert "dpr alpha values after training" in t' "$TRAIN/${name}.log"
  rm -rf -- "$TRAIN/$name"
  rm -f -- "$TRAIN/${name}.log" "$TRAIN/${name}.pid"
}

smoke h1_dpr_p3_spd 0
smoke h4_dpr_dwdown 1
smoke h5_dpr_replace 2
smoke h2_dpr_p5 0
smoke h3_dpr_dual 1
smoke h6_faar_dpr 2
state smoke_complete all

start_train() {
  local stage=$1 gpu=$2 seed=$3 name=dpr_${RUN_ID}_${1}_seed${3}
  CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/tools/train_dpr_hyperace_worker.py" --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" > "$TRAIN/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"
}
wait_train() { local bad=0; for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" && test -s "$TRAIN/${NAMES[$i]}/weights/best.pt" || bad=1; done; return "$bad"; }
start_test() {
  local stage=$1 gpu=$2 seed=$3 name=dpr_${RUN_ID}_${1}_seed${3}
  CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/test.py" --weights "$TRAIN/$name/weights/best.pt" --name "$name" --device 0 --batch 16 --imgsz 640 > "$TEST/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  echo "$!" > "$STATE/${stage}_seed${seed}.test.pid"
}
wait_test() { local bad=0; for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" && test -s "$TEST/${NAMES[$i]}/summary_metrics.json" || bad=1; done; return "$bad"; }

DONE=()
for stage in "${STAGES[@]}"; do
  CURRENT=${stage}.train
  state running "$CURRENT"
  PIDS=(); NAMES=()
  start_train "$stage" 0 0
  start_train "$stage" 1 1
  start_train "$stage" 2 2
  wait_train
  CURRENT=${stage}.test
  state running "$CURRENT"
  PIDS=(); NAMES=()
  start_test "$stage" 0 0
  start_test "$stage" 1 1
  start_test "$stage" 2 2
  wait_test
  "$PY" "$ROOT/tools/collect_dpr_hyperace_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "$stage"
  DONE+=("$stage")
  state stage_complete "$stage"
done
CURRENT=reporting
state running "$CURRENT"
"$PY" "$ROOT/tools/collect_dpr_hyperace_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${DONE[@]}"
CURRENT=complete
state complete "$CURRENT"
