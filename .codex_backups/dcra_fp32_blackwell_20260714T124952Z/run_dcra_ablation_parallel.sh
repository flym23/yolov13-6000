#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rtx6000/ZZF/yolov13-6000
PY=/home/rtx6000/.conda/envs/yolov13/bin/python
TRAIN=$ROOT/runs/train
TEST=$ROOT/runs/test
STATE=$ROOT/runs/dcra_ablation
RUN_ID=${DCRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOCK=$STATE/chain.lock
CURRENT=initialization
STAGES=(a1_main a2_no_entropy a3_deep_only a4_tau020 a5_k5)

mkdir -p "$TRAIN" "$TEST" "$STATE"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir "$LOCK" 2>/dev/null || { echo "DCRA chain already running" >&2; exit 73; }

write_state() {
  DCRA_STATUS=$1 DCRA_STAGE=$2 DCRA_CODE=${3:-0} DCRA_RUN_ID=$RUN_ID DCRA_PID=$$ DCRA_STATE=$STATE \
    "$PY" -c 'import json,os; from pathlib import Path; from datetime import datetime,timezone; p=Path(os.environ["DCRA_STATE"])/"state.json"; d={"run_id":os.environ["DCRA_RUN_ID"],"status":os.environ["DCRA_STATUS"],"stage":os.environ["DCRA_STAGE"],"exit_code":int(os.environ["DCRA_CODE"]),"launcher_pid":int(os.environ["DCRA_PID"]),"updated_at":datetime.now(timezone.utc).isoformat(),"stage_order":["a1_main","a2_no_entropy","a3_deep_only","a4_tau020","a5_k5"]}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
}

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  rmdir "$LOCK" 2>/dev/null || true
  ((rc == 0)) || write_state failed "$CURRENT" "$rc" || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo $$ > "$STATE/launcher.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
write_state running "$CURRENT"

CURRENT=preflight
write_state running "$CURRENT"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/tools/validate_dcra_models.py" --device cuda:0 \
  > "$STATE/preflight_${RUN_ID}.log" 2>&1

train_one() {
  local stage=$1 seed=$2 gpu=$3 epochs=$4 name=$5
  CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/tools/train_dcra_worker.py" \
    --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" --epochs "$epochs" \
    > "$TRAIN/${name}.log" 2>&1
  test -s "$TRAIN/$name/weights/best.pt"
  test -s "$TRAIN/$name/weights/last.pt"
}

test_one() {
  local name=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/test.py" \
    --weights "$TRAIN/$name/weights/best.pt" --name "$name" --device 0 --batch 16 --imgsz 640 \
    > "$TEST/${name}.log" 2>&1
  test -s "$TEST/$name/summary_metrics.json"
  test -s "$TEST/$name/scale_ap_metrics.json"
}

audit_one() {
  local name=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/tools/audit_dcra_checkpoint.py" \
    "$TRAIN/$name/weights/best.pt" --imgsz 640 --device cuda:0 \
    > "$TEST/${name}.dcra_audit.log" 2>&1
}

CURRENT=smoke.a1_main.train
write_state running "$CURRENT"
SMOKE=dcra_${RUN_ID}_smoke_a1_seed0
train_one a1_main 0 0 1 "$SMOKE"
CURRENT=smoke.a1_main.test
write_state running "$CURRENT"
test_one "$SMOKE" 0
audit_one "$SMOKE" 0
CUDA_VISIBLE_DEVICES=0 "$PY" -c 'import sys; from ultralytics import YOLO; [YOLO(path) for path in sys.argv[1:]]; print("Smoke best.pt and last.pt strict object reload passed.")' \
  "$TRAIN/$SMOKE/weights/best.pt" "$TRAIN/$SMOKE/weights/last.pt" \
  > "$TEST/${SMOKE}.reload.log" 2>&1
write_state smoke_complete a1_main

for stage in "${STAGES[@]}"; do
  CURRENT=${stage}.seeds0_1_2.train
  write_state running "$CURRENT"
  PIDS=()
  for seed in 0 1 2; do
    gpu=0
    name=dcra_${RUN_ID}_${stage}_seed${seed}
    CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/tools/train_dcra_worker.py" \
      --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" --epochs 200 \
      > "$TRAIN/${name}.log" 2>&1 &
    PIDS+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"
  done
  bad=0
  for pid in "${PIDS[@]}"; do wait "$pid" || bad=1; done
  ((bad == 0))
  for seed in 0 1 2; do
    test -s "$TRAIN/dcra_${RUN_ID}_${stage}_seed${seed}/weights/best.pt"
    test -s "$TRAIN/dcra_${RUN_ID}_${stage}_seed${seed}/weights/last.pt"
  done

  CURRENT=${stage}.seeds0_1_2.test
  write_state running "$CURRENT"
  PIDS=()
  for seed in 0 1 2; do
    gpu=0
    name=dcra_${RUN_ID}_${stage}_seed${seed}
    CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" "$ROOT/test.py" \
      --weights "$TRAIN/$name/weights/best.pt" --name "$name" --device 0 --batch 16 --imgsz 640 \
      > "$TEST/${name}.log" 2>&1 &
    PIDS+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.test.pid"
  done
  bad=0
  for pid in "${PIDS[@]}"; do wait "$pid" || bad=1; done
  ((bad == 0))

  CURRENT=${stage}.seeds0_1_2.audit
  write_state running "$CURRENT"
  PIDS=()
  for seed in 0 1 2; do
    name=dcra_${RUN_ID}_${stage}_seed${seed}
    test -s "$TEST/$name/summary_metrics.json"
    test -s "$TEST/$name/scale_ap_metrics.json"
    CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/tools/audit_dcra_checkpoint.py" \
      "$TRAIN/$name/weights/best.pt" --imgsz 640 --device cuda:0 \
      > "$TEST/${name}.dcra_audit.log" 2>&1 &
    PIDS+=("$!")
  done
  bad=0
  for pid in "${PIDS[@]}"; do wait "$pid" || bad=1; done
  ((bad == 0))

  "$PY" "$ROOT/tools/collect_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" \
    --stages "$stage" --seeds 0 1 2
  write_state stage_complete "$stage"
done

CURRENT=reporting
write_state running "$CURRENT"
"$PY" "$ROOT/tools/collect_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" \
  --stages "${STAGES[@]}" --seeds 0 1 2
CURRENT=complete
write_state complete "$CURRENT"
echo "DCRA ablation chain completed: run_id=$RUN_ID"
