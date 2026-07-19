#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/room305/ZZF/yolov13-6000
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=$ROOT/data.yaml
TRAIN=$ROOT/runs/train
TEST=$ROOT/runs/test
STATE=$ROOT/runs/urpc2020_baseline_a4
RUN_ID=${URPC_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
PREFIX=urpc2020
LOCK=$STATE/chain.lock
CURRENT=initialization
ALL_STAGES=(baseline_original a4_tau020)
RESUME_FROM=${URPC_RESUME_FROM:-}
SKIP_SMOKE=${URPC_SKIP_SMOKE:-false}
STAGES=("${ALL_STAGES[@]}")

if [[ -n "$RESUME_FROM" ]]; then
  start_index=-1
  for index in "${!ALL_STAGES[@]}"; do
    if [[ "${ALL_STAGES[$index]}" == "$RESUME_FROM" ]]; then
      start_index=$index
      break
    fi
  done
  ((start_index >= 0)) || { echo "Unknown resume stage: $RESUME_FROM" >&2; exit 64; }
  STAGES=("${ALL_STAGES[@]:$start_index}")
fi

mkdir -p "$TRAIN" "$TEST" "$STATE"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir "$LOCK" 2>/dev/null || { echo "URPC2020 baseline/A4 chain already running" >&2; exit 73; }

write_state() {
  URPC_STATUS=$1 URPC_STAGE=$2 URPC_CODE=${3:-0} URPC_RUN_ID=$RUN_ID URPC_PID=$$ URPC_STATE=$STATE \
    "$PY" -c 'import json,os; from pathlib import Path; from datetime import datetime,timezone; p=Path(os.environ["URPC_STATE"])/"state.json"; d={"run_id":os.environ["URPC_RUN_ID"],"status":os.environ["URPC_STATUS"],"stage":os.environ["URPC_STAGE"],"exit_code":int(os.environ["URPC_CODE"]),"launcher_pid":int(os.environ["URPC_PID"]),"updated_at":datetime.now(timezone.utc).isoformat(),"dataset":"/home/room305/ZZF/URPC2020/data.yaml","precision":"fp32","amp":False,"physical_gpu":"cuda:0","parallel_seed_processes":3,"stage_order":["baseline_original","a4_tau020"]}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
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
"$PY" "$ROOT/tools/validate_dcra_models.py" --device cpu \
  > "$STATE/preflight_${RUN_ID}.log" 2>&1

prepare_dataset_cache() {
  CURRENT=dataset_cache
  write_state running "$CURRENT"
  "$PY" "$ROOT/tools/prepare_dcra_dataset_cache.py" --data "$DATA" --imgsz 640 --batch 16 \
    > "$STATE/dataset_cache_${RUN_ID}.log" 2>&1
}

train_one() {
  local stage=$1 seed=$2 name=$3 epochs=$4
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_dcra_worker.py" \
    --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs "$epochs" \
    > "$TRAIN/${name}.log" 2>&1
  test -s "$TRAIN/$name/weights/best.pt"
  test -s "$TRAIN/$name/weights/last.pt"
}

test_one() {
  local name=$1
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/test.py" \
    --weights "$TRAIN/$name/weights/best.pt" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 \
    > "$TEST/${name}.log" 2>&1
  test -s "$TEST/$name/summary_metrics.json"
  test -s "$TEST/$name/scale_ap_metrics.json"
}

audit_a4() {
  local name=$1
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/tools/audit_dcra_checkpoint.py" \
    "$TRAIN/$name/weights/best.pt" --imgsz 640 --device cuda:0 \
    > "$TEST/${name}.dcra_audit.log" 2>&1
}

if [[ "$SKIP_SMOKE" != "true" ]]; then
  prepare_dataset_cache
  CURRENT=smoke.baseline_original.train
  write_state running "$CURRENT"
  SMOKE=${PREFIX}_${RUN_ID}_smoke_baseline_original_seed0
  train_one baseline_original 0 "$SMOKE" 1
  CURRENT=smoke.baseline_original.test
  write_state running "$CURRENT"
  test_one "$SMOKE"
  write_state smoke_complete baseline_original
fi

for stage in "${STAGES[@]}"; do
  prepare_dataset_cache
  CURRENT=${stage}.seeds0_1_2.train
  write_state running "$CURRENT"
  PIDS=()
  for seed in 0 1 2; do
    name=${PREFIX}_${RUN_ID}_${stage}_seed${seed}
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_dcra_worker.py" \
      --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 \
      > "$TRAIN/${name}.log" 2>&1 &
    PIDS+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"
  done
  bad=0
  for pid in "${PIDS[@]}"; do wait "$pid" || bad=1; done
  ((bad == 0))
  for seed in 0 1 2; do
    test -s "$TRAIN/${PREFIX}_${RUN_ID}_${stage}_seed${seed}/weights/best.pt"
    test -s "$TRAIN/${PREFIX}_${RUN_ID}_${stage}_seed${seed}/weights/last.pt"
  done

  CURRENT=${stage}.seeds0_1_2.test
  write_state running "$CURRENT"
  PIDS=()
  for seed in 0 1 2; do
    name=${PREFIX}_${RUN_ID}_${stage}_seed${seed}
    test_one "$name" &
    PIDS+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.test.pid"
  done
  bad=0
  for pid in "${PIDS[@]}"; do wait "$pid" || bad=1; done
  ((bad == 0))

  if [[ "$stage" == "a4_tau020" ]]; then
    CURRENT=${stage}.seeds0_1_2.audit
    write_state running "$CURRENT"
    for seed in 0 1 2; do
      audit_a4 "${PREFIX}_${RUN_ID}_${stage}_seed${seed}"
    done
  fi

  "$PY" "$ROOT/tools/collect_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" \
    --prefix "$PREFIX" --stages "$stage" --seeds 0 1 2
  write_state stage_complete "$stage"
done

CURRENT=reporting
write_state running "$CURRENT"
"$PY" "$ROOT/tools/collect_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" \
  --prefix "$PREFIX" --stages "${ALL_STAGES[@]}" --seeds 0 1 2
CURRENT=complete
write_state complete "$CURRENT"
echo "URPC2020 baseline/A4 chain completed: run_id=$RUN_ID"
