#!/usr/bin/env bash
# Resume interrupted U7, complete U8, test every seed, and persist a verified completion state for RAMP.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
DATA="/home/room305/ZZF/URPC2020half/data.yaml"
STATE="$ROOT/runs/urr_dcra_u2_u8"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID="${URR_DCRA_RUN_ID:-urr_dcra_20260720_142421}"
LOCK="$STATE/.u7_u8_resume_lock"
CURRENT="initializing"
PID="$$"
ALL_STAGES=(u2_m7_rho020 u3_adaptive u4_mean u5_power05 u6_power20 u7_strict u8_none)
STAGE_ORDER_CSV="$(IFS=,; echo "${ALL_STAGES[*]}")"
DATA_ROOT="$(dirname "$DATA")"
TRAIN_CACHE="$DATA_ROOT/labels/train.cache"
VAL_CACHE="$DATA_ROOT/labels/test.cache"
CACHE_READY=false

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "URR-DCRA U7/U8 recovery chain already running" >&2; exit 73; }

write_state() {
  local status="$1"
  local stage="$2"
  local code="$3"
  URR_DCRA_STATUS="$status" URR_DCRA_STAGE="$stage" URR_DCRA_CODE="$code" URR_DCRA_RUN_ID="$RUN_ID" \
    URR_DCRA_PID="$PID" URR_DCRA_STATE="$STATE" URR_DCRA_DATA="$DATA" URR_DCRA_STAGE_ORDER="$STAGE_ORDER_CSV" \
    "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["URR_DCRA_STATE"]) / "state.json"
payload = {
    "run_id": os.environ["URR_DCRA_RUN_ID"],
    "status": os.environ["URR_DCRA_STATUS"],
    "stage": os.environ["URR_DCRA_STAGE"],
    "exit_code": int(os.environ["URR_DCRA_CODE"]),
    "launcher_pid": int(os.environ["URR_DCRA_PID"]),
    "dataset": os.environ["URR_DCRA_DATA"],
    "epochs": 300,
    "patience": 40,
    "workers": 2,
    "amp": False,
    "plots": False,
    "parallel_workers_per_stage": 3,
    "stage_order": os.environ["URR_DCRA_STAGE_ORDER"].split(","),
    "recovery": "resume_u7_then_u8",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temporary.replace(path)
PY
}

cleanup() {
  rmdir "$LOCK" 2>/dev/null || true
}

on_error() {
  local rc="$?"
  write_state failed "$CURRENT" "$rc" || true
  exit "$rc"
}

trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

train_complete() {
  local name="$1"
  [[ -s "$TRAIN_DIR/$name/weights/best.pt" ]] || return 1
  "$PY" - "$TRAIN_DIR/$name.train.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
raise SystemExit(0 if json.loads(path.read_text(encoding="utf-8")).get("completed_at") else 1)
PY
}

test_complete() {
  local name="$1"
  [[ -s "$TEST_DIR/$name/summary_metrics.json" && -s "$TEST_DIR/$name/scale_ap_metrics.json" ]]
}

wait_for_dataset_caches() {
  local seed_pid="$1"
  local elapsed=0
  while ((elapsed < 120)); do
    if [[ -s "$TRAIN_CACHE" && -s "$VAL_CACHE" ]]; then
      CACHE_READY=true
      return 0
    fi
    if ! kill -0 "$seed_pid" 2>/dev/null; then
      echo "cache-preparation worker exited before the shared train/test caches were ready" >&2
      return 1
    fi
    sleep 1
    ((elapsed += 1))
  done
  echo "timed out while preparing shared train/test dataset caches" >&2
  return 1
}

test_one() {
  local name="$1"
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true PIN_MEMORY=false "$PY" "$ROOT/test.py" \
    --weights "$TRAIN_DIR/$name/weights/best.pt" --data "$DATA" --name "$name" --device 0 --batch 16 --workers 2 --imgsz 640 \
    > "$TEST_DIR/$name.log" 2>&1
  test_complete "$name"
}

run_stage() {
  local stage="$1"
  CURRENT="${stage}.seeds0_1_2.train"
  write_state running "$CURRENT" 0
  local -a pids=()
  local failed=0

  if [[ -s "$TRAIN_CACHE" && -s "$VAL_CACHE" ]]; then
    CACHE_READY=true
  fi

  for seed in 0 1 2; do
    local name="${RUN_ID}_${stage}_seed${seed}"
    if train_complete "$name"; then
      echo "${stage} seed${seed}: completed training already present"
      continue
    fi
    local -a resume_arg=()
    if [[ -d "$TRAIN_DIR/$name" ]]; then
      resume_arg=(--resume)
    fi
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true PIN_MEMORY=false "$PY" "$ROOT/tools/train_urr_dcra_worker.py" \
      --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 "${resume_arg[@]}" \
      > "$TRAIN_DIR/$name.log" 2>&1 &
    local worker_pid="$!"
    pids+=("$worker_pid")
    echo "$worker_pid" > "$STATE/${stage}_seed${seed}.train.pid"
    if [[ "$CACHE_READY" == false ]]; then
      wait_for_dataset_caches "$worker_pid"
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ((failed == 0)) || return 75

  CURRENT="${stage}.seeds0_1_2.test"
  write_state running "$CURRENT" 0
  for seed in 0 1 2; do
    local name="${RUN_ID}_${stage}_seed${seed}"
    train_complete "$name" || { echo "incomplete training: $name" >&2; return 76; }
    if test_complete "$name"; then
      echo "${stage} seed${seed}: completed test already present"
      continue
    fi
    test_one "$name"
  done
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
"$PY" -m py_compile "$ROOT/tools/train_urr_dcra_worker.py" "$ROOT/test.py"
echo "$$" > "$STATE/u7_u8_resume_launcher.pid"

run_stage u7_strict
"$PY" "$ROOT/tools/collect_urr_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${ALL_STAGES[@]:0:6}"
run_stage u8_none
"$PY" "$ROOT/tools/collect_urr_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${ALL_STAGES[@]}"

CURRENT="complete"
write_state complete "$CURRENT" 0
echo "URR-DCRA U7/U8 recovery chain completed: run_id=$RUN_ID"
