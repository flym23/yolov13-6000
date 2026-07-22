#!/usr/bin/env bash
# Run the preregistered L0--L3 LCER-DCRA ablation, three seeds in parallel per stage.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
DATA="/home/room305/ZZF/URPC2020half/data.yaml"
STATE="$ROOT/runs/lcer_dcra_l0_l3"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID="${LCER_DCRA_RUN_ID:-lcer_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LOCK="$STATE/.chain_lock"
ALL_STAGES=(l0_baseline l1_strict_rho020 l2_channel_power20 l3_local_consensus)
STAGE_ORDER_CSV="$(IFS=,; echo "${ALL_STAGES[*]}")"
CURRENT="initializing"
PID="$$"
COMPLETED=()

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "LCER-DCRA L0--L3 chain already running" >&2; exit 73; }

write_state() {
  local status="$1"
  local stage="$2"
  local code="$3"
  LCER_DCRA_STATUS="$status" LCER_DCRA_STAGE="$stage" LCER_DCRA_CODE="$code" LCER_DCRA_RUN_ID="$RUN_ID" \
    LCER_DCRA_PID="$PID" LCER_DCRA_STATE="$STATE" LCER_DCRA_DATA="$DATA" LCER_DCRA_STAGE_ORDER="$STAGE_ORDER_CSV" \
    LCER_DCRA_COMPLETED="$(IFS=,; echo "${COMPLETED[*]}")" "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["LCER_DCRA_STATE"]) / "state.json"
completed = [stage for stage in os.environ["LCER_DCRA_COMPLETED"].split(",") if stage]
payload = {
    "run_id": os.environ["LCER_DCRA_RUN_ID"],
    "status": os.environ["LCER_DCRA_STATUS"],
    "stage": os.environ["LCER_DCRA_STAGE"],
    "completed_stages": completed,
    "exit_code": int(os.environ["LCER_DCRA_CODE"]),
    "launcher_pid": int(os.environ["LCER_DCRA_PID"]),
    "dataset": os.environ["LCER_DCRA_DATA"],
    "epochs": 300,
    "patience": 40,
    "workers": 2,
    "amp": False,
    "plots": False,
    "parallel_workers_per_stage": 3,
    "stage_order": os.environ["LCER_DCRA_STAGE_ORDER"].split(","),
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

preflight() {
  CURRENT="preflight"
  write_state running "$CURRENT" 0
  "$PY" -m compileall -q ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py \
    tests/test_lcer_dcra.py tools/train_lcer_dcra_worker.py tools/collect_lcer_dcra_ablation.py lcer_dcra_experiments.py
  "$PY" -m pytest -q tests/test_lcer_dcra.py > "$STATE/preflight_${RUN_ID}.log" 2>&1
}

test_one() {
  local name="$1"
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true PIN_MEMORY=false "$PY" "$ROOT/test.py" \
    --weights "$TRAIN_DIR/$name/weights/best.pt" --data "$DATA" --name "$name" \
    --device 0 --batch 16 --workers 2 --imgsz 640 > "$TEST_DIR/${name}.log" 2>&1
  [[ -s "$TEST_DIR/$name/summary_metrics.json" && -s "$TEST_DIR/$name/scale_ap_metrics.json" ]]
}

run_stage() {
  local stage="$1"
  local failed=0
  local -a pids=()
  CURRENT="${stage}.seeds0_1_2.train"
  write_state running "$CURRENT" 0
  for seed in 0 1 2; do
    local name="${RUN_ID}_${stage}_seed${seed}"
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true PIN_MEMORY=false "$PY" "$ROOT/tools/train_lcer_dcra_worker.py" \
      --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" \
      --epochs 300 --patience 40 > "$TRAIN_DIR/${name}.log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || return 75

  CURRENT="${stage}.seeds0_1_2.test"
  write_state running "$CURRENT" 0
  pids=()
  for seed in 0 1 2; do
    local name="${RUN_ID}_${stage}_seed${seed}"
    [[ -f "$TRAIN_DIR/$name/weights/best.pt" ]] || { echo "missing checkpoint: $name" >&2; return 76; }
    test_one "$name" &
    pids+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.test.pid"
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || return 77

  COMPLETED+=("$stage")
  "$PY" "$ROOT/tools/collect_lcer_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}"
}

echo "$$" > "$STATE/launcher.pid"
write_state initializing "$CURRENT" 0
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
preflight
for stage in "${ALL_STAGES[@]}"; do
  run_stage "$stage"
done
CURRENT="complete"
write_state complete "$CURRENT" 0
echo "LCER-DCRA L0--L3 chain completed: run_id=$RUN_ID"
