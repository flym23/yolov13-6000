#!/usr/bin/env bash
# Start P0--P3 only after the CPCR chain in yolov13yuan-6000 has finished.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
DATA="/home/room305/ZZF/URPC2020half/data.yaml"
PREDECESSOR_SCRIPT="/home/room305/ZZF/yolov13yuan-6000/run_cpcr_ablation.sh"
STATE="$ROOT/runs/spc_lcer_dcra_p0_p3"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID="${SPC_LCER_DCRA_RUN_ID:-spc_lcer_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LOCK="$STATE/.chain_lock"
ALL_STAGES=(p0_baseline p1_lcer_endpoint p2_naive_phase p3_spc_main)
STAGE_ORDER_CSV="$(IFS=,; echo "${ALL_STAGES[*]}")"
CURRENT="initializing"
PID="$$"
COMPLETED=()

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "SPC-LCER-DCRA P0--P3 chain already running" >&2; exit 73; }

write_state() {
  local status="$1"
  local stage="$2"
  local code="$3"
  SPC_STATUS="$status" SPC_STAGE="$stage" SPC_CODE="$code" SPC_RUN_ID="$RUN_ID" SPC_PID="$PID" \
    SPC_STATE="$STATE" SPC_DATA="$DATA" SPC_PREDECESSOR="$PREDECESSOR_SCRIPT" SPC_STAGES="$STAGE_ORDER_CSV" \
    SPC_COMPLETED="$(IFS=,; echo "${COMPLETED[*]}")" "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["SPC_STATE"]) / "state.json"
payload = {
    "run_id": os.environ["SPC_RUN_ID"],
    "status": os.environ["SPC_STATUS"],
    "stage": os.environ["SPC_STAGE"],
    "completed_stages": [item for item in os.environ["SPC_COMPLETED"].split(",") if item],
    "exit_code": int(os.environ["SPC_CODE"]),
    "launcher_pid": int(os.environ["SPC_PID"]),
    "dataset": os.environ["SPC_DATA"],
    "predecessor_script": os.environ["SPC_PREDECESSOR"],
    "epochs": 300,
    "patience": 40,
    "workers": 2,
    "amp": False,
    "plots": False,
    "parallel_workers_per_stage": 3,
    "stage_order": os.environ["SPC_STAGES"].split(","),
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

wait_for_cpcr() {
  CURRENT="waiting_for_cpcr"
  while pgrep -f "$PREDECESSOR_SCRIPT" >/dev/null; do
    write_state waiting_for_predecessor "$CURRENT" 0
    sleep 30
  done
}

preflight() {
  CURRENT="preflight"
  write_state running "$CURRENT" 0
  "$PY" -m compileall -q ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py \
    spc_lcer_dcra_experiments.py tools/train_spc_lcer_dcra_worker.py tools/collect_spc_lcer_dcra_ablation.py test_spc_lcer_dcra.py
  "$PY" test_spc_lcer_dcra.py > "$STATE/mechanism_${RUN_ID}.log" 2>&1
  for yaml in \
    "$ROOT/ultralytics/cfg/models/v13/yolov13-spc-p1-lcer-endpoint.yaml" \
    "$ROOT/ultralytics/cfg/models/v13/yolov13-spc-p2-naive.yaml" \
    "$ROOT/ultralytics/cfg/models/v13/yolov13-spc-p3-main.yaml"; do
    "$PY" test_spc_lcer_dcra.py --yaml "$yaml" >> "$STATE/interface_${RUN_ID}.log" 2>&1
  done
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
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true PIN_MEMORY=false "$PY" "$ROOT/tools/train_spc_lcer_dcra_worker.py" \
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
  "$PY" "$ROOT/tools/collect_spc_lcer_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}"
}

echo "$$" > "$STATE/launcher.pid"
write_state initializing "$CURRENT" 0
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
[[ -f "$PREDECESSOR_SCRIPT" ]] || { echo "CPCR predecessor script unavailable: $PREDECESSOR_SCRIPT" >&2; exit 80; }
wait_for_cpcr
preflight
for stage in "${ALL_STAGES[@]}"; do
  run_stage "$stage"
done
CURRENT="complete"
write_state complete "$CURRENT" 0
echo "SPC-LCER-DCRA P0--P3 chain completed: run_id=$RUN_ID"
