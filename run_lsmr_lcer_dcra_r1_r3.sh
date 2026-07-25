#!/usr/bin/env bash
# Run R1--R3 LSMR-LCER-DCRA ablations sequentially by stage and in parallel by seed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
DATA="/home/room305/ZZF/URPC2020half/data.yaml"
LCER_BASELINE="/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json"
SPC_BASELINE="/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json"
STATE="$ROOT/runs/lsmr_lcer_dcra_r1_r3"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID="${LSMR_LCER_DCRA_RUN_ID:-lsmr_lcer_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LOCK="$STATE/.chain_lock"
ALL_STAGES=(r1_matched_endpoint r2_global_endpoint r3_lsmr_local)
STAGE_ORDER_CSV="$(IFS=,; echo "${ALL_STAGES[*]}")"
CURRENT="initializing"
PID="$$"
COMPLETED=()

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "LSMR-LCER-DCRA R1--R3 chain already running" >&2; exit 73; }

write_state() {
  local status="$1"
  local stage="$2"
  local code="$3"
  LSMR_STATUS="$status" LSMR_STAGE="$stage" LSMR_CODE="$code" LSMR_RUN_ID="$RUN_ID" LSMR_PID="$PID" \
    LSMR_STATE="$STATE" LSMR_DATA="$DATA" LSMR_STAGES="$STAGE_ORDER_CSV" \
    LSMR_LCER_BASELINE="$LCER_BASELINE" LSMR_SPC_BASELINE="$SPC_BASELINE" \
    LSMR_COMPLETED="$(IFS=,; echo "${COMPLETED[*]}")" "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["LSMR_STATE"]) / "state.json"
payload = {
    "run_id": os.environ["LSMR_RUN_ID"],
    "status": os.environ["LSMR_STATUS"],
    "stage": os.environ["LSMR_STAGE"],
    "completed_stages": [item for item in os.environ["LSMR_COMPLETED"].split(",") if item],
    "exit_code": int(os.environ["LSMR_CODE"]),
    "launcher_pid": int(os.environ["LSMR_PID"]),
    "dataset": os.environ["LSMR_DATA"],
    "reused_baselines": [os.environ["LSMR_LCER_BASELINE"], os.environ["LSMR_SPC_BASELINE"]],
    "epochs": 300,
    "patience": 40,
    "workers": 2,
    "amp": False,
    "plots": False,
    "parallel_workers_per_stage": 3,
    "stage_order": os.environ["LSMR_STAGES"].split(","),
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
    lsmr_lcer_dcra_experiments.py tools/train_lsmr_lcer_dcra_worker.py tools/collect_lsmr_lcer_dcra_ablation.py test_lsmr_lcer_dcra.py
  "$PY" test_lsmr_lcer_dcra.py > "$STATE/mechanism_${RUN_ID}.log" 2>&1
  for yaml in \
    "$ROOT/ultralytics/cfg/models/v13/yolov13-lsmr-lcer-dcra-r1-matched.yaml" \
    "$ROOT/ultralytics/cfg/models/v13/yolov13-lsmr-lcer-dcra-r2-global.yaml" \
    "$ROOT/ultralytics/cfg/models/v13/yolov13-lsmr-lcer-dcra-r3-local.yaml"; do
    "$PY" test_lsmr_lcer_dcra.py --yaml "$yaml" >> "$STATE/interface_${RUN_ID}.log" 2>&1
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
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true PIN_MEMORY=false "$PY" "$ROOT/tools/train_lsmr_lcer_dcra_worker.py" \
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
  "$PY" "$ROOT/tools/collect_lsmr_lcer_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}" \
    --reference-baseline "$LCER_BASELINE" --reference-baseline "$SPC_BASELINE"
}

echo "$$" > "$STATE/launcher.pid"
write_state initializing "$CURRENT" 0
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
[[ -f "$LCER_BASELINE" && -f "$SPC_BASELINE" ]] || { echo "Reused baseline summary unavailable" >&2; exit 80; }
preflight
for stage in "${ALL_STAGES[@]}"; do
  run_stage "$stage"
done
CURRENT="complete"
write_state complete "$CURRENT" 0
echo "LSMR-LCER-DCRA R1--R3 chain completed: run_id=$RUN_ID"
