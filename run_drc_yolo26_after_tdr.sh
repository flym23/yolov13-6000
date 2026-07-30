#!/usr/bin/env bash
# DRC-YOLO26 three-seed chain, started only after the TDR-YOLOv13 chain exits.
set -euo pipefail

DRC_ROOT="${DRC_ROOT:-/home/room305/ZZF/yolov26}"
TDR_ROOT="${TDR_ROOT:-/home/room305/ZZF/yolov13-6000}"
PYTHON="${PYTHON:-/home/room305/.conda/envs/yolov13/bin/python}"
RUN_ROOT="$DRC_ROOT/runs/urpc2020half_drc"
RAW_DATA="${RAW_DATA:-/home/room305/ZZF/URPC2020half}"
DATA="$DRC_ROOT/datasets/URPC2020half_grouped/urpc2020half_grouped.yaml"
STATE="$RUN_ROOT/chain_state.json"
mkdir -p "$RUN_ROOT"

state() {
  printf '{"stage":"%s","updated_at":"%s"}\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE"
}

state waiting_for_tdr
while pgrep -f "$TDR_ROOT/run_tdr_yolov13_t1_t7_after_gmr.sh" >/dev/null; do
  sleep 60
done

if [[ ! -f "$DATA" ]]; then
  state building_grouped_split
  "$PYTHON" "$DRC_ROOT/tools/urpc/build_grouped_split.py" \
    --images "$RAW_DATA/images" --labels "$RAW_DATA/labels" --output "$(dirname "$DATA")" --seed 2026
fi

run_stage() {
  local stage="$1"
  local project="$RUN_ROOT/$stage"
  local pids=()
  state "$stage"
  for seed in 0 1 2; do
    mkdir -p "$project"
    setsid "$PYTHON" "$DRC_ROOT/tools/urpc/train_drc_yolo26_worker.py" \
      --stage "$stage" --data "$DATA" --seed "$seed" --project "$project" --name "drc_n_${stage}_seed${seed}" \
      > "$project/seed${seed}.log" 2>&1 < /dev/null &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || return 1
}

run_stage smoke
run_stage formal
state complete
