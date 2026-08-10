#!/usr/bin/env bash
# Wait for the active yolov26 three-seed run, then reproduce TDR D0 with three seeds.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
YOLO26_RUN_DIR="${YOLO26_RUN_DIR:-/home/room305/ZZF/yolov26/runs/urpc2019_yolo26_original_20260807}"
RUN_ID="${TDR_URPC2019_RUN_ID:-tdr_urpc2019_$(date -u +%Y%m%d_%H%M%S)}"
STATE="$ROOT/runs/tdr_urpc2019"
PROJECT="$STATE/$RUN_ID"
LOG_DIR="$PROJECT/logs"

mkdir -p "$LOG_DIR"
printf '{"status":"waiting_yolov26","run_id":"%s"}\n' "$RUN_ID" > "$PROJECT/state.json"

for seed in 0 1 2; do
  pid_file="$YOLO26_RUN_DIR/seed${seed}.pid"
  [[ -s "$pid_file" ]] || { echo "missing yolov26 PID: $pid_file" >&2; exit 80; }
done

is_yolov26_worker() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local cwd command_line
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cwd" == /home/room305/ZZF/yolov26* || "$command_line" == *"/home/room305/ZZF/yolov26"* ]]
}

while :; do
  alive=0
  for seed in 0 1 2; do
    pid="$(tr -d '[:space:]' < "$YOLO26_RUN_DIR/seed${seed}.pid")"
    if is_yolov26_worker "$pid"; then
      alive=1
      break
    fi
  done
  ((alive == 0)) && break
  sleep 120
done

printf '{"status":"running_d0","run_id":"%s"}\n' "$RUN_ID" > "$PROJECT/state.json"
cd "$ROOT"
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 81; }

pids=()
for seed in 0 1 2; do
  name="${RUN_ID}_d0_seed${seed}"
  "$PY" "$ROOT/tools/train_tdr_worker.py" --variant d0 --seed "$seed" --project "$PROJECT" --name "$name" \
    > "$LOG_DIR/seed${seed}.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  printf '%s\n' "$pid" > "$PROJECT/seed${seed}.pid"
done

for _ in 0 1 2; do
  if ! wait -n; then
    for other in "${pids[@]}"; do kill "$other" 2>/dev/null || true; done
    printf '{"status":"failed","run_id":"%s"}\n' "$RUN_ID" > "$PROJECT/state.json"
    exit 82
  fi
done

printf '{"status":"complete_d0","run_id":"%s"}\n' "$RUN_ID" > "$PROJECT/state.json"
