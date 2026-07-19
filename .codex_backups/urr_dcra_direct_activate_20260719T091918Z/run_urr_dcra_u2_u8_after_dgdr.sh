#!/usr/bin/env bash
# Stage and run U2--U8 only after the active DGDR ablation has completed successfully.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
DATA="/home/room305/ZZF/URPC2020half/data.yaml"
PREDECESSOR_ROOT="/home/room305/ZZF/yolov13yuan-6000"
PREDECESSOR_SCRIPT="$PREDECESSOR_ROOT/run_dgdr_ablation_after_medcra.sh"
PREDECESSOR_STATE="$PREDECESSOR_ROOT/runs/dgdr_ablation/state.json"
STAGING="$ROOT/.codex_staging/urr_dcra"
STATE="$ROOT/runs/urr_dcra_u2_u8"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID="${URR_DCRA_RUN_ID:-urr_dcra_$(date -u +%Y%m%d_%H%M%S)}"
LOCK="$STATE/.chain_lock"
ALL_STAGES=(u2_m7_rho020 u3_adaptive u4_mean u5_power05 u6_power20 u7_strict u8_none)
STAGE_ORDER_CSV="$(IFS=,; echo "${ALL_STAGES[*]}")"
CURRENT="initializing"
PID="$$"
CACHE_READY=false
DATA_ROOT="$(dirname "$DATA")"
TRAIN_CACHE="$DATA_ROOT/labels/train.cache"
VAL_CACHE="$DATA_ROOT/labels/test.cache"
COMPLETED=()

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "URR-DCRA U2--U8 chain already running" >&2; exit 73; }

write_state() {
  local status="$1"
  local stage="$2"
  local code="$3"
  URR_DCRA_STATUS="$status" URR_DCRA_STAGE="$stage" URR_DCRA_CODE="$code" URR_DCRA_RUN_ID="$RUN_ID" \
    URR_DCRA_PID="$PID" URR_DCRA_STATE="$STATE" URR_DCRA_DATA="$DATA" URR_DCRA_PREDECESSOR="$PREDECESSOR_STATE" \
    URR_DCRA_STAGE_ORDER="$STAGE_ORDER_CSV" "$PY" - <<'PY'
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
    "dependency_state": os.environ["URR_DCRA_PREDECESSOR"],
    "stage_order": os.environ["URR_DCRA_STAGE_ORDER"].split(","),
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

dependency_status() {
  "$PY" - "$PREDECESSOR_STATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("missing")
print(str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")).lower())
PY
}

wait_for_dgdr() {
  CURRENT="waiting_for_dgdr"
  while true; do
    if pgrep -f "$PREDECESSOR_SCRIPT" >/dev/null; then
      write_state waiting_for_predecessor "$CURRENT" 0
    elif [[ ! -f "$PREDECESSOR_STATE" ]]; then
      write_state waiting_for_predecessor "$CURRENT" 0
    else
      local result
      result="$(dependency_status)"
      case "$result" in
        complete|completed|success)
          return 0
          ;;
        initializing|pending|queued|running|preflight_complete|stage_complete)
          write_state waiting_for_predecessor "$CURRENT" 0
          ;;
        *)
          echo "DGDR predecessor ended with status '$result'; URR-DCRA will not start." >&2
          return 74
          ;;
      esac
    fi
    sleep 30
  done
}

activate_staging() {
  CURRENT="activate_staged_urr_dcra"
  write_state running "$CURRENT" 0
  [[ -f "$STAGING/files.list" && -f "$STAGING/manifest.sha256" && -f "$STAGING/preimage.sha256" ]] || {
    echo "URR-DCRA staging manifest is missing: $STAGING" >&2
    return 64
  }
  (cd "$STAGING" && sha256sum -c manifest.sha256)
  # Refuse to overwrite a source file changed after the read-only remote snapshot.
  (cd "$ROOT" && sha256sum -c "$STAGING/preimage.sha256")
  local backup="$ROOT/.codex_backups/urr_dcra_activate_$(date -u +%Y%m%dT%H%M%SZ)"
  while IFS= read -r relative; do
    [[ -n "$relative" && "$relative" != /* && "$relative" != *".."* ]] || {
      echo "unsafe staging relative path: $relative" >&2
      return 64
    }
    [[ -f "$STAGING/$relative" ]] || { echo "staged file missing: $relative" >&2; return 65; }
    mkdir -p "$backup/$(dirname "$relative")" "$(dirname "$ROOT/$relative")"
    [[ ! -e "$ROOT/$relative" ]] || cp -p "$ROOT/$relative" "$backup/$relative"
    cp -p "$STAGING/$relative" "$ROOT/$relative"
  done < "$STAGING/files.list"
  printf '%s\n' "$backup" > "$STATE/activation_backup.txt"
}

preflight() {
  CURRENT="preflight"
  write_state running "$CURRENT" 0
  "$PY" -m compileall -q ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py \
    tests/test_urr_dcra_up.py tools/train_urr_dcra_worker.py tools/collect_urr_dcra_ablation.py urr_dcra_experiments.py
  "$PY" -m pytest -q tests/test_urr_dcra_up.py \
    -k "parameter or nearest or endpoint or confidence or reliability or power or centered or gradient or invalid" \
    > "$STATE/mechanism_${RUN_ID}.log" 2>&1
  "$PY" -m pytest -q tests/test_urr_dcra_up.py -k "yaml or topology or full_model or cuda" \
    > "$STATE/interface_${RUN_ID}.log" 2>&1
}

prepare_dataset_caches() {
  rm -f "$TRAIN_CACHE" "$TRAIN_CACHE.npy" "$VAL_CACHE" "$VAL_CACHE.npy"
  CACHE_READY=false
}

wait_for_dataset_caches() {
  local seed0_pid="$1"
  local elapsed=0
  while ((elapsed < 120)); do
    if [[ -s "$TRAIN_CACHE" && -s "$VAL_CACHE" ]]; then
      return 0
    fi
    if ! kill -0 "$seed0_pid" 2>/dev/null; then
      echo "seed-0 exited before shared train/test dataset caches were prepared" >&2
      return 1
    fi
    sleep 1
    ((elapsed += 1))
  done
  echo "timed out while waiting for shared train/test dataset caches" >&2
  return 1
}

test_one() {
  local name="$1"
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/test.py" --weights "$TRAIN_DIR/$name/weights/best.pt" \
    --data "$DATA" --name "$name" --device 0 --batch 16 --workers 2 --imgsz 640 > "$TEST_DIR/${name}.log" 2>&1
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
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_urr_dcra_worker.py" --root "$ROOT" --stage "$stage" \
      --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 > "$TRAIN_DIR/${name}.log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"
    if [[ "$CACHE_READY" == false && "$seed" == 0 ]]; then
      wait_for_dataset_caches "$!" || return 75
      CACHE_READY=true
    fi
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
  "$PY" "$ROOT/tools/collect_urr_dcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}"
}

echo "$$" > "$STATE/launcher.pid"
write_state initializing "$CURRENT" 0
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
wait_for_dgdr
activate_staging
preflight
prepare_dataset_caches
for stage in "${ALL_STAGES[@]}"; do
  run_stage "$stage"
done
CURRENT="complete"
write_state complete "$CURRENT" 0
echo "URR-DCRA U2--U8 chain completed: run_id=$RUN_ID"
