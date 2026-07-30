#!/usr/bin/env bash
# L3-CRU T0--T7: three deterministic seed workers run concurrently on device 0.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
YAML_DIR="$ROOT/ultralytics/cfg/models/v13/l3cru"
CARM_STATE_FILE=${1:?usage: run_l3cru_t0_t7.sh /absolute/path/to/completed-carm-state.json}
STATE_DIR="$ROOT/runs/l3cru"
TRAIN_LOG_DIR="$ROOT/runs/train"
RUN_ID=${L3CRU_RUN_ID:-l3cru_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
PREFLIGHT_MARKER="$STATE_DIR/l3cru_preflight_ready.marker"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
GROUPS=(t0_baseline t1_amsc t2_bgdr t3_ugdr t4_amsc_bgdr t5_amsc_ugdr t6_bgdr_ugdr t7_full)
L0_REFERENCE="$ROOT/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json"
P0_REFERENCE="$ROOT/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json"

mkdir -p "$STATE_DIR" "$TRAIN_LOG_DIR"
[[ -x "$PY" && -f "$DATA" && -d "$YAML_DIR" && -f "$CARM_STATE_FILE" && -f "$L0_REFERENCE" && -f "$P0_REFERENCE" ]] || {
  echo "L3-CRU input file missing" >&2
  exit 78
}
CARM_STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$CARM_STATE_FILE")
[[ "$CARM_STATUS" == "complete" ]] || { echo "CARM has not completed: status=$CARM_STATUS" >&2; exit 79; }
mkdir "$LOCK" 2>/dev/null || { echo "L3-CRU chain already running for $RUN_ID" >&2; exit 73; }

write_state() {
  "$PY" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; p=Path(sys.argv[1]); d={"run_id":sys.argv[2],"status":sys.argv[3],"detail":sys.argv[4],"launcher_pid":int(sys.argv[5]),"dependency":{"carm_state":sys.argv[6]},"settings":{"epochs":300,"patience":40,"device":0,"workers":2,"amp":False,"deterministic":True,"plots":False,"imgsz":640,"batch":16,"parallel_seed_processes":3},"updated_at":datetime.now(timezone.utc).isoformat()}; p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")' "$STATE_FILE" "$RUN_ID" "$1" "${2:-}" "$$" "$CARM_STATE_FILE"
}
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
on_error() { code=$?; write_state failed "exit_code=$code" || true; exit "$code"; }
trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_group() {
  local failed=0 pid other
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
      for other in "$@"; do
        [[ "$other" == "$pid" ]] || { kill -0 "$other" 2>/dev/null && kill "$other" 2>/dev/null || true; }
      done
    fi
  done
  (( failed == 0 )) || { echo "L3-CRU seed worker failed" >&2; return 75; }
}

export WANDB_DISABLED=true PIN_MEMORY=false
echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
write_state preflight "carm_complete=true"
if [[ -f "$PREFLIGHT_MARKER" ]]; then
  echo "Using completed L3-CRU preflight: $PREFLIGHT_MARKER"
else
  "$PY" -m py_compile "$ROOT/ultralytics/nn/modules/block.py" "$ROOT/ultralytics/nn/modules/head.py" \
    "$ROOT/ultralytics/nn/modules/__init__.py" "$ROOT/ultralytics/nn/tasks.py" \
    "$ROOT/tests/AMSCLCERDCRAUp.py" "$ROOT/tests/BGDRP3Fuse.py" "$ROOT/tests/UGDRDetect.py" \
    "$ROOT/tests/test_l3cru_modules_isolated.py" "$ROOT/tests/audit_l3cru_second_pass.py" \
    "$ROOT/tests/test_l3cru_repo_integration.py" "$ROOT/tools/run_l3cru_ablation.py"
  "$PY" "$ROOT/tests/test_l3cru_modules_isolated.py"
  "$PY" "$ROOT/tests/audit_l3cru_second_pass.py"
  "$PY" "$ROOT/tests/test_l3cru_repo_integration.py" --yaml-dir "$YAML_DIR" --imgsz 64 --device cpu
  "$PY" "$ROOT/tests/test_l3cru_repo_integration.py" --yaml-dir "$YAML_DIR" --imgsz 640 --device 0
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)" >"$PREFLIGHT_MARKER"
fi
TEST_COMMAND="$PY $ROOT/test.py --weights {weights} --data {data} --name ${RUN_ID}_{group}_seed{seed} --device 0 --batch 16 --imgsz 640 --workers 2"

write_state training "three_parallel_seed_workers=true"
PIDS=()
for seed in 0 1 2; do
  "$PY" "$ROOT/tools/run_l3cru_ablation.py" --yaml-dir "$YAML_DIR" --data "$DATA" --device 0 \
    --groups "${GROUPS[@]}" --seeds "$seed" --epochs 300 --patience 40 --imgsz 640 --batch 16 --workers 2 \
    --project "$STATE_DIR/$RUN_ID/seed$seed" --run-id "$RUN_ID" --test-command "$TEST_COMMAND" \
    >"$TRAIN_LOG_DIR/$RUN_ID.seed$seed.log" 2>&1 &
  PIDS+=("$!")
done
printf '%s\n' "${PIDS[@]}" >"$STATE_DIR/$RUN_ID.worker.pids"
wait_group "${PIDS[@]}"
write_state complete "t0_t7_finished=true; three_parallel_seed_workers=true"
echo "L3-CRU T0--T7 chain completed: run_id=$RUN_ID"
