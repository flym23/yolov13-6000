#!/usr/bin/env bash
# L3-CRU T1--T7: starts after the CARM chain, then runs pretrained stage-synchronous three-seed experiments.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
PRETRAINED="$ROOT/yolov13n.pt"
CARM_CHAIN_PATTERN='[r]un_carm_ablation.sh'
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
STATE_DIR="$TRAIN_DIR"
RUN_ID=${L3CRU_RUN_ID:-l3cru_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
PREFLIGHT_MARKER="$TRAIN_DIR/.l3cru_preflight_v2_ready.marker"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
STAGES=(t1_amsc t2_bgdr t3_ugdr t4_amsc_bgdr t5_amsc_ugdr t6_bgdr_ugdr t7_full)
START_STAGE=${L3CRU_START_STAGE:-${STAGES[0]}}
COMPLETED=()
if [[ -n ${L3CRU_COMPLETED_STAGES:-} ]]; then
  IFS=, read -r -a COMPLETED <<<"$L3CRU_COMPLETED_STAGES"
fi
CURRENT_STAGE=initializing
L0_REFERENCE="$ROOT/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json"
P0_REFERENCE="$ROOT/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json"

mkdir -p "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "L3-CRU chain already running for $RUN_ID" >&2; exit 73; }
export WANDB_DISABLED=true PIN_MEMORY=false

write_state() {
  "$PY" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; p=Path(sys.argv[1]); d={"run_id":sys.argv[2],"status":sys.argv[3],"stage":sys.argv[4],"detail":sys.argv[5],"launcher_pid":int(sys.argv[6]),"launch_mode":"after_carm_chain_completion","references":{"l0":sys.argv[7],"p0":sys.argv[8]},"initialization":{"method":"YOLO.load","pretrained":sys.argv[9]},"completed_stages":sys.argv[10:],"settings":{"epochs":300,"patience":40,"device":0,"workers":2,"amp":False,"deterministic":True,"plots":False,"imgsz":640,"batch":16,"parallel_seed_processes":3,"stage_synchronous":True},"updated_at":datetime.now(timezone.utc).isoformat()}; p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")' "$STATE_FILE" "$RUN_ID" "$1" "$CURRENT_STAGE" "${2:-}" "$$" "$L0_REFERENCE" "$P0_REFERENCE" "$PRETRAINED" "${COMPLETED[@]}"
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
on_error() { code=$?; write_state failed "exit_code=$code" || true; exit "$code"; }
trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_group() {
  local label="$1" failed=0 pid other
  shift
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
      for other in "$@"; do
        [[ "$other" == "$pid" ]] || { kill -0 "$other" 2>/dev/null && kill "$other" 2>/dev/null || true; }
      done
    fi
  done
  (( failed == 0 )) || { echo "L3-CRU $CURRENT_STAGE $label failed" >&2; return 75; }
}

[[ -x "$PY" && -f "$DATA" && -f "$PRETRAINED" && -f "$L0_REFERENCE" && -f "$P0_REFERENCE" ]] || {
  echo "L3-CRU input file missing" >&2
  exit 78
}

echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=waiting_carm
write_state waiting "waiting_for_carm_chain=true"
while pgrep -f "$CARM_CHAIN_PATTERN" >/dev/null; do
  echo "Waiting for the CARM chain to finish..."
  sleep 60
done
CURRENT_STAGE=preflight
write_state running "carm_chain_finished=true; original_yolov13_reused=true; initialization=yolov13n.pt"
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
  "$PY" "$ROOT/tests/test_l3cru_repo_integration.py" --yaml-dir "$ROOT/ultralytics/cfg/models/v13/l3cru" --imgsz 64 --device cpu
  "$PY" "$ROOT/tests/test_l3cru_repo_integration.py" --yaml-dir "$ROOT/ultralytics/cfg/models/v13/l3cru" --imgsz 640 --device 0
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)" >"$PREFLIGHT_MARKER"
fi

start_found=false
for stage in "${STAGES[@]}"; do
  if [[ "$stage" == "$START_STAGE" ]]; then
    start_found=true
  fi
  [[ "$start_found" == true ]] || continue
  CURRENT_STAGE="$stage"
  write_state training "carm_chain_finished=true; initialization=yolov13n.pt; three_parallel_seeds=true"
  TRAIN_PIDS=()
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/run_l3cru_ablation.py" --root "$ROOT" --stage "$stage" --data "$DATA" \
      --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    TRAIN_PIDS+=("$!")
  done
  printf '%s\n' "${TRAIN_PIDS[@]}" >"$STATE_DIR/$RUN_ID.$stage.train.pids"
  wait_group training "${TRAIN_PIDS[@]}"

  write_state testing "three_parallel_seeds=true"
  TEST_PIDS=()
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    weights="$TRAIN_DIR/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; exit 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 \
      >"$TEST_DIR/$name.log" 2>&1 &
    TEST_PIDS+=("$!")
  done
  printf '%s\n' "${TEST_PIDS[@]}" >"$STATE_DIR/$RUN_ID.$stage.test.pids"
  wait_group testing "${TEST_PIDS[@]}"
  COMPLETED+=("$stage")
done

[[ "$start_found" == true ]] || { echo "Unknown L3-CRU restart stage: $START_STAGE" >&2; exit 74; }

CURRENT_STAGE=complete
write_state complete "t1_t7_finished=true; original_yolov13_reused=true; stage_synchronous=true"
echo "L3-CRU T1--T7 chain completed: run_id=$RUN_ID"
