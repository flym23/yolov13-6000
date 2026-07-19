#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rtx6000/ZZF/yolov13-6000
PY=/home/rtx6000/.conda/envs/yolov13/bin/python
DATA=$ROOT/data.yaml
DURC_STATE=/home/rtx6000/ZZF/yolov13yuan-6000/runs/durc_ablation/state.json
STAGING=$ROOT/.codex_staging/medcra
STATE=$ROOT/runs/urpc2020_medcra_m0_m7
RUN_ID=${MEDCRA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOCK=$STATE/chain.lock
TRAIN=$ROOT/runs/train
TEST=$ROOT/runs/test
STAGES=(m0_original m1_a4 m2_full m3_no_moment m4_no_center m5_no_bound m6_rho005 m7_rho020)

mkdir -p "$STATE" "$TRAIN" "$TEST"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir "$LOCK" 2>/dev/null || { echo "ME-DCRA M0--M7 chain already running" >&2; exit 73; }

write_state() {
  MEDCRA_STATUS=$1 MEDCRA_STAGE=$2 MEDCRA_CODE=${3:-0} MEDCRA_RUN_ID=$RUN_ID MEDCRA_PID=$$ MEDCRA_STATE=$STATE \
    "$PY" -c 'import json,os; from pathlib import Path; from datetime import datetime,timezone; p=Path(os.environ["MEDCRA_STATE"])/"state.json"; d={"run_id":os.environ["MEDCRA_RUN_ID"],"status":os.environ["MEDCRA_STATUS"],"stage":os.environ["MEDCRA_STAGE"],"exit_code":int(os.environ["MEDCRA_CODE"]),"launcher_pid":int(os.environ["MEDCRA_PID"]),"dataset":"/home/rtx6000/ZZF/URPC2020/data.yaml","epochs":300,"patience":40,"amp":False,"dependency_state":"/home/rtx6000/ZZF/yolov13yuan-6000/runs/durc_ablation/state.json","stage_order":["m0_original","m1_a4","m2_full","m3_no_moment","m4_no_center","m5_no_bound","m6_rho005","m7_rho020"],"updated_at":datetime.now(timezone.utc).isoformat()}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
}

current=waiting_for_durc
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  rmdir "$LOCK" 2>/dev/null || true
  ((rc == 0)) || write_state failed "$current" "$rc" || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo $$ > "$STATE/launcher.pid"
write_state running "$current"

while true; do
  if [[ -f "$DURC_STATE" ]]; then
    durc_status=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$DURC_STATE")
    case "$durc_status" in
      complete|completed|success|failed|error)
        durc_active=$(ps -eo args= | grep -E '[r]un_durc_ablation_parallel\.sh|[t]rain_durc_worker\.py' || true)
        if [[ -z "$durc_active" ]]; then
          printf '%s\n' "$durc_status" > "$STATE/durc_terminal_status.txt"
          break
        fi
        ;;
    esac
  fi
  sleep 60
done

current=activate_staged_medcra
write_state running "$current"
test -f "$STAGING/files.list"
test -f "$STAGING/manifest.sha256"
(cd "$STAGING" && sha256sum -c manifest.sha256)
backup="$ROOT/.codex_backups/medcra_m0_m7_activate_$(date -u +%Y%m%dT%H%M%SZ)"
while IFS= read -r relative; do
  [[ -n "$relative" && "$relative" != /* && "$relative" != *..* ]] || exit 64
  mkdir -p "$backup/$(dirname "$relative")" "$(dirname "$ROOT/$relative")"
  test ! -e "$ROOT/$relative" || cp -p "$ROOT/$relative" "$backup/$relative"
  cp -p "$STAGING/$relative" "$ROOT/$relative"
done < "$STAGING/files.list"
printf '%s\n' "$backup" > "$STATE/activation_backup.txt"

current=preflight
write_state running "$current"
"$PY" -m compileall -q ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py tests/test_medcra_up.py tools/train_medcra_worker.py tools/collect_medcra_ablation.py
"$PY" -m pytest -q tests/test_medcra_up.py -k "parameters or nearest or constraints_disabled or moment or energy or gradient" > "$STATE/mechanism_${RUN_ID}.log" 2>&1
"$PY" -m pytest -q tests/test_medcra_up.py -k "yaml or topology or full_model or cuda" > "$STATE/interface_${RUN_ID}.log" 2>&1

prepare_cache() {
  "$PY" "$ROOT/tools/prepare_dcra_dataset_cache.py" --data "$DATA" --imgsz 640 --batch 16 > "$STATE/cache_${RUN_ID}.log" 2>&1
}
test_one() {
  local name=$1
  CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/test.py" --weights "$TRAIN/$name/weights/best.pt" --data "$DATA" --name "$name" --device 0 --batch 16 --workers 2 --imgsz 640 > "$TEST/${name}.log" 2>&1
  test -s "$TEST/$name/summary_metrics.json"
  test -s "$TEST/$name/scale_ap_metrics.json"
}

for stage in "${STAGES[@]}"; do
  current=${stage}.seeds0_1_2.train
  write_state running "$current"
  prepare_cache
  pids=()
  for seed in 0 1 2; do
    name=medcra_${RUN_ID}_${stage}_seed${seed}
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_medcra_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 > "$TRAIN/${name}.log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.train.pid"
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0))
  current=${stage}.seeds0_1_2.test
  write_state running "$current"
  pids=()
  for seed in 0 1 2; do
    name=medcra_${RUN_ID}_${stage}_seed${seed}
    test_one "$name" &
    pids+=("$!")
    echo "$!" > "$STATE/${stage}_seed${seed}.test.pid"
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0))
  completed=()
  for candidate in "${STAGES[@]}"; do
    completed+=("$candidate")
    [[ "$candidate" == "$stage" ]] && break
  done
  "$PY" "$ROOT/tools/collect_medcra_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${completed[@]}"
done

current=complete
write_state complete "$current"
echo "ME-DCRA M0--M7 chain completed: run_id=$RUN_ID"
