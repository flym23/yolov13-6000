#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rom305/zzf/yolov13-305
PY=/home/rom305/miniconda3/envs/yolov13/bin/python
DATA=$ROOT/data.yaml
PRETRAIN=$ROOT/yolov13n.pt

run_group() {
  local cfg=$1
  local tag=$2
  local cfg_path=$ROOT/ultralytics/cfg/models/v13/$cfg

  echo "========== launch $tag $(date) =========="
  for idx in 1 2 3; do
    local gpu=$((idx - 1))
    local name=${tag}_${idx}
    local log=$ROOT/runs/train/${name}.log
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES=$gpu WANDB_DISABLED=true "$PY" - <<PY
from ultralytics import YOLO

model = YOLO("$cfg_path")
model.load("$PRETRAIN")
model.train(
    data="$DATA",
    epochs=200,
    patience=40,
    batch=16,
    workers=8,
    amp=True,
    deterministic=False,
    plots=False,
    project="$ROOT/runs/train",
    name="$name",
)
PY
    ) >"$log" 2>&1 &
    echo $! > "$ROOT/runs/train/${name}.pid"
    echo "$name gpu=$gpu pid=$!"
  done

  wait "$(cat "$ROOT/runs/train/${tag}_1.pid")" "$(cat "$ROOT/runs/train/${tag}_2.pid")" "$(cat "$ROOT/runs/train/${tag}_3.pid")"
  echo "========== test $tag $(date) =========="
  for idx in 1 2 3; do
    local gpu=$((idx - 1))
    local name=${tag}_${idx}
    "$PY" "$ROOT/test.py" \
      --weights "$ROOT/runs/train/$name/weights/best.pt" \
      --name "$name" \
      --device "$gpu" \
      --batch 16 \
      --imgsz 640
  done
  "$PY" - <<PY
import json
from pathlib import Path

root = Path("$ROOT")
tag = "$tag"
runs = []
for idx in (1, 2, 3):
    path = root / "runs/test" / f"{tag}_{idx}" / "summary_metrics.json"
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    scale = data.get("scale_metrics_percent", {})
    runs.append({
        "P": metrics.get("metrics/precision(B)"),
        "R": metrics.get("metrics/recall(B)"),
        "mAP50": metrics.get("metrics/mAP50(B)"),
        "mAP75": metrics.get("metrics/mAP75(B)"),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)"),
        "APS": scale.get("APS"),
        "APM": scale.get("APM"),
        "APL": scale.get("APL"),
    })

mean = {k: sum(float(r[k]) for r in runs) / len(runs) for k in runs[0]}
out = {"runs": runs, "mean": mean}
(root / "runs/test" / f"{tag}_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY
}

mkdir -p "$ROOT/runs/train" "$ROOT/runs/test"
run_group yolov13-ducra-v7-udc.yaml ducra_v7_udc
run_group yolov13-ducra-v7-sirucra.yaml ducra_v7_sirucra
run_group yolov13-ducra-v7-rldhead.yaml ducra_v7_rldhead
