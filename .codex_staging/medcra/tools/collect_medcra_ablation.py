#!/usr/bin/env python3
"""Collect one fully retrained URPC2020 M0--M7 ME-DCRA experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


METRICS = {
    "P": ("metrics", "metrics/precision(B)", 100.0),
    "R": ("metrics", "metrics/recall(B)", 100.0),
    "mAP50": ("metrics", "metrics/mAP50(B)", 100.0),
    "mAP75": ("metrics", "metrics/mAP75(B)", 100.0),
    "mAP50-95": ("metrics", "metrics/mAP50-95(B)", 100.0),
    "APS": ("scale_metrics_percent", "APS", 1.0),
    "APM": ("scale_metrics_percent", "APM", 1.0),
    "APL": ("scale_metrics_percent", "APL", 1.0),
}
CONFIGS = {
    "m0_original": ("yolov13-original.yaml", "M0: original YOLOv13", None),
    "m1_a4": ("yolov13-dcra-tau020.yaml", "M1: A4 DCRA k=3, tau=0.20", None),
    "m2_full": ("yolov13-medcra.yaml", "M2: full ME-DCRA, rho=0.10", None),
    "m3_no_moment": ("yolov13-medcra-no-moment.yaml", "M3: M2 without moment preservation", None),
    "m4_no_center": ("yolov13-medcra-no-center.yaml", "M4: M2 without correction centering", None),
    "m5_no_bound": ("yolov13-medcra-no-bound.yaml", "M5: M2 without RMS energy bound", None),
    "m6_rho005": ("yolov13-medcra-rho005.yaml", "M6: M2 with rho=0.05", None),
    "m7_rho020": ("yolov13-medcra-rho020.yaml", "M7: M2 with rho=0.20", None),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(CONFIGS), required=True)
    return parser.parse_args()


def metric_row(root, run_id, stage, seed):
    yaml_name, structure, source_stage = CONFIGS[stage]
    assert source_stage is None, f"Unexpected external source stage: {source_stage}"
    path = root / "runs/test" / f"medcra_{run_id}_{stage}_seed{seed}" / "summary_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {"stage": stage, "seed": seed, "yaml": yaml_name, "structure": structure, "summary_path": str(path)}
    for name, (section, key, scale) in METRICS.items():
        row[name] = float(data[section][key]) * scale
    return row


def main():
    args = parse_args()
    root = args.root.resolve()
    rows, summaries = [], {}
    for stage in args.stages:
        stage_rows = [metric_row(root, args.run_id, stage, seed) for seed in (0, 1, 2)]
        rows.extend(stage_rows)
        detail = {}
        for metric in METRICS:
            values = [row[metric] for row in stage_rows]
            detail[metric] = {"mean": statistics.fmean(values), "std": statistics.stdev(values), "values": values, "best": max(values), "worst": min(values)}
        payload = {
            "run_id": args.run_id,
            "stage": stage,
            "config": {"yaml": CONFIGS[stage][0], "structure": CONFIGS[stage][1]},
            "dataset": "/home/room305/ZZF/URPC2020/data.yaml",
            "epochs": 300,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "seeds": [0, 1, 2],
            "rows_percent": stage_rows,
            "mean": {key: value["mean"] for key, value in detail.items()},
            "std": {key: value["std"] for key, value in detail.items()},
            "detail": detail,
        }
        summaries[stage] = payload
        (root / "runs/test" / f"medcra_{args.run_id}_{stage}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = root / "runs/test" / f"medcra_{args.run_id}_ablation.csv"
    columns = ["stage", "seed", "yaml", "structure", *METRICS, "summary_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result = {"run_id": args.run_id, "stage_order": args.stages, "csv": str(csv_path), "summaries": summaries}
    (root / "runs/test" / f"medcra_{args.run_id}_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
