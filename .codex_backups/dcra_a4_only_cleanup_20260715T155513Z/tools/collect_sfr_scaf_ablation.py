#!/usr/bin/env python3
"""Collect SFR-SCAF validation metrics into reusable JSON and CSV summaries."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


METRIC_KEYS = {
    "P": ("metrics", "metrics/precision(B)", 100.0),
    "R": ("metrics", "metrics/recall(B)", 100.0),
    "mAP50": ("metrics", "metrics/mAP50(B)", 100.0),
    "mAP75": ("metrics", "metrics/mAP75(B)", 100.0),
    "mAP50-95": ("metrics", "metrics/mAP50-95(B)", 100.0),
    "APS": ("scale_metrics_percent", "APS", 1.0),
    "APM": ("scale_metrics_percent", "APM", 1.0),
    "APL": ("scale_metrics_percent", "APL", 1.0),
}

STAGE_CONFIGS = {
    "f3_sfr_scaf": ("yolov13_sfr_scaf.yaml", "SFR-SCAF full: semantic-filtered detail route + hybrid single P5->P4 fusion."),
    "f4_no_semantic_filter": ("yolov13_sfr_scaf_no_semantic_filter.yaml", "F3 without cross-scale semantic filtering: detail-only route."),
    "f5_fixed_route": ("yolov13_sfr_scaf_fixed_route.yaml", "F3 with fixed route=0.5."),
    "f6_consistency_only": ("yolov13_sfr_scaf_consistency_only.yaml", "S2-like low-amplitude consistency recalibration only."),
    "f7_semantic_only": ("yolov13_sfr_scaf_semantic_only.yaml", "R4-like positive semantic enhancement of P5-up only."),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    test_dir = root / "runs/test"
    rows, stage_payloads = [], {}
    for stage in args.stages:
        yaml_name, structure = STAGE_CONFIGS[stage]
        stage_rows = []
        for seed in args.seeds:
            path = test_dir / f"sfr_scaf_{args.run_id}_{stage}_seed{seed}" / "summary_metrics.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            row = {"stage": stage, "seed": seed, "yaml": yaml_name, "structure": structure, "summary_path": str(path)}
            for metric, (section, key, scale) in METRIC_KEYS.items():
                row[metric] = float(data[section][key]) * scale
            stage_rows.append(row)
        rows.extend(stage_rows)
        detail = {}
        for metric in METRIC_KEYS:
            values = [row[metric] for row in stage_rows]
            detail[metric] = {
                "values": values,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values),
                "best": max(values),
                "worst": min(values),
            }
        payload = {"run_id": args.run_id, "stage": stage, "config": {"yaml": yaml_name, "structure": structure}, "seeds": args.seeds, "rows_percent": stage_rows, "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat()}
        stage_payloads[stage] = payload
        (test_dir / f"sfr_scaf_{args.run_id}_{stage}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = ["stage", "seed", "yaml", "structure", *METRIC_KEYS, "summary_path"]
    csv_path = test_dir / f"sfr_scaf_{args.run_id}_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"run_id": args.run_id, "stage_order": args.stages, "summaries": stage_payloads, "csv": str(csv_path), "updated_at": datetime.now(timezone.utc).isoformat()}
    (test_dir / f"sfr_scaf_{args.run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "stages": args.stages}, indent=2))


if __name__ == "__main__":
    main()
