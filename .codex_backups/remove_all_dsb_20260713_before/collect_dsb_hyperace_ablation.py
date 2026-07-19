#!/usr/bin/env python3
"""Collect DSB-HyperACE three-seed validation metrics into stage and combined summaries."""

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
    "k1_dualnorm": ("yolov13_dsb_hyperace_dualnorm.yaml", "K1: branch2 dense dual normalization, no top-k"),
    "k2_topk4": ("yolov13_dsb_hyperace.yaml", "K2: branch2 dense V->E and top-4 sparse E->V"),
    "k3_both": ("yolov13_dsb_hyperace_both.yaml", "K3: both high-order branches use top-4 DSB"),
    "k4_topk2": ("yolov13_dsb_hyperace_topk2.yaml", "K4: K2 with top-k=2"),
    "k5_topk6": ("yolov13_dsb_hyperace_topk6.yaml", "K5: K2 with top-k=6"),
    "k6_faar_dsb": ("yolov13_faar_dsb_hyperace.yaml", "K6: B3 dual FAARUp plus K2 DSB-HyperACE"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    test_root = root / "runs/test"
    rows, payloads = [], {}
    for stage in args.stages:
        yaml_name, structure = CONFIGS[stage]
        stage_rows = []
        for seed in range(3):
            path = test_root / f"dsb_{args.run_id}_{stage}_seed{seed}" / "summary_metrics.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            row = {"stage": stage, "seed": seed, "yaml": yaml_name, "structure": structure, "summary_path": str(path)}
            for metric, (section, key, scale) in METRICS.items():
                row[metric] = float(data[section][key]) * scale
            stage_rows.append(row)
        rows.extend(stage_rows)
        detail = {
            metric: {
                "values": [row[metric] for row in stage_rows],
                "mean": statistics.fmean(row[metric] for row in stage_rows),
                "std": statistics.stdev(row[metric] for row in stage_rows),
                "best": max(row[metric] for row in stage_rows),
                "worst": min(row[metric] for row in stage_rows),
            }
            for metric in METRICS
        }
        payloads[stage] = {
            "run_id": args.run_id,
            "stage": stage,
            "config": {"yaml": yaml_name, "structure": structure},
            "rows_percent": stage_rows,
            "detail": detail,
        }
        (test_root / f"dsb_{args.run_id}_{stage}_summary.json").write_text(
            json.dumps(payloads[stage], indent=2), encoding="utf-8"
        )

    fields = ["stage", "seed", "yaml", "structure", *METRICS, "summary_path"]
    with (test_root / f"dsb_{args.run_id}_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (test_root / f"dsb_{args.run_id}_summary.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "stage_order": args.stages,
                "summaries": payloads,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

