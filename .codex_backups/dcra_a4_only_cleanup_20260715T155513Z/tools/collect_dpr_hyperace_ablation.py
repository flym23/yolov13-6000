#!/usr/bin/env python3
"""Collect DPR-HyperACE three-seed test metrics into stage and combined summaries."""

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
    "h1_dpr_p3_spd": ("yolov13_dpr_hyperace.yaml", "H1: P3 SPD zero-start residual correction inside HyperACE"),
    "h4_dpr_dwdown": ("yolov13_dpr_hyperace_dwdown.yaml", "H4: H1 with stride-2 depthwise downsampling"),
    "h5_dpr_replace": ("yolov13_dpr_hyperace_replace.yaml", "H5: SPD directly replaces the P3 AvgPool path"),
    "h2_dpr_p5": ("yolov13_dpr_hyperace_p5.yaml", "H2: P5 semantic zero-start residual only"),
    "h3_dpr_dual": ("yolov13_dpr_hyperace_dual.yaml", "H3: P3 detail plus P5 semantic residuals"),
    "h6_faar_dpr": ("yolov13_faar_dpr_hyperace.yaml", "H6: B3 dual FAARUp plus H1 DPR-HyperACE"),
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
            path = test_root / f"dpr_{args.run_id}_{stage}_seed{seed}" / "summary_metrics.json"
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
        (test_root / f"dpr_{args.run_id}_{stage}_summary.json").write_text(
            json.dumps(payloads[stage], indent=2), encoding="utf-8"
        )
    fields = ["stage", "seed", "yaml", "structure", *METRICS, "summary_path"]
    with (test_root / f"dpr_{args.run_id}_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (test_root / f"dpr_{args.run_id}_summary.json").write_text(
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
