#!/usr/bin/env python3
"""Collect FAAR validation summaries into JSON and CSV files."""

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
    "b3_scale_specific": {
        "yaml": "yolov13_faar.yaml",
        "structure": "FAARUp semantic for P5->P4, FAARUp detail for P4->P3, Detect(P3,P4,P5)",
    },
    "b2_shared": {
        "yaml": "yolov13_faar_shared.yaml",
        "structure": "FAARUp detail for both P5->P4 and P4->P3, Detect(P3,P4,P5)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def load_seed_metrics(root, run_id, stage, seed):
    path = root / "runs/test" / f"faar_{run_id}_{stage}_seed{seed}" / "summary_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "stage": stage,
        "seed": seed,
        "yaml": STAGE_CONFIGS.get(stage, {}).get("yaml", ""),
        "structure": STAGE_CONFIGS.get(stage, {}).get("structure", ""),
        "summary_path": str(path),
    }
    for name, (section, key, scale) in METRIC_KEYS.items():
        row[name] = float(data[section][key]) * scale
    return row


def summarize(rows):
    out = {}
    for key in METRIC_KEYS:
        vals = [row[key] for row in rows]
        out[key] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
            "best": max(vals),
            "worst": min(vals),
        }
    return out


def main():
    args = parse_args()
    root = args.root.resolve()
    test_dir = root / "runs/test"
    all_rows = []
    stage_payloads = {}
    for stage in args.stages:
        rows = [load_seed_metrics(root, args.run_id, stage, seed) for seed in args.seeds]
        all_rows.extend(rows)
        summary = summarize(rows)
        payload = {
            "run_id": args.run_id,
            "stage": stage,
            "config": STAGE_CONFIGS.get(stage, {}),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "seeds": args.seeds,
            "rows_percent": rows,
            "mean": {k: v["mean"] for k, v in summary.items()},
            "std": {k: v["std"] for k, v in summary.items()},
            "detail": summary,
        }
        stage_payloads[stage] = payload
        (test_dir / f"faar_{args.run_id}_{stage}_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    csv_path = test_dir / f"faar_{args.run_id}_ablation.csv"
    fieldnames = ["stage", "seed", "yaml", "structure", *METRIC_KEYS.keys(), "summary_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    combined = {
        "run_id": args.run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage_order": args.stages,
        "csv": str(csv_path),
        "summaries": stage_payloads,
    }
    (test_dir / f"faar_{args.run_id}_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "stages": args.stages}, indent=2))


if __name__ == "__main__":
    main()
