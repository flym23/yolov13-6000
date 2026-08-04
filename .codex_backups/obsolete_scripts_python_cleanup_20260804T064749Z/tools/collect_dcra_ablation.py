#!/usr/bin/env python3
"""Collect DCRA results with the exact D0 three-seed stage-summary schema."""

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
    "baseline_original": {
        "yaml": "yolov13-original.yaml",
        "structure": "Original YOLOv13: HyperACE + original nearest P5->P4/P4->P3 upsampling",
    },
    "a4_tau020": {
        "yaml": "yolov13-dcra-tau020.yaml",
        "structure": "A1 with softmax temperature 0.10->0.20; all other settings unchanged",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGE_CONFIGS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--prefix", default="dcra", help="Run-directory and result-file prefix.")
    return parser.parse_args()


def load_seed_metrics(root, run_id, stage, seed, prefix):
    path = root / "runs/test" / f"{prefix}_{run_id}_{stage}_seed{seed}" / "summary_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "stage": stage,
        "seed": seed,
        "yaml": STAGE_CONFIGS[stage]["yaml"],
        "structure": STAGE_CONFIGS[stage]["structure"],
        "summary_path": str(path),
    }
    for name, (section, key, scale) in METRIC_KEYS.items():
        row[name] = float(data[section][key]) * scale
    return row


def summarize(rows):
    output = {}
    for key in METRIC_KEYS:
        values = [row[key] for row in rows]
        output[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
            "best": max(values),
            "worst": min(values),
        }
    return output


def main():
    args = parse_args()
    root = args.root.resolve()
    test_dir = root / "runs/test"
    all_rows = []
    stage_payloads = {}
    for stage in args.stages:
        rows = [load_seed_metrics(root, args.run_id, stage, seed, args.prefix) for seed in args.seeds]
        all_rows.extend(rows)
        detail = summarize(rows)
        payload = {
            "run_id": args.run_id,
            "stage": stage,
            "config": STAGE_CONFIGS[stage],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "seeds": args.seeds,
            "rows_percent": rows,
            "mean": {key: value["mean"] for key, value in detail.items()},
            "std": {key: value["std"] for key, value in detail.items()},
            "detail": detail,
        }
        stage_payloads[stage] = payload
        (test_dir / f"{args.prefix}_{args.run_id}_{stage}_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    csv_path = test_dir / f"{args.prefix}_{args.run_id}_ablation.csv"
    fieldnames = ["stage", "seed", "yaml", "structure", *METRIC_KEYS.keys(), "summary_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    combined = {
        "run_id": args.run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage_order": args.stages,
        "csv": str(csv_path),
        "summaries": stage_payloads,
    }
    (test_dir / f"{args.prefix}_{args.run_id}_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print(json.dumps({"csv": str(csv_path), "stages": args.stages, "seeds": args.seeds}, indent=2))


if __name__ == "__main__":
    main()
