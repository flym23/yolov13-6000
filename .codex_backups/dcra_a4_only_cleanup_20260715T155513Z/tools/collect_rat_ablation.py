#!/usr/bin/env python3
"""Collect RATunnel three-seed test metrics into per-stage and combined summaries."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


METRICS = {"P": ("metrics", "metrics/precision(B)", 100.0), "R": ("metrics", "metrics/recall(B)", 100.0), "mAP50": ("metrics", "metrics/mAP50(B)", 100.0), "mAP75": ("metrics", "metrics/mAP75(B)", 100.0), "mAP50-95": ("metrics", "metrics/mAP50-95(B)", 100.0), "APS": ("scale_metrics_percent", "APS", 1.0), "APM": ("scale_metrics_percent", "APM", 1.0), "APL": ("scale_metrics_percent", "APL", 1.0)}
CONFIGS = {
    "t1_rat_initial": ("yolov13_rat_initial.yaml", "T1: initial three HyperACE distribution nodes use RATunnel"),
    "t4_rat_no_amplitude": ("yolov13_rat_no_amplitude.yaml", "T4: T1 without amplitude alignment"),
    "t5_rat_channel_only": ("yolov13_rat_channel_only.yaml", "T5: T1 channel reliability only"),
    "t6_rat_amplitude_only": ("yolov13_rat_amplitude_only.yaml", "T6: T1 amplitude alignment plus bounded scalar only"),
    "t2_rat_late": ("yolov13_rat_late.yaml", "T2: later four repeated injection nodes use RATunnel"),
    "t3_rat_all": ("yolov13_rat_all.yaml", "T3: all seven FullPAD nodes use RATunnel"),
    "t7_faar_rat_initial": ("yolov13_faar_rat_initial.yaml", "T7: B3 dual FAARUp plus initial three RATunnel nodes"),
}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--stages", nargs="+", required=True)
    args = parser.parse_args(); root = args.root.resolve(); test = root / "runs/test"; rows = []; payloads = {}
    for stage in args.stages:
        yaml_name, structure = CONFIGS[stage]; stage_rows = []
        for seed in range(3):
            path = test / f"rat_{args.run_id}_{stage}_seed{seed}" / "summary_metrics.json"
            data = json.loads(path.read_text(encoding="utf-8")); row = {"stage": stage, "seed": seed, "yaml": yaml_name, "structure": structure, "summary_path": str(path)}
            for metric, (section, key, scale) in METRICS.items(): row[metric] = float(data[section][key]) * scale
            stage_rows.append(row)
        rows.extend(stage_rows)
        detail = {m: {"values": [r[m] for r in stage_rows], "mean": statistics.fmean(r[m] for r in stage_rows), "std": statistics.stdev(r[m] for r in stage_rows), "best": max(r[m] for r in stage_rows), "worst": min(r[m] for r in stage_rows)} for m in METRICS}
        payloads[stage] = {"run_id": args.run_id, "stage": stage, "config": {"yaml": yaml_name, "structure": structure}, "rows_percent": stage_rows, "detail": detail}
        (test / f"rat_{args.run_id}_{stage}_summary.json").write_text(json.dumps(payloads[stage], indent=2), encoding="utf-8")
    fields = ["stage", "seed", "yaml", "structure", *METRICS, "summary_path"]
    with (test / f"rat_{args.run_id}_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (test / f"rat_{args.run_id}_summary.json").write_text(json.dumps({"run_id": args.run_id, "stage_order": args.stages, "summaries": payloads, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
