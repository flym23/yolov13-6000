#!/usr/bin/env python3
"""Collect three-seed FAARUp/LGARUp validation summaries."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


METRICS = {"P": ("metrics", "metrics/precision(B)", 100.0), "R": ("metrics", "metrics/recall(B)", 100.0), "mAP50": ("metrics", "metrics/mAP50(B)", 100.0), "mAP75": ("metrics", "metrics/mAP75(B)", 100.0), "mAP50-95": ("metrics", "metrics/mAP50-95(B)", 100.0), "APS": ("scale_metrics_percent", "APS", 1.0), "APM": ("scale_metrics_percent", "APM", 1.0), "APL": ("scale_metrics_percent", "APL", 1.0)}
CONFIGS = {"u1_faar_p4_only": ("yolov13_faar_p4_only.yaml", "U1: semantic FAARUp at P5->P4 only"), "u2_faar_p3_only": ("yolov13_faar_p3_only.yaml", "U2: detail FAARUp at P4->P3 only"), "g1_lgar_p4": ("yolov13_lgar_p4.yaml", "G1: semantic LGARUp at P5->P4 only"), "g2_lgar_p3": ("yolov13_lgar_p3.yaml", "G2: detail LGARUp at P4->P3 only"), "g3_lgar": ("yolov13_lgar.yaml", "G3: LGARUp at both top-down nodes"), "g4_lgar_p4_no_lateral": ("yolov13_lgar_p4_no_lateral.yaml", "G4: G1 without lateral guidance"), "g5_lgar_p4_no_offset": ("yolov13_lgar_p4_no_offset.yaml", "G5: G1 without learned offset"), "g6_lgar_p4_no_confidence": ("yolov13_lgar_p4_no_confidence.yaml", "G6: G1 without confidence")}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--stages", nargs="+", required=True)
    args = parser.parse_args(); root = args.root.resolve(); test = root / "runs/test"; rows = []; payloads = {}
    for stage in args.stages:
        yaml_name, structure = CONFIGS[stage]; stage_rows = []
        for seed in range(3):
            path = test / f"lgar_{args.run_id}_{stage}_seed{seed}" / "summary_metrics.json"; data = json.loads(path.read_text(encoding="utf-8")); row = {"stage": stage, "seed": seed, "yaml": yaml_name, "structure": structure, "summary_path": str(path)}
            for metric, (section, key, scale) in METRICS.items(): row[metric] = float(data[section][key]) * scale
            stage_rows.append(row)
        rows.extend(stage_rows); detail = {m: {"values": [r[m] for r in stage_rows], "mean": statistics.fmean(r[m] for r in stage_rows), "std": statistics.stdev(r[m] for r in stage_rows)} for m in METRICS}
        payloads[stage] = {"run_id": args.run_id, "stage": stage, "config": {"yaml": yaml_name, "structure": structure}, "rows_percent": stage_rows, "detail": detail}
        (test / f"lgar_{args.run_id}_{stage}_summary.json").write_text(json.dumps(payloads[stage], indent=2), encoding="utf-8")
    fields = ["stage", "seed", "yaml", "structure", *METRICS, "summary_path"]
    with (test / f"lgar_{args.run_id}_ablation.csv").open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (test / f"lgar_{args.run_id}_summary.json").write_text(json.dumps({"run_id": args.run_id, "stage_order": args.stages, "summaries": payloads, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
