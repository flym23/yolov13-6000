#!/usr/bin/env python3
"""Collect AG-DSB-HyperACE three-seed validation and gate metrics."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch


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
    "a1_dense_fixed": ("yolov13_ag_dsb_dense_fixed.yaml", "A1: branch2 dense dual norm, fixed eta=0.05"),
    "a2_dense": ("yolov13_ag_dsb_dense.yaml", "A2: A1 with learnable active eta"),
    "a3_topk2": ("yolov13_ag_dsb_topk2.yaml", "A3: A2 with E->V top-k=2"),
    "a4_topk3": ("yolov13_ag_dsb_topk3.yaml", "A4: A2 with E->V top-k=3"),
    "a5_topk2_no_norm": ("yolov13_ag_dsb_topk2_no_norm.yaml", "A5: A3 without projected-delta RMS normalization"),
    "a6_topk2_both": ("yolov13_ag_dsb_topk2_both.yaml", "A6: both high-order branches use A3 AG-DSB"),
}


def checkpoint_gates(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise RuntimeError(f"Checkpoint contains no model: {path}")
    records = []
    for name, module in model.named_modules():
        if hasattr(module, "effective_eta"):
            eta_tensor = module.effective_eta().detach().float().cpu().reshape(-1)
            eta = eta_tensor.tolist()
            eta_init = float(getattr(module, "eta_init", 0.05))
            move = (eta_tensor - eta_init).abs()
            raw = (
                module.eta_head_bias.detach().float().cpu().reshape(-1).tolist()
                if hasattr(module, "eta_head_bias")
                else None
            )
            records.append(
                {
                    "name": name,
                    "effective_eta": eta,
                    "eta_init": eta_init,
                    "eta_delta": (eta_tensor - eta_init).tolist(),
                    "mean_abs_eta_move": float(move.mean()),
                    "max_abs_eta_move": float(move.max()),
                    "moved_head_count": int((move >= 1e-6).sum()),
                    "eta_head_bias": raw,
                    "gate_mode": getattr(module, "gate_mode", None),
                    "num_heads": int(module.num_heads),
                    "topk": int(module.topk),
                    "num_hyperedges": int(module.num_hyperedges),
                }
            )
    return records


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
            name = f"agdsb_{args.run_id}_{stage}_seed{seed}"
            summary_path = test_root / name / "summary_metrics.json"
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            gates = checkpoint_gates(root / "runs/train" / name / "weights/best.pt")
            row = {
                "stage": stage,
                "seed": seed,
                "yaml": yaml_name,
                "structure": structure,
                "gates": gates,
                "summary_path": str(summary_path),
            }
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
        (test_root / f"agdsb_{args.run_id}_{stage}_summary.json").write_text(
            json.dumps(payloads[stage], indent=2), encoding="utf-8"
        )

    fields = ["stage", "seed", "yaml", "structure", *METRICS, "gates", "summary_path"]
    with (test_root / f"agdsb_{args.run_id}_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, "gates": json.dumps(row["gates"])} for row in rows)
    (test_root / f"agdsb_{args.run_id}_summary.json").write_text(
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
