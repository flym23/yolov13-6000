#!/usr/bin/env python3
"""Aggregate TDR T1--T7 seeds and retain the two authorized historical baselines as references."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


STAGES = {
    "t1": ("T1", "yolov13n-tdr-t1-tier.yaml", True, False, False),
    "t2": ("T2", "yolov13n-tdr-t2-sadi.yaml", False, True, False),
    "t3": ("T3", "yolov13n-tdr-t3-brd.yaml", False, False, True),
    "t4": ("T4", "yolov13n-tdr-t4-tier-sadi.yaml", True, True, False),
    "t5": ("T5", "yolov13n-tdr-t5-tier-brd.yaml", True, False, True),
    "t6": ("T6", "yolov13n-tdr-t6-sadi-brd.yaml", False, True, True),
    "t7": ("T7", "yolov13n-tdr-t7-full.yaml", True, True, True),
}
METRIC_KEYS = {
    "P": "metrics/precision(B)", "R": "metrics/recall(B)", "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)", "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_KEYS = ("APS", "APM", "APL")
BASELINES = (
    "runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_result(root: Path, run_id: str, stage: str, seed: int) -> dict:
    name = f"{run_id}_{stage}_seed{seed}"
    summary_path = root / "runs" / "test" / name / "summary_metrics.json"
    manifest_path = root / "runs" / "train" / f"{name}.train.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete TDR seed artifact for {name}: {summary_path} / {manifest_path}")
    payload = load_json(summary_path)
    metrics = {key: float(payload["metrics"][source]) * 100.0 for key, source in METRIC_KEYS.items()}
    metrics.update({key: float(payload["scale_metrics_percent"][key]) for key in SCALE_KEYS})
    return {"seed": seed, "name": name, "training_manifest": str(manifest_path),
            "validation_summary": str(summary_path), "metrics_percent": metrics}


def aggregate(seeds: list[dict]) -> dict:
    return {
        key: {"mean": statistics.fmean(values := [seed["metrics_percent"][key] for seed in seeds]),
              "std": statistics.stdev(values), "min": min(values), "max": max(values)}
        for key in (*METRIC_KEYS, *SCALE_KEYS)
    }


def paired_delta(left: dict, right: dict, contrast: str) -> dict:
    output = {"contrast": contrast, "metrics_percent": {}}
    for key in (*METRIC_KEYS, *SCALE_KEYS):
        deltas = [left["seeds"][index]["metrics_percent"][key] - right["seeds"][index]["metrics_percent"][key]
                  for index in range(3)]
        output["metrics_percent"][key] = {"mean": statistics.fmean(deltas), "std": statistics.stdev(deltas),
                                           "seed_deltas": deltas, "positive_seeds": sum(value > 0.0 for value in deltas)}
    return output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = {}
    for stage in args.stages:
        label, yaml_name, tier, sadi, brd = STAGES[stage]
        seeds = [seed_result(root, args.run_id, stage, seed) for seed in range(3)]
        results[stage] = {"label": label, "yaml": str(root / "ultralytics/cfg/models/v13" / yaml_name),
                          "structure": {"TIERDCRA": tier, "SADIP3": sadi, "BRDHead": brd},
                          "seeds": seeds, "aggregate_percent": aggregate(seeds)}
        (root / "runs" / "test" / f"{args.run_id}_{stage}_summary.json").write_text(
            json.dumps(results[stage], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    contrasts = []
    for left, right, label in (("t4", "t2", "TIER conditional on SADI"), ("t5", "t3", "TIER conditional on BRD"),
                               ("t6", "t3", "SADI conditional on BRD"), ("t7", "t4", "BRD added to TIER+SADI"),
                               ("t7", "t5", "SADI added to TIER+BRD"), ("t7", "t6", "TIER added to SADI+BRD")):
        if left in results and right in results:
            contrasts.append(paired_delta(results[left], results[right], label))
    references = []
    for relative in BASELINES:
        path = root / relative
        references.append({"path": str(path), "available": path.is_file(),
                           "payload": load_json(path) if path.is_file() else None})
    overview = {
        "run_id": args.run_id, "updated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"dataset": "/home/room305/ZZF/URPC2020half/data.yaml", "epochs": 300, "patience": 40,
                     "workers": 2, "amp": False, "deterministic": True, "parallel_seed_processes": 3,
                     "baseline_policy": "Reuse the two user-authorized historical L0/P0 summaries as references."},
        "baseline_references": references, "stages": results, "paired_factorial_contrasts": contrasts,
    }
    (root / "runs" / "test" / f"{args.run_id}_summary.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = ["group", "seed", "TIERDCRA", "SADIP3", "BRDHead", *METRIC_KEYS, *SCALE_KEYS]
    with (root / "runs" / "test" / f"{args.run_id}_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results.values():
            for seed in result["seeds"]:
                writer.writerow({"group": result["label"], "seed": seed["seed"], **result["structure"], **seed["metrics_percent"]})
    print(root / "runs" / "test" / f"{args.run_id}_summary.json")


if __name__ == "__main__":
    main()
