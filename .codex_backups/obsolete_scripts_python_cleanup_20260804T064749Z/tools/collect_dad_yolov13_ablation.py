#!/usr/bin/env python3
"""Aggregate DAD-YOLOv13 seed summaries into stage and full-factorial overviews."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


STAGES = {
    "a1": {"label": "A1", "yaml": "yolov13-dad-a1-ddfc.yaml", "DDFCalib": True, "TIERDCRA": False, "SDDCHead": False},
    "a2": {"label": "A2", "yaml": "yolov13-dad-a2-tier.yaml", "DDFCalib": False, "TIERDCRA": True, "SDDCHead": False},
    "a3": {"label": "A3", "yaml": "yolov13-dad-a3-sddc.yaml", "DDFCalib": False, "TIERDCRA": False, "SDDCHead": True},
    "a4": {"label": "A4", "yaml": "yolov13-dad-a4-ddfc-tier.yaml", "DDFCalib": True, "TIERDCRA": True, "SDDCHead": False},
    "a5": {"label": "A5", "yaml": "yolov13-dad-a5-tier-sddc.yaml", "DDFCalib": False, "TIERDCRA": True, "SDDCHead": True},
    "a6": {"label": "A6", "yaml": "yolov13-dad-a6-ddfc-sddc.yaml", "DDFCalib": True, "TIERDCRA": False, "SDDCHead": True},
    "a7": {"label": "A7", "yaml": "yolov13-dad-a7-full.yaml", "DDFCalib": True, "TIERDCRA": True, "SDDCHead": True},
}
METRIC_KEYS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_KEYS = ("APS", "APM", "APL")
BASELINES = {
    "LCER-L0 original YOLOv13": "runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "SPC-P0 original YOLOv13": "runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(payload: dict, key: str) -> float:
    value = payload.get("metrics", {}).get(METRIC_KEYS[key])
    if value is None:
        raise KeyError(f"Missing metric {METRIC_KEYS[key]} in validation summary.")
    return float(value) * 100.0


def _seed_payload(root: Path, run_id: str, stage: str, seed: int) -> dict:
    name = f"{run_id}_{stage}_seed{seed}"
    summary_path = root / "runs" / "test" / name / "summary_metrics.json"
    train_manifest = root / "runs" / "train" / f"{name}.train.json"
    if not summary_path.is_file() or not train_manifest.is_file():
        raise FileNotFoundError(f"Incomplete DAD seed artifact for {name}: {summary_path} / {train_manifest}")
    summary = _load_json(summary_path)
    values = {key: _metric_value(summary, key) for key in METRIC_KEYS}
    scale = summary.get("scale_metrics_percent", {})
    values.update({key: float(scale[key]) for key in SCALE_KEYS})
    return {
        "seed": seed,
        "name": name,
        "training_manifest": str(train_manifest),
        "validation_summary": str(summary_path),
        "metrics_percent": values,
    }


def _aggregate(seeds: list[dict]) -> dict:
    output = {}
    for key in (*METRIC_KEYS, *SCALE_KEYS):
        values = [seed["metrics_percent"][key] for seed in seeds]
        output[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return output


def _paired_delta(left: dict, right: dict, label: str) -> dict:
    output = {"contrast": label, "metrics_percent": {}}
    for key in (*METRIC_KEYS, *SCALE_KEYS):
        values = [
            left["seeds"][index]["metrics_percent"][key] - right["seeds"][index]["metrics_percent"][key]
            for index in range(3)
        ]
        output["metrics_percent"][key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
            "seed_deltas": values,
            "positive_seeds": sum(value > 0.0 for value in values),
        }
    return output


def _baseline_reference(root: Path) -> dict:
    references = {}
    for label, relative_path in BASELINES.items():
        path = root / relative_path
        references[label] = {"path": str(path), "available": path.is_file()}
        if path.is_file():
            payload = _load_json(path)
            references[label]["payload"] = payload
    return references


def _write_csv(path: Path, stages: dict) -> None:
    fields = ["group", "seed", "DDFCalib", "TIERDCRA", "SDDCHead", *METRIC_KEYS, *SCALE_KEYS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in stages.values():
            structure = stage["structure"]
            for seed in stage["seeds"]:
                writer.writerow(
                    {
                        "group": stage["label"],
                        "seed": seed["seed"],
                        **{key: structure[key] for key in ("DDFCalib", "TIERDCRA", "SDDCHead")},
                        **seed["metrics_percent"],
                    }
                )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    test_dir = root / "runs" / "test"
    stage_results = {}
    for stage in args.stages:
        seeds = [_seed_payload(root, args.run_id, stage, seed) for seed in range(3)]
        config = STAGES[stage]
        stage_results[stage] = {
            "label": config["label"],
            "yaml": str(root / "ultralytics" / "cfg" / "models" / "v13" / config["yaml"]),
            "structure": {key: config[key] for key in ("DDFCalib", "TIERDCRA", "SDDCHead")},
            "seeds": seeds,
            "aggregate_percent": _aggregate(seeds),
        }
        stage_path = test_dir / f"{args.run_id}_{stage}_summary.json"
        stage_path.write_text(json.dumps(stage_results[stage], ensure_ascii=False, indent=2), encoding="utf-8")

    contrasts = []
    for left, right, label in (
        ("a4", "a1", "A4-A1: TIER-DCRA conditional on DDFCalib"),
        ("a4", "a2", "A4-A2: DDFCalib conditional on TIER-DCRA"),
        ("a5", "a2", "A5-A2: SDDC head conditional on TIER-DCRA"),
        ("a5", "a3", "A5-A3: TIER-DCRA conditional on SDDC head"),
        ("a6", "a1", "A6-A1: SDDC head conditional on DDFCalib"),
        ("a6", "a3", "A6-A3: DDFCalib conditional on SDDC head"),
        ("a7", "a4", "A7-A4: SDDC head in full combination"),
        ("a7", "a5", "A7-A5: DDFCalib in full combination"),
        ("a7", "a6", "A7-A6: TIER-DCRA in full combination"),
    ):
        if left in stage_results and right in stage_results:
            contrasts.append(_paired_delta(stage_results[left], stage_results[right], label))

    overview = {
        "run_id": args.run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
            "epochs": 300,
            "patience": 40,
            "workers": 2,
            "amp": False,
            "plots": False,
            "parallel_seed_processes": 3,
            "baseline_policy": "A0 is not retrained; designated LCER-L0 and SPC-P0 original-YOLOv13 summaries are reused as unpaired references.",
        },
        "baseline_references": _baseline_reference(root),
        "stages": stage_results,
        "paired_factorial_contrasts": contrasts,
    }
    overview_path = test_dir / f"{args.run_id}_summary.json"
    overview_path.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(test_dir / f"{args.run_id}_ablation.csv", stage_results)
    print(overview_path)


if __name__ == "__main__":
    main()
