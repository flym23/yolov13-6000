#!/usr/bin/env python3
"""Aggregate D1--D4 DGMR results, paired statistics, and reused historical baselines."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dgmr_lcer_dcra_experiments import REFERENCE_BASELINES, STAGE_ORDER, STRUCTURES  # noqa: E402


METRICS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_METRICS = ("APS", "APM", "APL")
ALL_METRICS = (*METRICS, *SCALE_METRICS)
CONTROL_STAGE = "d1_matched_endpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, required=True)
    parser.add_argument("--reference-baseline", type=Path, action="append", default=None)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def paired_statistics(control: list[dict], candidate: list[dict]) -> dict[str, dict]:
    control_by_seed = {row["seed"]: row for row in control}
    candidate_by_seed = {row["seed"]: row for row in candidate}
    if set(control_by_seed) != {0, 1, 2} or set(candidate_by_seed) != {0, 1, 2}:
        raise ValueError("paired DGMR analysis requires exactly seeds 0, 1, and 2 for every compared stage")
    result = {}
    for metric in ALL_METRICS:
        differences = [candidate_by_seed[seed][metric] - control_by_seed[seed][metric] for seed in range(3)]
        result[metric] = {
            "per_seed": differences,
            "mean": statistics.mean(differences),
            "std": statistics.stdev(differences),
            "wins": sum(value > 0.0 for value in differences),
            "ties": sum(value == 0.0 for value in differences),
            "losses": sum(value < 0.0 for value in differences),
        }
    return result


def load_reference(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 3 or {int(row["seed"]) for row in runs} != {0, 1, 2}:
        raise ValueError(f"{path}: reference baseline must contain exactly seeds 0, 1, and 2")
    metrics = payload.get("metrics_percent", {})
    if set(ALL_METRICS) - set(metrics):
        raise KeyError(f"{path}: missing required reference metrics")
    return {
        "source": str(path),
        "stage": runs[0].get("stage", path.stem),
        "structure": payload.get("structure", "unknown"),
        "n": 3,
        "runs": runs,
        "metrics_percent": {metric: metrics[metric] for metric in ALL_METRICS},
        "paired_with_dgmr": False,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = []
    for stage in args.stages:
        for seed in range(3):
            test_name = f"{args.run_id}_{stage}_seed{seed}"
            path = root / "runs" / "test" / test_name / "summary_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            row = {
                "stage": stage,
                "seed": seed,
                "test_name": test_name,
                "structure": STRUCTURES[stage],
                "weights": payload["weights"],
                "FPS": None,
                "PeakMemoryGB": None,
            }
            for name, key in METRICS.items():
                if key not in metrics:
                    raise KeyError(f"{path}: missing required metric {key}")
                row[name] = float(metrics[key]) * 100.0
            for name in SCALE_METRICS:
                row[name] = float(payload.get("scale_metrics_percent", {}).get(name, 0.0))
            model = payload.get("model", {})
            row["Params"] = int(model.get("parameters", 0))
            row["GFLOPs"] = float(model.get("gflops", 0.0))
            records.append(row)

    reference_paths = args.reference_baseline or [Path(path) for path in REFERENCE_BASELINES]
    references = [load_reference(path) for path in reference_paths]
    output_dir = root / "runs" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": args.run_id,
        "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
        "settings": {"epochs": 300, "patience": 40, "workers": 2, "amp": False, "plots": False},
        "comparison_rule": (
            "D1--D4 use same-batch seed-matched paired comparisons. Reused LCER L0 and SPC P0 summaries are "
            "unpaired historical references only; D0 is intentionally not retrained under the user's reuse instruction."
        ),
        "reference_baselines": references,
        "unmeasured": {"FPS": True, "PeakMemoryGB": True},
        "stages": {},
    }
    rows_by_stage = {stage: [row for row in records if row["stage"] == stage] for stage in args.stages}
    for stage, stage_rows in rows_by_stage.items():
        stage_summary = {
            "structure": STRUCTURES[stage],
            "n": len(stage_rows),
            "runs": stage_rows,
            "metrics_percent": {name: mean_std([float(row[name]) for row in stage_rows]) for name in ALL_METRICS},
            "Params": stage_rows[0]["Params"],
            "GFLOPs": stage_rows[0]["GFLOPs"],
            "FPS": None,
            "PeakMemoryGB": None,
        }
        if stage != CONTROL_STAGE and CONTROL_STAGE in rows_by_stage:
            stage_summary["paired_vs_d1"] = paired_statistics(rows_by_stage[CONTROL_STAGE], stage_rows)
        if stage == "d4_dgmr_dual":
            for endpoint in ("d2_global_endpoint", "d3_local_endpoint"):
                if endpoint in rows_by_stage:
                    stage_summary[f"paired_vs_{endpoint[:2]}"] = paired_statistics(rows_by_stage[endpoint], stage_rows)
        summary["stages"][stage] = stage_summary
        (output_dir / f"{args.run_id}_{stage}_summary.json").write_text(
            json.dumps(stage_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    fields = ["stage", "seed", *ALL_METRICS, "Params", "GFLOPs", "FPS", "PeakMemoryGB"]
    csv_path = output_dir / f"{args.run_id}_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)
    summary["csv"] = str(csv_path)
    overview_path = output_dir / f"{args.run_id}_summary.json"
    overview_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overview_path)


if __name__ == "__main__":
    main()
