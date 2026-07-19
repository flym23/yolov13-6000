#!/usr/bin/env python3
"""Gate A4 execution: run A4 only when A3 passes the Original-YOLOv13 thresholds."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


THRESHOLDS = {
    "mAP75": 1.0,
    "mAP50-95": 0.8,
    "APS": 1.3,
    "APM": -0.2,
    "APL": -0.2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--max-map5095-std", type=float, default=0.50)
    return parser.parse_args()


def load_summary(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "mean" in data:
        return data
    metrics = data.get("metrics", {})
    scale = data.get("scale_metrics_percent", {})
    return {
        "mean": {
            "P": float(metrics["metrics/precision(B)"]) * 100.0,
            "R": float(metrics["metrics/recall(B)"]) * 100.0,
            "mAP50": float(metrics["metrics/mAP50(B)"]) * 100.0,
            "mAP75": float(metrics["metrics/mAP75(B)"]) * 100.0,
            "mAP50-95": float(metrics["metrics/mAP50-95(B)"]) * 100.0,
            "APS": float(scale["APS"]),
            "APM": float(scale["APM"]),
            "APL": float(scale["APL"]),
        },
        "std": {},
    }


def find_baseline(root, cli_path):
    candidates = []
    if cli_path is not None:
        candidates.append(cli_path)
    if os.getenv("UFCR_A0_SUMMARY"):
        candidates.append(Path(os.environ["UFCR_A0_SUMMARY"]))
    candidates.extend(
        [
            root / "runs/test/ufcr_a0_summary.json",
            root / "runs/test/a0_summary.json",
            root / "runs/test/original_summary.json",
            root / "runs/test/yolov13_original_summary.json",
        ]
    )
    for path in candidates:
        path = path if path.is_absolute() else root / path
        if path.exists():
            return path
    return None


def main():
    args = parse_args()
    root = args.root.resolve()
    state_dir = root / "runs/ufcr_ablation"
    state_dir.mkdir(parents=True, exist_ok=True)
    a3_path = root / "runs/test" / f"ufcr_{args.run_id}_a3_summary.json"
    baseline_path = find_baseline(root, args.baseline_summary)
    payload = {
        "run_id": args.run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "a3_summary": str(a3_path),
        "baseline_summary": str(baseline_path) if baseline_path else None,
        "thresholds_pp": THRESHOLDS,
        "max_mAP50_95_std": args.max_map5095_std,
        "a4_allowed": False,
        "reason": "",
    }
    if not a3_path.exists():
        payload["reason"] = f"missing A3 summary: {a3_path}"
        (state_dir / f"a4_gate_{args.run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)
    if baseline_path is None:
        payload["reason"] = "missing Original-YOLOv13/A0 baseline summary; A4 skipped to avoid ungated auxiliary training"
        (state_dir / f"a4_gate_{args.run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit(3)

    a3 = load_summary(a3_path)
    base = load_summary(baseline_path)
    delta = {key: a3["mean"][key] - base["mean"][key] for key in THRESHOLDS}
    checks = {key: delta[key] >= threshold for key, threshold in THRESHOLDS.items()}
    checks["mAP50-95_std"] = float(a3.get("std", {}).get("mAP50-95", 0.0)) <= args.max_map5095_std
    payload.update(
        {
            "baseline_mean_percent": base["mean"],
            "a3_mean_percent": a3["mean"],
            "a3_std_percent": a3.get("std", {}),
            "delta_pp": delta,
            "checks": checks,
            "a4_allowed": all(checks.values()),
        }
    )
    if not payload["a4_allowed"]:
        payload["reason"] = "A3 did not satisfy all SCI gate thresholds; A4 skipped"
    else:
        payload["reason"] = "A3 satisfied all SCI gate thresholds; A4 allowed"
    (state_dir / f"a4_gate_{args.run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["a4_allowed"] else 3)


if __name__ == "__main__":
    main()
