#!/usr/bin/env python3
"""Train one deterministic TDR-YOLOv13 seed on URPC2020half."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


STAGES = {
    "t1": ("yolov13n-tdr-t1-tier.yaml", "T1 / TIER-DCRA"),
    "t2": ("yolov13n-tdr-t2-sadi.yaml", "T2 / SADI-P3"),
    "t3": ("yolov13n-tdr-t3-brd.yaml", "T3 / BRD-Head"),
    "t4": ("yolov13n-tdr-t4-tier-sadi.yaml", "T4 / TIER-DCRA + SADI-P3"),
    "t5": ("yolov13n-tdr-t5-tier-brd.yaml", "T5 / TIER-DCRA + BRD-Head"),
    "t6": ("yolov13n-tdr-t6-sadi-brd.yaml", "T6 / SADI-P3 + BRD-Head"),
    "t7": ("yolov13n-tdr-t7-full.yaml", "T7 / full TDR-YOLOv13"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    yaml_name, structure = STAGES[args.stage]
    model_yaml = root / "ultralytics" / "cfg" / "models" / "v13" / yaml_name
    pretrained = root / "yolov13n.pt"
    train_dir = root / "runs" / "train" / args.name
    manifest = root / "runs" / "train" / f"{args.name}.train.json"
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "pretrained weights")):
        if not path.is_file():
            raise FileNotFoundError(f"TDR {label} unavailable: {path}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PIN_MEMORY"] = "false"
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import ultralytics

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"TDR worker resolved external Ultralytics: {ultralytics.__file__}") from error
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    model.load(str(pretrained))
    model.train(
        data=str(data), epochs=args.epochs, patience=args.patience, batch=16, imgsz=640, workers=2,
        amp=False, plots=False, deterministic=True, seed=args.seed, resume=False, device=0,
        project=str(root / "runs" / "train"), name=args.name, exist_ok=False,
    )
    best = train_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"TDR training finished without best.pt: {best}")
    payload = {
        "name": args.name, "stage": args.stage.upper(), "seed": args.seed, "structure": structure,
        "model_yaml": str(model_yaml), "dataset": str(data), "weights": str(best),
        "settings": {"epochs": args.epochs, "patience": args.patience, "batch": 16, "imgsz": 640,
                     "workers": 2, "amp": False, "plots": False, "deterministic": True,
                     "resume": False, "device": 0},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
