#!/usr/bin/env python3
"""Run one deterministic DAD-YOLOv13 seed and persist its immutable training manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


STAGES = {
    "a1": ("yolov13-dad-a1-ddfc.yaml", "DDFCalib only"),
    "a2": ("yolov13-dad-a2-tier.yaml", "TIER-DCRA only"),
    "a3": ("yolov13-dad-a3-sddc.yaml", "SDDC head only"),
    "a4": ("yolov13-dad-a4-ddfc-tier.yaml", "DDFCalib + TIER-DCRA"),
    "a5": ("yolov13-dad-a5-tier-sddc.yaml", "TIER-DCRA + SDDC head"),
    "a6": ("yolov13-dad-a6-ddfc-sddc.yaml", "DDFCalib + SDDC head"),
    "a7": ("yolov13-dad-a7-full.yaml", "DDFCalib + TIER-DCRA + SDDC head"),
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
    root = args.root.resolve()
    data = args.data.resolve()
    yaml_name, structure = STAGES[args.stage]
    model_yaml = root / "ultralytics" / "cfg" / "models" / "v13" / yaml_name
    pretrained = root / "yolov13n.pt"
    train_dir = root / "runs" / "train" / args.name
    manifest = root / "runs" / "train" / f"{args.name}.train.json"
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "pretrained weights")):
        if not path.is_file():
            raise FileNotFoundError(f"DAD {label} unavailable: {path}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PIN_MEMORY"] = "false"
    # This worker is launched as ``tools/train_dad_yolov13_worker.py``. Put the repository root before site-packages
    # so YOLO resolves the project-specific DSC3k2/DAD parser instead of a globally installed Ultralytics package.
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import ultralytics

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"DAD worker resolved external Ultralytics: {ultralytics.__file__}") from error
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    model.load(str(pretrained))
    model.train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        batch=16,
        imgsz=640,
        workers=2,
        amp=False,
        plots=False,
        deterministic=True,
        seed=args.seed,
        resume=False,
        device=0,
        project=str(root / "runs" / "train"),
        name=args.name,
        exist_ok=False,
    )
    best = train_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"DAD training finished without best.pt: {best}")
    payload = {
        "name": args.name,
        "stage": args.stage.upper(),
        "seed": args.seed,
        "structure": structure,
        "model_yaml": str(model_yaml),
        "dataset": str(data),
        "weights": str(best),
        "settings": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch": 16,
            "imgsz": 640,
            "workers": 2,
            "amp": False,
            "plots": False,
            "deterministic": True,
            "resume": False,
            "device": 0,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
