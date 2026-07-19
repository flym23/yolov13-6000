#!/usr/bin/env python3
"""Train one SFR-SCAF ablation worker using the server's established environment."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "f3_sfr_scaf": "yolov13_sfr_scaf.yaml",
    "f4_no_semantic_filter": "yolov13_sfr_scaf_no_semantic_filter.yaml",
    "f5_fixed_route": "yolov13_sfr_scaf_fixed_route.yaml",
    "f6_consistency_only": "yolov13_sfr_scaf_consistency_only.yaml",
    "f7_semantic_only": "yolov13_sfr_scaf_semantic_only.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train one SFR-SCAF stage/seed.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(MODEL_FILES), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = root / "ultralytics/cfg/models/v13" / MODEL_FILES[args.stage]
    data_yaml = root / "data.yaml"
    pretrained = root / "yolov13n.pt"
    for path in (model_cfg, data_yaml, pretrained):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    model = YOLO(str(model_cfg))
    model.load(str(pretrained))
    model.train(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        epochs=args.epochs,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        close_mosaic=5,
        patience=40,
        workers=8,
        amp=True,
        deterministic=True,
        plots=False,
        seed=args.seed,
        project=str(root / "runs/train"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
