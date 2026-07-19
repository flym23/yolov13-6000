#!/usr/bin/env python3
"""Train one WSDR-YOLOv13 ablation stage and seed."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "w2_fixed": "yolov13_wsdr_fixed.yaml",
    "w3_main": "yolov13_wsdr.yaml",
    "w4_no_hf": "yolov13_wsdr_no_hf.yaml",
    "w5_avgpool": "yolov13_wsdr_avgpool.yaml",
    "w6_hf_reweight": "yolov13_wsdr_hf_reweight.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single WSDR stage/seed.")
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
        if not path.is_file():
            raise FileNotFoundError(path)
    model = YOLO(str(model_cfg))
    # Use the same original YOLOv13 initialization as D0/B3; never load a B3 experiment checkpoint.
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
