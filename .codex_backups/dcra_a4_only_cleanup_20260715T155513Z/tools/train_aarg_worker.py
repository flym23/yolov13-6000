#!/usr/bin/env python3
"""Train one AARG-YOLOv13 full/ablation worker with the server's YOLOv13 environment."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "e3_full": "yolov13_aarg.yaml",
    "e1_aarup": "yolov13_aarg_aarup.yaml",
    "e2_p2guide": "yolov13_aarg_p2guide.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single AARG stage/seed.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(MODEL_FILES), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    os.environ["WANDB_DISABLED"] = "true"

    model_cfg = root / "ultralytics/cfg/models/v13" / MODEL_FILES[args.stage]
    data_yaml = root / "data.yaml"
    pretrained = root / "yolov13n.pt"
    if not model_cfg.exists():
        raise FileNotFoundError(f"Missing model config: {model_cfg}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data config: {data_yaml}")
    if not pretrained.exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained}")

    model = YOLO(str(model_cfg))
    model.load(str(pretrained))
    model.train(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        epochs=200,
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
