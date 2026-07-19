#!/usr/bin/env python3
"""Train one SCAF-YOLOv13 ablation worker with the server environment."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "s4_scaf": "yolov13_scaf.yaml",
    "s5_scaf_no_consistency": "yolov13_scaf_no_consistency.yaml",
    "s2_scaf_p4": "yolov13_scaf_p4.yaml",
    "s3_scaf_p3": "yolov13_scaf_p3.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single SCAF stage/seed.")
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
