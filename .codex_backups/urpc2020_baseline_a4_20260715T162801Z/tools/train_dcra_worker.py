#!/usr/bin/env python3
"""Train one DCRA-YOLOv13 A4 stage and seed with the D0 protocol."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {"a4_tau020": "yolov13-dcra-tau020.yaml"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train one DCRA stage/seed.")
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
        # RTX PRO 6000 Blackwell + torch 2.13/cu132 raises cudaErrorIllegalAddress
        # on the first real DCRA batch under FP16 autocast. Use one consistent
        # FP32 policy for every stage and seed on server 2.
        amp=False,
        deterministic=True,
        plots=False,
        seed=args.seed,
        project=str(root / "runs/train"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
