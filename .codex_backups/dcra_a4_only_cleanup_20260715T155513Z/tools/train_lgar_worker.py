#!/usr/bin/env python3
"""Train one FAARUp/LGARUp ablation with the established server parameters."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "u1_faar_p4_only": "yolov13_faar_p4_only.yaml",
    "u2_faar_p3_only": "yolov13_faar_p3_only.yaml",
    "g1_lgar_p4": "yolov13_lgar_p4.yaml",
    "g2_lgar_p3": "yolov13_lgar_p3.yaml",
    "g3_lgar": "yolov13_lgar.yaml",
    "g4_lgar_p4_no_lateral": "yolov13_lgar_p4_no_lateral.yaml",
    "g5_lgar_p4_no_offset": "yolov13_lgar_p4_no_offset.yaml",
    "g6_lgar_p4_no_confidence": "yolov13_lgar_p4_no_confidence.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(MODEL_FILES), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ["WANDB_DISABLED"] = "true"
    config, data, weights = root / "ultralytics/cfg/models/v13" / MODEL_FILES[args.stage], root / "data.yaml", root / "yolov13n.pt"
    for path in (config, data, weights):
        if not path.exists():
            raise FileNotFoundError(path)
    model = YOLO(str(config))
    model.load(str(weights))
    model.train(data=str(data), imgsz=640, batch=16, epochs=args.epochs, optimizer="auto", lr0=0.01, lrf=0.01, close_mosaic=5, patience=40, workers=8, amp=True, deterministic=True, plots=False, seed=args.seed, project=str(root / "runs/train"), name=args.name)


if __name__ == "__main__":
    main()
