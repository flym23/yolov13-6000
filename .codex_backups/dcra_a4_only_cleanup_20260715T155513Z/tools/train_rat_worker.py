#!/usr/bin/env python3
"""Train one RATunnel ablation with the established server hyperparameters."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "t1_rat_initial": "yolov13_rat_initial.yaml",
    "t4_rat_no_amplitude": "yolov13_rat_no_amplitude.yaml",
    "t5_rat_channel_only": "yolov13_rat_channel_only.yaml",
    "t6_rat_amplitude_only": "yolov13_rat_amplitude_only.yaml",
    "t2_rat_late": "yolov13_rat_late.yaml",
    "t3_rat_all": "yolov13_rat_all.yaml",
    "t7_faar_rat_initial": "yolov13_faar_rat_initial.yaml",
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
    config = root / "ultralytics/cfg/models/v13" / MODEL_FILES[args.stage]
    data, weights = root / "data.yaml", root / "yolov13n.pt"
    for path in (config, data, weights):
        if not path.exists():
            raise FileNotFoundError(path)
    model = YOLO(str(config))
    model.load(str(weights))
    model.train(data=str(data), imgsz=640, batch=16, epochs=args.epochs, optimizer="auto", lr0=0.01, lrf=0.01, close_mosaic=5, patience=40, workers=8, amp=True, deterministic=True, plots=False, seed=args.seed, project=str(root / "runs/train"), name=args.name)


if __name__ == "__main__":
    main()
