#!/usr/bin/env python3
"""Train one URPC2020 ME-DCRA ablation seed with fixed FP32 settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "m0_original": "yolov13-original.yaml",
    "m1_a4": "yolov13-dcra-tau020.yaml",
    "m2_full": "yolov13-medcra.yaml",
    "m3_no_moment": "yolov13-medcra-no-moment.yaml",
    "m4_no_center": "yolov13-medcra-no-center.yaml",
    "m5_no_bound": "yolov13-medcra-no-bound.yaml",
    "m6_rho005": "yolov13-medcra-rho005.yaml",
    "m7_rho020": "yolov13-medcra-rho020.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(MODEL_FILES), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
    if args.epochs != 300:
        raise ValueError(f"ME-DCRA experiments require exactly 300 epochs, got {args.epochs}.")
    model_yaml = root / "ultralytics/cfg/models/v13" / MODEL_FILES[args.stage]
    if not model_yaml.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {model_yaml}")

    model = YOLO(str(model_yaml))
    model.train(
        data=str(data),
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        seed=args.seed,
        deterministic=True,
        pretrained=False,
        optimizer="auto",
        amp=False,
        project=str(root / "runs/train"),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
