#!/usr/bin/env python3
"""Train one UFCR-Next full/ablation worker with the server's YOLOv13 settings."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "full": "yolov13_ufcr_next.yaml",
    "wo_wt": "yolov13_ufcr_next_wo_wt.yaml",
    "wo_edge": "yolov13_ufcr_next_wo_edge.yaml",
    "wo_aarup": "yolov13_ufcr_next_wo_aarup.yaml",
    "wo_aardown": "yolov13_ufcr_next_wo_aardown.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single UFCR-Next stage/seed.")
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
        epochs=200,
        patience=40,
        batch=16,
        workers=8,
        amp=True,
        deterministic=False,
        plots=False,
        seed=args.seed,
        project=str(root / "runs/train"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
