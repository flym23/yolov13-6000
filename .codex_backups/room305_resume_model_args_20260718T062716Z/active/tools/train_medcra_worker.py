#!/usr/bin/env python3
"""Train one URPC2020 ME-DCRA ablation seed with fixed FP32 and stopping settings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
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
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this exact run from its last.pt after relocating checkpoint paths.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a full training checkpoint across supported PyTorch releases."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary at {path}, got {type(checkpoint)!r}.")
    return checkpoint


def prepare_relocated_resume_checkpoint(root: Path, data: Path, name: str) -> Path:
    """Copy last.pt with account-independent arguments while preserving the original checkpoint."""
    source = root / "runs/train" / name / "weights/last.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {source}")

    checkpoint = load_checkpoint(source)
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        raise TypeError(f"Checkpoint train_args are invalid in {source}.")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0 or epoch >= 299:
        raise ValueError(f"Checkpoint is not resumable to epoch 300: {source}, epoch={epoch!r}.")

    relocated_args = dict(train_args)
    relocated_args.update(
        {
            "data": str(data),
            "project": str(root / "runs/train"),
            "name": name,
            "exist_ok": True,
            "epochs": 300,
            "patience": 40,
            "workers": 2,
            "device": 0,
            "amp": False,
            "plots": False,
            "resume": True,
        }
    )
    checkpoint["train_args"] = relocated_args
    target = source.with_name("last.relocated_resume.pt")
    temporary = target.with_suffix(".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def main():
    args = parse_args()
    root = args.root.resolve()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
    if args.epochs != 300:
        raise ValueError(f"ME-DCRA experiments require exactly 300 epochs, got {args.epochs}.")
    if args.patience != 40:
        raise ValueError(f"ME-DCRA experiments require patience=40, got {args.patience}.")
    model_yaml = root / "ultralytics/cfg/models/v13" / MODEL_FILES[args.stage]
    if not model_yaml.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {model_yaml}")

    resume_checkpoint = prepare_relocated_resume_checkpoint(root, data, args.name) if args.resume else None
    model = YOLO(str(resume_checkpoint or model_yaml))
    model.train(
        data=str(data),
        epochs=300,
        patience=40,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        seed=args.seed,
        deterministic=True,
        pretrained=False,
        optimizer="auto",
        amp=False,
        plots=False,
        resume=bool(resume_checkpoint),
        project=str(root / "runs/train"),
        name=args.name,
        exist_ok=bool(resume_checkpoint),
    )


if __name__ == "__main__":
    main()
