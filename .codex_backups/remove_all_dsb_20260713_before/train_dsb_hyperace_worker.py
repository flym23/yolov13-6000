#!/usr/bin/env python3
"""Train one DSB-HyperACE ablation with the established server hyperparameters."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ultralytics import YOLO


MODEL_FILES = {
    "k1_dualnorm": "yolov13_dsb_hyperace_dualnorm.yaml",
    "k2_topk4": "yolov13_dsb_hyperace.yaml",
    "k3_both": "yolov13_dsb_hyperace_both.yaml",
    "k4_topk2": "yolov13_dsb_hyperace_topk2.yaml",
    "k5_topk6": "yolov13_dsb_hyperace_topk6.yaml",
    "k6_faar_dsb": "yolov13_faar_dsb_hyperace.yaml",
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
    expected_eta = 2 if args.stage == "k3_both" else 1

    def verify_optimizer(trainer):
        eta = {id(parameter): name for name, parameter in trainer.model.named_parameters() if name.endswith("eta_raw")}
        optimizer_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
        missing = [name for parameter_id, name in eta.items() if parameter_id not in optimizer_ids]
        if len(eta) != expected_eta or missing:
            raise RuntimeError(
                f"DSB eta optimizer registration failed: expected={expected_eta}, eta={list(eta.values())}, missing={missing}"
            )
        print(f"DSB eta optimizer registration: PASS ({', '.join(eta.values())})")

    model.add_callback("on_train_start", verify_optimizer)
    model.train(
        data=str(data),
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
    eta_values = {
        name: parameter.detach().float().cpu().tolist()
        for name, parameter in model.model.named_parameters()
        if name.endswith("eta_raw")
    }
    if len(eta_values) != expected_eta or not all(torch.isfinite(torch.tensor(value)).all() for value in eta_values.values()):
        raise RuntimeError(f"Invalid trained DSB eta values: {eta_values}")
    print(f"DSB eta values after training: {eta_values}")


if __name__ == "__main__":
    main()

