#!/usr/bin/env python3
"""Train one DPR-HyperACE ablation with the established server hyperparameters."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


MODEL_FILES = {
    "h1_dpr_p3_spd": "yolov13_dpr_hyperace.yaml",
    "h4_dpr_dwdown": "yolov13_dpr_hyperace_dwdown.yaml",
    "h5_dpr_replace": "yolov13_dpr_hyperace_replace.yaml",
    "h2_dpr_p5": "yolov13_dpr_hyperace_p5.yaml",
    "h3_dpr_dual": "yolov13_dpr_hyperace_dual.yaml",
    "h6_faar_dpr": "yolov13_faar_dpr_hyperace.yaml",
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

    def verify_optimizer(trainer):
        alpha = {id(parameter): name for name, parameter in trainer.model.named_parameters() if name.endswith("alpha_raw")}
        optimizer_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
        missing = [name for parameter_id, name in alpha.items() if parameter_id not in optimizer_ids]
        if len(alpha) != 2 or missing:
            raise RuntimeError(f"DPR alpha optimizer registration failed: alpha={list(alpha.values())}, missing={missing}")
        print(f"DPR alpha optimizer registration: PASS ({', '.join(alpha.values())})")

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
    alpha_values = {name: parameter.detach().float().cpu().tolist() for name, parameter in model.model.named_parameters() if name.endswith("alpha_raw")}
    print(f"DPR alpha values after training: {alpha_values}")


if __name__ == "__main__":
    main()
