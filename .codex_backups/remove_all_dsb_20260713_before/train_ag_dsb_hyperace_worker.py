#!/usr/bin/env python3
"""Train one AG-DSB-HyperACE ablation with the established server hyperparameters."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds


MODEL_FILES = {
    "a1_dense_fixed": "yolov13_ag_dsb_dense_fixed.yaml",
    "a2_dense": "yolov13_ag_dsb_dense.yaml",
    "a3_topk2": "yolov13_ag_dsb_topk2.yaml",
    "a4_topk3": "yolov13_ag_dsb_topk3.yaml",
    "a5_topk2_no_norm": "yolov13_ag_dsb_topk2_no_norm.yaml",
    "a6_topk2_both": "yolov13_ag_dsb_topk2_both.yaml",
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

    # BaseTrainer seeds with seed + 1 + RANK, but it is created only after
    # YOLO has already parsed and initialized the model. Seed model
    # construction explicitly so AG-DSB-only parameters are reproducible.
    init_seeds(args.seed + 1, deterministic=True)
    model = YOLO(str(config))
    model.load(str(weights))
    expected_eta = 2 if args.stage == "a6_topk2_both" else (0 if args.stage == "a1_dense_fixed" else 1)

    def verify_optimizer(trainer):
        eta = {
            id(parameter): name
            for name, parameter in trainer.model.named_parameters()
            if name.endswith("eta_head_bias")
        }
        optimizer_groups = {
            id(parameter): float(group.get("weight_decay", 0.0))
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        missing = [name for parameter_id, name in eta.items() if parameter_id not in optimizer_groups]
        decayed = [name for parameter_id, name in eta.items() if optimizer_groups.get(parameter_id, 0.0) != 0.0]
        if len(eta) != expected_eta or missing or decayed:
            raise RuntimeError(
                "AG-DSB eta optimizer registration failed: "
                f"expected={expected_eta}, eta={list(eta.values())}, missing={missing}, decayed={decayed}"
            )
        print(f"AG-DSB eta optimizer registration: PASS ({', '.join(eta.values()) or 'fixed eta'})")

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
    print("AG-DSB training worker completed.")


if __name__ == "__main__":
    main()
