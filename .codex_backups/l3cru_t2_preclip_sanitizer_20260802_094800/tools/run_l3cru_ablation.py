#!/usr/bin/env python3
"""Train one deterministic, pretrained L3-CRU-YOLOv13 ablation seed on URPC2020half."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch


STAGES = {
    "t1_amsc": "yolov13-l3cru-t1_amsc.yaml",
    "t2_bgdr": "yolov13-l3cru-t2_bgdr.yaml",
    "t3_ugdr": "yolov13-l3cru-t3_ugdr.yaml",
    "t4_amsc_bgdr": "yolov13-l3cru-t4_amsc_bgdr.yaml",
    "t5_amsc_ugdr": "yolov13-l3cru-t5_amsc_ugdr.yaml",
    "t6_bgdr_ugdr": "yolov13-l3cru-t6_bgdr_ugdr.yaml",
    "t7_full": "yolov13-l3cru-t7_full.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_epoch(results_csv: Path) -> int | None:
    if not results_csv.is_file():
        return None
    with results_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    metric = "metrics/mAP50-95(B)"
    values = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise FloatingPointError(f"Invalid {metric} in {results_csv}: {row.get(metric)!r}") from error
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite {metric} in {results_csv}: {value!r}")
        values.append((value, row))
    best = max(values, key=lambda item: item[0])[1]
    return int(float(best["epoch"]))


def add_finite_guard(model) -> None:
    """Stop the seed immediately if the loss or BGDR residual projection becomes non-finite."""

    def check_batch(trainer) -> None:
        loss = trainer.loss.detach()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite training loss.")
        for name, parameter in trainer.model.named_parameters():
            if ".detail_out." in name and not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite BGDR residual parameter: {name}")

    model.add_callback("on_train_batch_end", check_batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    pretrained = root / "yolov13n.pt"
    model_yaml = root / "ultralytics" / "cfg" / "models" / "v13" / "l3cru" / STAGES[args.stage]
    output_dir = root / "runs" / "train" / args.name
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "YOLOv13n pretrained weights")):
        if not path.is_file():
            raise FileNotFoundError(f"L3-CRU {label} unavailable: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse L3-CRU output: {output_dir}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PIN_MEMORY"] = "false"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ultralytics

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"L3-CRU worker resolved external Ultralytics: {ultralytics.__file__}") from error
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    model.load(str(pretrained))
    if not model.ckpt:
        raise RuntimeError("YOLO.load() did not retain a checkpoint for Trainer")
    add_finite_guard(model)
    initialization = {
        "method": "YOLO.load",
        "pretrained": str(pretrained),
        "pretrained_sha256": sha256(pretrained),
        "trainer_receives_loaded_model": bool(model.ckpt),
    }

    started = perf_counter()
    model.train(
        data=str(data), epochs=args.epochs, patience=args.patience, batch=16, imgsz=640, workers=2,
        amp=False, deterministic=True, plots=False, seed=args.seed, resume=False, device=0,
        project=str(root / "runs" / "train"), name=args.name, exist_ok=False,
    )
    elapsed_seconds = perf_counter() - started
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"L3-CRU training finished without best.pt: {best}")
    payload = {
        "name": args.name,
        "stage": args.stage,
        "seed": args.seed,
        "model_yaml": str(model_yaml),
        "dataset": str(data),
        "weights": str(best),
        "weights_sha256": sha256(best),
        "initialization": initialization,
        "best_epoch": best_epoch(output_dir / "results.csv"),
        "training_seconds": elapsed_seconds,
        "settings": {
            "epochs": args.epochs, "patience": args.patience, "batch": 16, "imgsz": 640, "workers": 2,
            "amp": False, "deterministic": True, "plots": False, "resume": False, "device": 0,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "runs" / "train" / f"{args.name}.train.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
