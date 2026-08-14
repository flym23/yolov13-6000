#!/usr/bin/env python3
"""Train one immutable DOR factorial configuration under the locked URPC2019 protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-yaml", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_path = Path(ultralytics.__file__).resolve()
    if ROOT not in package_path.parents:
        raise RuntimeError(f"Imported ultralytics outside project root: {package_path}")
    model_yaml = args.model_yaml.resolve()
    data_yaml = Path(os.environ.get("URPC2019_ROOT", "/home/room305/ZZF/URPC2019")) / "data.yaml"
    if not model_yaml.is_file() or not data_yaml.is_file():
        raise FileNotFoundError(f"model={model_yaml}, data={data_yaml}")
    os.chdir(ROOT)
    model = YOLO(str(model_yaml)).load("yolov13n.pt")
    model.train(
        data=str(data_yaml), epochs=300, patience=40, device=0, workers=2, amp=False,
        deterministic=True, plots=False, imgsz=640, batch=16, seed=args.seed,
        project=str(args.project.resolve()), name=args.name, exist_ok=False,
    )


if __name__ == "__main__":
    main()
