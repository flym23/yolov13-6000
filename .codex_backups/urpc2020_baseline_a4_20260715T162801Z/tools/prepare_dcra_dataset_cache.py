#!/usr/bin/env python3
"""Build DCRA train/validation label caches once before parallel seed workers start."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare detection label caches serially.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def prepare_label_caches(data_yaml: Path, imgsz: int, batch: int) -> None:
    """Validate the dataset and populate train/val caches in this single process."""
    data = check_det_dataset(data_yaml, autodownload=False)
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "mode": "train",
            "imgsz": imgsz,
            "batch": batch,
            "cache": False,
            "rect": False,
            "single_cls": False,
            "fraction": 1.0,
            "classes": None,
        }
    )
    for mode in ("train", "val"):
        dataset = build_yolo_dataset(cfg, data[mode], batch=batch, data=data, mode=mode, stride=32)
        if not len(dataset):
            raise RuntimeError(f"The DCRA {mode} dataset is empty: {data[mode]}")
        print(f"Prepared {mode} label cache: {len(dataset)} images")


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("--imgsz and --batch must both be positive.")
    prepare_label_caches(args.data.resolve(), args.imgsz, args.batch)


if __name__ == "__main__":
    main()
