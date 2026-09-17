"""Train one locked URPC2019 baseline/POD reproduction variant."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANT_CONFIGS = {
    "l0": "yolov13-lcer-dcra-l0.yaml",
    "o": "yolov13n-pod-o.yaml",
    "d": "yolov13n-pod-d.yaml",
    "od": "yolov13n-pod-od.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANT_CONFIGS), required=True)
    parser.add_argument("--seed", choices=(0, 1, 2), type=int, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import ultralytics
    from ultralytics import YOLO

    package_path = Path(ultralytics.__file__).resolve()
    if ROOT not in package_path.parents:
        raise RuntimeError(f"Imported ultralytics outside project root: {package_path}")
    model_yaml = ROOT / "ultralytics" / "cfg" / "models" / "v13" / VARIANT_CONFIGS[args.variant]
    data_yaml = Path(os.environ.get("URPC2019_ROOT", "/home/room305/ZZF/URPC2019")) / "data.yaml"
    weights = ROOT / "yolov13n.pt"
    for path in (model_yaml, data_yaml, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.project.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    model = YOLO(str(model_yaml)).load("yolov13n.pt")
    model.train(
        data=str(data_yaml),
        epochs=160,
        device=0,
        workers=0,
        amp=False,
        deterministic=True,
        plots=False,
        imgsz=640,
        batch=16,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
