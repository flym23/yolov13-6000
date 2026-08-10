#!/usr/bin/env python3
"""Run required POD implementation gates before any complete factorial training starts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIANT_CONFIGS = (
    "yolov13n-pod-d0.yaml",
    "yolov13n-pod-p.yaml",
    "yolov13n-pod-o.yaml",
    "yolov13n-pod-d.yaml",
    "yolov13n-pod-po.yaml",
    "yolov13n-pod-pd.yaml",
    "yolov13n-pod-od.yaml",
    "yolov13n-pod.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def finite_tree(value: Any, torch) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item, torch) for item in value)
    return True


def run_forward_modes(root: Path, config: Path) -> dict[str, str]:
    import torch
    from ultralytics import YOLO

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results: dict[str, str] = {}

    model = YOLO(str(config)).model.to(device).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640, device=device))
    if not finite_tree(output, torch):
        raise RuntimeError("FP32 full-model forward produced non-finite values.")
    results["fp32"] = "passed"

    if device.type == "cuda":
        model = YOLO(str(config)).model.to(device).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            output = model(torch.zeros(1, 3, 640, 640, device=device))
        if not finite_tree(output, torch):
            raise RuntimeError("CUDA AMP full-model forward produced non-finite values.")
        results["amp_fp16"] = "passed"

        model = YOLO(str(config)).model.to(device).half().eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 640, 640, device=device, dtype=torch.float16))
        if not finite_tree(output, torch):
            raise RuntimeError("CUDA FP16 full-model forward produced non-finite values.")
        results["fp16"] = "passed"
    else:
        results["amp_fp16"] = "skipped_no_cuda"
        results["fp16"] = "skipped_no_cuda"
    return results


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import ultralytics
    from ultralytics import YOLO

    package_path = Path(ultralytics.__file__).resolve()
    if root not in package_path.parents:
        raise RuntimeError(f"Imported ultralytics outside project root: {package_path}")
    config_dir = root / "ultralytics" / "cfg" / "models" / "v13"
    configs = [config_dir / name for name in VARIANT_CONFIGS]
    missing = [str(path) for path in configs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    pytest_log = output / "pytest.log"
    with pytest_log.open("wb") as stream:
        test = subprocess.run(
            [str(Path(sys.executable).resolve()), "-m", "pytest", "-q", "tests/test_pod_yolov13.py", "tests/test_urpc2019_one_based_labels.py"],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if test.returncode != 0:
        raise RuntimeError(f"unit tests failed; see {pytest_log}")

    for config in configs:
        YOLO(str(config)).model

    modes = run_forward_modes(root, config_dir / "yolov13n-pod.yaml")
    onnx_dir = output / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_config = onnx_dir / "yolov13n-pod.yaml"
    shutil.copy2(config_dir / "yolov13n-pod.yaml", onnx_config)
    export_result = YOLO(str(onnx_config)).export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=False,
        device=0,
    )
    export_path = Path(export_result).resolve()
    if not export_path.is_file():
        raise RuntimeError(f"ONNX export did not produce a file: {export_path}")
    if onnx_dir not in export_path.parents:
        destination = onnx_dir / export_path.name
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite an existing ONNX gate artifact: {destination}")
        shutil.move(str(export_path), str(destination))
        export_path = destination

    report = {
        "status": "passed",
        "project_root": str(root),
        "ultralytics": str(package_path),
        "configs": [str(path) for path in configs],
        "unit_tests": str(pytest_log),
        "forward_modes": modes,
        "onnx": str(export_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
