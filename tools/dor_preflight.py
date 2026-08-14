#!/usr/bin/env python3
"""Mandatory DOR gates: modules, eight 2^3 topologies, real URPC loss, precision and ONNX."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
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


def real_urpc_loss(config: Path, data_yaml: Path) -> str:
    """Run one real URPC batch through DOR loss/backward without beginning an epoch."""
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_cfg(overrides={"data": str(data_yaml), "imgsz": 640, "batch": 2, "workers": 0, "cache": False, "rect": False})
    data = check_det_dataset(str(data_yaml))
    model = YOLO(str(config)).model.to(device).train()
    model.nc, model.names, model.args = data["nc"], data["names"], args
    dataset = build_yolo_dataset(args, data["train"], batch=2, data=data, mode="train", rect=False, stride=max(int(model.stride.max()), 32))
    batch = next(iter(build_dataloader(dataset, batch=2, workers=0, shuffle=False, rank=-1)))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=False)
    batch["img"] = batch["img"].float() / 255.0
    loss, loss_items = model.loss(batch)
    if not torch.isfinite(loss) or not finite_tree(loss_items, torch):
        raise RuntimeError("real URPC DOR loss is non-finite")
    loss.backward()
    relevant = [parameter.grad for name, parameter in model.named_parameters() if any(token in name for token in ("DCPR", "OCARFuse", "cvo")) and parameter.grad is not None]
    if not relevant or not all(torch.isfinite(gradient).all() for gradient in relevant):
        raise RuntimeError("DOR-specific gradients are missing or non-finite on the real URPC batch")
    return f"passed:{device.type}:loss={float(loss.detach().cpu()):.6f}"


def forward_modes(config: Path) -> dict[str, str]:
    import torch
    from ultralytics import YOLO

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result: dict[str, str] = {}
    model = YOLO(str(config)).model.to(device).eval()
    with torch.no_grad():
        if not finite_tree(model(torch.zeros(1, 3, 640, 640, device=device)), torch):
            raise RuntimeError("DOR FP32 forward produced non-finite values")
    result["fp32"] = "passed"
    if device.type == "cuda":
        model = YOLO(str(config)).model.to(device).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if not finite_tree(model(torch.zeros(1, 3, 640, 640, device=device)), torch):
                raise RuntimeError("DOR CUDA AMP forward produced non-finite values")
        result["amp_fp16"] = "passed"
        model = YOLO(str(config)).model.to(device).half().eval()
        with torch.no_grad():
            if not finite_tree(model(torch.zeros(1, 3, 640, 640, device=device, dtype=torch.float16)), torch):
                raise RuntimeError("DOR CUDA FP16 forward produced non-finite values")
        result["fp16"] = "passed"
    else:
        result.update({"amp_fp16": "skipped_no_cuda", "fp16": "skipped_no_cuda"})
    return result


def main() -> None:
    args = parse_args()
    root, config_dir, output, data_yaml = args.project_root.resolve(), args.config_dir.resolve(), args.output_dir.resolve(), args.data.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ultralytics
    from ultralytics import YOLO

    package_path = Path(ultralytics.__file__).resolve()
    if root not in package_path.parents:
        raise RuntimeError(f"Imported ultralytics outside project root: {package_path}")
    configs = sorted(config_dir.glob("yolov13n-dor-*.yaml"))
    if len(configs) != 8 or not data_yaml.is_file():
        raise FileNotFoundError(f"expected eight DOR YAMLs and data YAML: {config_dir}, {data_yaml}")
    pytest_log = output / "pytest.log"
    with pytest_log.open("wb") as stream:
        completed = subprocess.run([str(Path(sys.executable).resolve()), "-m", "pytest", "-q", "tests/test_dor_yolov13.py", "tests/test_urpc2019_one_based_labels.py"], cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"DOR unit tests failed; see {pytest_log}")
    for config in configs:
        YOLO(str(config)).model
    full_config = config_dir / "yolov13n-dor-dor.yaml"
    modes, batch_loss = forward_modes(full_config), real_urpc_loss(full_config, data_yaml)
    onnx_dir = output / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_config = onnx_dir / full_config.name
    shutil.copy2(full_config, onnx_config)
    exported = Path(YOLO(str(onnx_config)).export(format="onnx", imgsz=640, opset=17, simplify=False, device=0)).resolve()
    if not exported.is_file():
        raise RuntimeError(f"ONNX export did not produce a file: {exported}")
    if onnx_dir not in exported.parents:
        destination = onnx_dir / exported.name
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite existing ONNX artifact: {destination}")
        shutil.move(str(exported), str(destination))
        exported = destination
    report = {"status": "passed", "project_root": str(root), "ultralytics": str(package_path), "configs": [str(path) for path in configs], "unit_tests": str(pytest_log), "real_urpc_batch_loss": batch_loss, "forward_modes": modes, "onnx": str(exported), "completed_at": datetime.now(timezone.utc).isoformat()}
    (output / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
