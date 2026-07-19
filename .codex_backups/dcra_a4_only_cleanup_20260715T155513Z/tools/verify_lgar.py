#!/usr/bin/env python3
"""Model, identity, backward and forward checks for the LGARUp ablations."""

from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import LGARUp


ALL_CONFIGS = (
    "yolov13_faar_p4_only.yaml",
    "yolov13_faar_p3_only.yaml",
    "yolov13_lgar_p4.yaml",
    "yolov13_lgar_p3.yaml",
    "yolov13_lgar.yaml",
    "yolov13_lgar_p4_no_lateral.yaml",
    "yolov13_lgar_p4_no_offset.yaml",
    "yolov13_lgar_p4_no_confidence.yaml",
)
LGAR_CONFIGS = ALL_CONFIGS[2:]


def check_module(c_deep, c_lat, mode, deep_shape, lateral_shape):
    module = LGARUp(c_deep, c_lat, mode=mode, groups=4)
    deep = torch.randn(*deep_shape, requires_grad=True)
    lateral = torch.randn(*lateral_shape, requires_grad=True)
    output = module([deep, lateral])
    reference = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    assert output.shape == (deep.shape[0], deep.shape[1], lateral.shape[2], lateral.shape[3])
    assert torch.allclose(output, reference, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert deep.grad is not None and torch.isfinite(deep.grad).all()
    print(f"identity_backward_ok mode={mode} shape={tuple(output.shape)}")


def main():
    check_module(1024, 512, "semantic", (2, 1024, 20, 20), (2, 512, 40, 40))
    check_module(512, 256, "detail", (2, 512, 40, 40), (2, 256, 80, 80))
    dynamic = LGARUp(512, 256, mode="detail", groups=4)
    dynamic.gamma_raw.data.fill_(0.5)
    deep, lateral = torch.randn(2, 512, 40, 40, requires_grad=True), torch.randn(2, 256, 80, 80, requires_grad=True)
    output = dynamic([deep, lateral])
    assert output.shape == (2, 512, 80, 80) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in dynamic.parameters() if p.grad is not None)
    print("dynamic_branch_ok")

    root = Path(__file__).resolve().parents[1]
    for name in ALL_CONFIGS:
        model = YOLO(str(root / "ultralytics/cfg/models/v13" / name)).model.eval()
        assert len(model.model[-1].f) == 3
        params = sum(p.numel() for p in model.parameters())
        print(f"build_ok cfg={name} params={params}")
        model.info(imgsz=640, verbose=False)
    for name in LGAR_CONFIGS:
        model = YOLO(str(root / "ultralytics/cfg/models/v13" / name)).model.eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 640, 640)) is not None
        model.fuse().eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 640, 640)) is not None
        print(f"forward_fuse_ok cfg={name}")


if __name__ == "__main__":
    main()
