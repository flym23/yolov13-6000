#!/usr/bin/env python3
"""Static and forward verification for all SFR-SCAF ablation configurations."""

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import SFRSCAFFuse


CFG_NAMES = (
    "yolov13_sfr_scaf.yaml",
    "yolov13_sfr_scaf_no_semantic_filter.yaml",
    "yolov13_sfr_scaf_fixed_route.yaml",
    "yolov13_sfr_scaf_consistency_only.yaml",
    "yolov13_sfr_scaf_semantic_only.yaml",
)


def main():
    root = Path(__file__).resolve().parents[1]
    module = SFRSCAFFuse(512, 512).eval()
    x_up = torch.randn(2, 512, 40, 40)
    x_lat = torch.randn(2, 512, 40, 40)
    with torch.no_grad():
        output = module([x_up, x_lat])
    delta = (output - torch.cat((x_up, x_lat), dim=1)).abs().max().item()
    assert output.shape == (2, 1024, 40, 40)
    assert torch.isfinite(output).all()
    assert delta < 1e-3, delta
    print(f"module_identity_ok shape={tuple(output.shape)} max_abs_diff={delta:.8f}")

    for cfg_name in CFG_NAMES:
        cfg = root / "ultralytics/cfg/models/v13" / cfg_name
        detector = YOLO(str(cfg))
        model = detector.model.eval()
        with torch.no_grad():
            prediction = model(torch.randn(1, 3, 640, 640))
        assert prediction is not None
        model.fuse().eval()
        with torch.no_grad():
            fused_prediction = model(torch.randn(1, 3, 640, 640))
        assert fused_prediction is not None
        params = sum(parameter.numel() for parameter in model.parameters())
        print(f"model_ok cfg={cfg_name} params={params}")
        model.info(imgsz=640, verbose=False)


if __name__ == "__main__":
    main()
