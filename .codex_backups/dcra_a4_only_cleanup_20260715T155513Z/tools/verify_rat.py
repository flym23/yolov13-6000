#!/usr/bin/env python3
"""Self-check RATunnel identity, gradients, model construction, forward and fuse-forward."""

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import RATunnel


CONFIGS = {
    "t1_rat_initial": ("yolov13_rat_initial.yaml", 3),
    "t4_rat_no_amplitude": ("yolov13_rat_no_amplitude.yaml", 3),
    "t5_rat_channel_only": ("yolov13_rat_channel_only.yaml", 3),
    "t6_rat_amplitude_only": ("yolov13_rat_amplitude_only.yaml", 3),
    "t2_rat_late": ("yolov13_rat_late.yaml", 4),
    "t3_rat_all": ("yolov13_rat_all.yaml", 7),
    "t7_faar_rat_initial": ("yolov13_faar_rat_initial.yaml", 3),
}


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (list, tuple)):
        return all(finite(x) for x in value)
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    return True


def module_checks():
    for channels, height, width, mode, spatial in ((256, 80, 80, "p3", False), (512, 40, 40, "p4", True), (1024, 20, 20, "p5", False)):
        module = RATunnel(channels, channels, mode=mode, use_amplitude=True, use_channel=True, use_spatial=spatial)
        original = torch.randn(2, channels, height, width, requires_grad=True)
        enhanced = torch.randn_like(original, requires_grad=True)
        output = module([original, enhanced])
        assert output.shape == original.shape and torch.allclose(output, original, atol=1e-6, rtol=1e-6)
        assert finite(output)
        output.mean().backward()
        assert finite(original.grad) and module.gamma_raw.grad is not None and finite(module.gamma_raw.grad)
    module = RATunnel(256, 128, mode="p4", use_amplitude=True, use_channel=True, use_spatial=True)
    module.gamma_raw.data.fill_(0.5)
    original, enhanced = torch.randn(2, 256, 40, 40, requires_grad=True), torch.randn(2, 128, 20, 20, requires_grad=True)
    output = module([original, enhanced])
    assert output.shape == original.shape and finite(output) and not torch.allclose(output, original)
    output.square().mean().backward()
    assert all(finite(p.grad) for p in module.parameters() if p.grad is not None)
    print("RATunnel identity/backward/non-zero-gamma checks: PASS")


def model_checks(root):
    cfg_root = root / "ultralytics/cfg/models/v13"
    x = torch.randn(1, 3, 640, 640)
    for stage, (filename, rat_count) in CONFIGS.items():
        model = YOLO(str(cfg_root / filename)).model.eval()
        assert sum(isinstance(m, RATunnel) for m in model.modules()) == rat_count, stage
        assert model.model[-1].nl == 3, stage
        with torch.no_grad():
            output = model(x)
        assert finite(output), stage
        model.fuse()
        with torch.no_grad():
            fused_output = model(x)
        assert finite(fused_output), stage
        print(f"{stage}: RAT={rat_count}, Detect=3, forward/fuse-forward: PASS")


if __name__ == "__main__":
    torch.set_num_threads(4)
    module_checks()
    model_checks(Path(__file__).resolve().parents[1])
    print("All RAT self-checks: PASS")
