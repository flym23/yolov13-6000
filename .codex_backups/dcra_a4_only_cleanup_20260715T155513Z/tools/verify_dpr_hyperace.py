#!/usr/bin/env python3
"""Verify DPR-HyperACE identity, gradients, parsing, forward, fuse-forward and loss paths."""

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import DPRFuseModule, DPRHyperACE
from ultralytics.nn.modules.block import FuseModule


CONFIGS = {
    "h1_dpr_p3_spd": "yolov13_dpr_hyperace.yaml",
    "h4_dpr_dwdown": "yolov13_dpr_hyperace_dwdown.yaml",
    "h5_dpr_replace": "yolov13_dpr_hyperace_replace.yaml",
    "h2_dpr_p5": "yolov13_dpr_hyperace_p5.yaml",
    "h3_dpr_dual": "yolov13_dpr_hyperace_dual.yaml",
    "h6_faar_dpr": "yolov13_faar_dpr_hyperace.yaml",
}


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def make_inputs(c=64, p3_size=(80, 80), p4_size=(40, 40), p5_size=(20, 20), requires_grad=False):
    return [
        torch.randn(2, c, *p3_size, requires_grad=requires_grad),
        torch.randn(2, c, *p4_size, requires_grad=requires_grad),
        torch.randn(2, 2 * c, *p5_size, requires_grad=requires_grad),
    ]


def identity_check(**kwargs):
    torch.manual_seed(0)
    base = FuseModule(c_in=64, channel_adjust=True).eval()
    dpr = DPRFuseModule(c_in=64, channel_adjust=True, **kwargs).eval()
    dpr.conv_out.load_state_dict(base.conv_out.state_dict())
    inputs = make_inputs()
    with torch.no_grad():
        expected, actual = base(inputs), dpr(inputs)
    difference = (expected - actual).abs().max().item()
    assert expected.shape == actual.shape
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6), difference
    return difference


def module_checks():
    p3_diff = identity_check(
        use_p3_detail=True,
        p3_down_mode="spd",
        p3_fusion_mode="residual",
        use_p5_semantic=False,
    )
    p5_diff = identity_check(
        use_p3_detail=False,
        p3_down_mode="spd",
        p3_fusion_mode="residual",
        use_p5_semantic=True,
    )
    dual_diff = identity_check(
        use_p3_detail=True,
        p3_down_mode="spd",
        p3_fusion_mode="residual",
        use_p5_semantic=True,
    )
    assert p3_diff == p5_diff == dual_diff == 0.0

    module = DPRFuseModule(64, use_p3_detail=True, use_p5_semantic=True)
    inputs = make_inputs(requires_grad=True)
    output = module(inputs)
    assert output.shape == (2, 64, 40, 40) and finite(output)
    output.square().mean().backward()
    for tensor in inputs:
        assert tensor.grad is not None and finite(tensor.grad)
    assert module.detail_alpha_raw.grad is not None and finite(module.detail_alpha_raw.grad)
    assert module.semantic_alpha_raw.grad is not None and finite(module.semantic_alpha_raw.grad)
    assert module.detail_alpha_raw.grad.abs().sum() > 0
    assert module.semantic_alpha_raw.grad.abs().sum() > 0
    assert all(finite(parameter.grad) for parameter in module.parameters() if parameter.grad is not None)

    odd_inputs = make_inputs(p3_size=(81, 79), p4_size=(41, 40), p5_size=(21, 20))
    with torch.no_grad():
        odd_output = module.eval()(odd_inputs)
    assert odd_output.shape == (2, 64, 41, 40) and finite(odd_output)

    replace = DPRFuseModule(64, use_p3_detail=True, p3_fusion_mode="replace").eval()
    base = FuseModule(64, True).eval()
    replace.conv_out.load_state_dict(base.conv_out.state_dict())
    inputs = make_inputs()
    with torch.no_grad():
        assert not torch.allclose(base(inputs), replace(inputs), atol=1e-6, rtol=1e-6)
    print("DPRFuse identity/backward/odd-size/replace checks: PASS")


def loss_batch(device, with_target):
    image = torch.rand(1, 3, 320, 320, device=device)
    if with_target:
        return {
            "img": image,
            "batch_idx": torch.tensor([0], device=device),
            "cls": torch.tensor([[0.0]], device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.12, 0.10]], device=device),
        }
    return {
        "img": image,
        "batch_idx": torch.empty(0, device=device),
        "cls": torch.empty((0, 1), device=device),
        "bboxes": torch.empty((0, 4), device=device),
    }


def model_checks(root):
    cfg_root = root / "ultralytics/cfg/models/v13"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for stage, filename in CONFIGS.items():
        model = YOLO(str(cfg_root / filename)).model.to(device)
        dpr_modules = [module for module in model.modules() if isinstance(module, DPRHyperACE)]
        assert len(dpr_modules) == 1, stage
        assert len(dpr_modules[0].m) == 1, f"{stage}: depth-scaled internal repeats should be 1 for n scale"
        assert model.model[-1].nl == 3, stage
        assert torch.count_nonzero(dpr_modules[0].fuse.detail_alpha_raw) == 0, stage
        assert torch.count_nonzero(dpr_modules[0].fuse.semantic_alpha_raw) == 0, stage
        alpha_names = [name for name, _ in model.named_parameters() if name.endswith("alpha_raw")]
        assert len(alpha_names) == 2, (stage, alpha_names)

        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 640, 640, device=device))
        assert finite(output), stage
        model.fuse()
        with torch.no_grad():
            fused_output = model(torch.randn(1, 3, 640, 640, device=device))
        assert finite(fused_output), stage

        for with_target in (False, True):
            loss_model = YOLO(str(cfg_root / filename)).model.to(device).train()
            loss, loss_items = loss_model(loss_batch(device, with_target))
            assert finite(loss) and finite(loss_items), (stage, with_target)
            loss.backward()
            assert all(finite(parameter.grad) for parameter in loss_model.parameters() if parameter.grad is not None)
            del loss_model, loss, loss_items
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{stage}: params={parameters:,}, Detect=3, forward/fuse/loss(empty+normal): PASS")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(4)
    module_checks()
    model_checks(Path(__file__).resolve().parents[1])
    print("All DPR-HyperACE self-checks: PASS")
