#!/usr/bin/env python3
"""Verify DSB-HyperACE identity, normalization, gradients, parsing, forward, fuse and loss paths."""

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import DSBAdaHGConv, DSBC3AH, DSBHyperACE
from ultralytics.nn.modules.block import AdaHGConv, C3AH, FuseModule


CONFIGS = {
    "k1_dualnorm": ("yolov13_dsb_hyperace_dualnorm.yaml", 1, 0),
    "k2_topk4": ("yolov13_dsb_hyperace.yaml", 1, 4),
    "k3_both": ("yolov13_dsb_hyperace_both.yaml", 2, 4),
    "k4_topk2": ("yolov13_dsb_hyperace_topk2.yaml", 1, 2),
    "k5_topk6": ("yolov13_dsb_hyperace_topk6.yaml", 1, 6),
    "k6_faar_dsb": ("yolov13_faar_dsb_hyperace.yaml", 1, 4),
}


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def max_abs_diff(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.shape == right.shape
        return (left - right).abs().max().item()
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        return max((max_abs_diff(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        return max((max_abs_diff(left[key], right[key]) for key in left), default=0.0)
    return 0.0


def identity_check():
    torch.manual_seed(0)
    base = AdaHGConv(64, 8, 4, 0.1, "both")
    dsb = DSBAdaHGConv(64, 8, 4, 0.1, "both", topk=4, eta_max=0.10)
    result = dsb.load_state_dict(base.state_dict(), strict=False)
    assert set(result.missing_keys) == {"eta_raw"}, result
    assert not result.unexpected_keys, result
    base.eval()
    dsb.eval()
    x = torch.randn(2, 400, 64)
    with torch.no_grad():
        expected, actual = base(x), dsb(x)
    difference = (expected - actual).abs().max().item()
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6), difference
    print(f"DSBAdaHGConv zero-start identity: PASS (max_diff={difference:.3g})")


def propagation_checks():
    module = DSBAdaHGConv(64, 8, 4, topk=4, eta_max=0.10).eval()
    x = torch.randn(2, 100, 64)
    with torch.no_grad():
        logits = module.edge_generator(x)
        a_v2e = torch.softmax(logits, dim=1)
        a_e2v = module._build_e2v_weights(logits)
    assert torch.allclose(a_v2e.sum(dim=1), torch.ones_like(a_v2e.sum(dim=1)), atol=1e-5, rtol=1e-5)
    assert torch.allclose(a_e2v.sum(dim=2), torch.ones_like(a_e2v.sum(dim=2)), atol=1e-5, rtol=1e-5)
    nonzero = (a_e2v > 0).sum(dim=2)
    assert int(nonzero.min()) == int(nonzero.max()) == 4

    dense = DSBAdaHGConv(64, 8, 4, topk=0).eval()
    with torch.no_grad():
        dense_logits = dense.edge_generator(x)
        dense_e2v = dense._build_e2v_weights(dense_logits)
    assert torch.all(dense_e2v > 0)
    assert torch.allclose(dense_e2v.sum(dim=2), torch.ones_like(dense_e2v.sum(dim=2)), atol=1e-5, rtol=1e-5)
    print("Dense V->E, dual E->V and top-k sparsity: PASS")


def backward_checks():
    module = DSBAdaHGConv(64, 8, 4, topk=4, eta_max=0.10)
    module.eta_raw.data.fill_(0.5)
    x = torch.randn(2, 400, 64, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape and finite(output)
    output.square().mean().backward()
    assert x.grad is not None and finite(x.grad)
    assert module.eta_raw.grad is not None and finite(module.eta_raw.grad)
    assert module.eta_raw.grad.abs().sum() > 0
    assert all(finite(parameter.grad) for parameter in module.parameters() if parameter.grad is not None)

    c3 = DSBC3AH(256, 256, 1.0, 8, "both", 4, 0.10)
    feature = torch.randn(1, 256, 20, 20, requires_grad=True)
    c3_output = c3(feature)
    assert c3_output.shape == feature.shape and finite(c3_output)
    c3_output.mean().backward()
    assert feature.grad is not None and finite(feature.grad)
    print(f"Backward/eta gradient and DSBC3AH shape: PASS (eta_grad={module.eta_raw.grad.item():.6g})")


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
    for stage, (filename, expected_dsb_branches, expected_topk) in CONFIGS.items():
        model = YOLO(str(cfg_root / filename)).model.to(device)
        modules = [module for module in model.modules() if isinstance(module, DSBHyperACE)]
        assert len(modules) == 1, stage
        hyper = modules[0]
        assert isinstance(hyper.fuse, FuseModule), stage
        actual_dsb = sum(not isinstance(branch, C3AH) for branch in (hyper.branch1, hyper.branch2))
        assert actual_dsb == expected_dsb_branches, (stage, actual_dsb)
        dsb_convs = [module for module in hyper.modules() if isinstance(module, DSBAdaHGConv)]
        assert len(dsb_convs) == expected_dsb_branches, stage
        assert all(module.topk == expected_topk for module in dsb_convs), stage
        assert all(torch.count_nonzero(module.eta_raw) == 0 for module in dsb_convs), stage
        assert len(hyper.m) == 1, f"{stage}: n-scale HyperACE low-order repeats must be 1"
        assert model.model[-1].nl == 3, stage

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
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
        print(f"{stage}: params={parameters:,}, Detect=3, forward/AMP/fuse/loss(empty+normal): PASS")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def full_model_identity_checks(root):
    cfg_root = root / "ultralytics/cfg/models/v13"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_configs = {
        "original": cfg_root / "yolov13-original.yaml",
        "faar": cfg_root / "yolov13_faar.yaml",
    }
    inputs = torch.randn(1, 3, 320, 320, device=device)
    for base_name, base_cfg in base_configs.items():
        base = YOLO(str(base_cfg)).model.to(device).eval()
        base_state = base.state_dict()
        candidates = (
            {stage: data for stage, data in CONFIGS.items() if stage != "k6_faar_dsb"}
            if base_name == "original"
            else {"k6_faar_dsb": CONFIGS["k6_faar_dsb"]}
        )
        with torch.no_grad():
            expected = base(inputs)
        for stage, (filename, expected_dsb_branches, _) in candidates.items():
            candidate = YOLO(str(cfg_root / filename)).model.to(device).eval()
            result = candidate.load_state_dict(base_state, strict=False)
            assert len(result.missing_keys) == expected_dsb_branches, (stage, result)
            assert all(key.endswith("eta_raw") for key in result.missing_keys), (stage, result)
            assert not result.unexpected_keys, (stage, result)
            with torch.no_grad():
                actual = candidate(inputs)
            difference = max_abs_diff(expected, actual)
            assert difference == 0.0, (stage, difference)
            print(f"{stage}: full-model zero-start identity vs {base_name}: PASS (max_diff=0)")
            del candidate
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(4)
    identity_check()
    propagation_checks()
    backward_checks()
    full_model_identity_checks(Path(__file__).resolve().parents[1])
    model_checks(Path(__file__).resolve().parents[1])
    print("All DSB-HyperACE self-checks: PASS")
