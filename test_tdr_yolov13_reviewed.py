#!/usr/bin/env python3
"""Blocking integration tests for reviewed TDR-YOLOv13 modules and all T1--T7 YAMLs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ultralytics.nn.modules.block import DCRAUp, SADIP3Fuse, TIERDCRAUp
from ultralytics.nn.modules.head import BRDDetect, Detect, _BRDLogitAdapter


ROOT = Path(__file__).resolve().parent
TDR_YAMLS = (
    ROOT / "ultralytics/cfg/models/v13/yolov13n-tdr-t1-tier.yaml",
    ROOT / "ultralytics/cfg/models/v13/yolov13n-tdr-t2-sadi.yaml",
    ROOT / "ultralytics/cfg/models/v13/yolov13n-tdr-t3-brd.yaml",
    ROOT / "ultralytics/cfg/models/v13/yolov13n-tdr-t4-tier-sadi.yaml",
    ROOT / "ultralytics/cfg/models/v13/yolov13n-tdr-t5-tier-brd.yaml",
    ROOT / "ultralytics/cfg/models/v13/yolov13n-tdr-t6-sadi-brd.yaml",
    ROOT / "ultralytics/cfg/models/v13/yolov13n-tdr-t7-full.yaml",
)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values.")


def test_tier_endpoint_and_fallback() -> None:
    for seed in (0, 7, 19):
        torch.manual_seed(seed)
        parent = DCRAUp(64, 32, temperature=0.20).eval()
        tier = TIERDCRAUp(64, 32, {"reliability_mode": "entropy", "temperature": 0.20}).eval()
        tier.load_state_dict(parent.state_dict(), strict=True)
        deep, lateral = torch.randn(2, 64, 7, 9), torch.randn(2, 32, 14, 18)
        with torch.no_grad():
            assert torch.equal(parent([deep, lateral]), tier([deep, lateral]))

    module = TIERDCRAUp(32, 16, {"reliability_mode": "tri"}).train()
    deep, lateral = torch.randn(2, 32, 7, 9), torch.randn(2, 16, 14, 18)
    output = module([deep, lateral])
    assert torch.equal(output, F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float())
    output.square().mean().backward()
    assert module.residual_out.weight.grad is not None and module.residual_out.weight.grad.abs().sum() > 0
    original = module._phase_correlate_and_reassemble

    def inject_nonfinite(*args, **kwargs):
        reassembled, weights = original(*args, **kwargs)
        reassembled = reassembled.clone()
        reassembled[..., 0, 0] = float("nan")
        return reassembled, weights

    module._phase_correlate_and_reassemble = inject_nonfinite
    _, residual, _, reliability = module._compute_alignment(deep, lateral)
    _assert_finite("TIER fallback residual", residual)
    _assert_finite("TIER reliability", reliability)
    assert torch.equal(residual[..., 0, 0], torch.zeros_like(residual[..., 0, 0]))


def test_sadi_identity_bound_and_gradient() -> None:
    torch.manual_seed(2)
    module = SADIP3Fuse(32, 48).eval()
    p2, p3 = torch.randn(2, 32, 28, 36), torch.randn(2, 48, 14, 18)
    with torch.no_grad():
        assert torch.equal(module([p2, p3]), p3)
        assert module._bound_correction(p3, torch.full_like(p3, 100.0)).abs().max() == 0
    clone = SADIP3Fuse(32, 48).eval()
    clone.load_state_dict(module.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(module([p2, p3]), clone([p2, p3]))

    module = SADIP3Fuse(32, 48).train()
    result = module([p2, p3])
    result.square().mean().backward()
    assert module.detail_out.weight.grad is not None and module.detail_out.weight.grad.abs().sum() > 0
    _assert_finite("SADI output", result)


def test_brd_endpoint_distribution_and_gradient() -> None:
    channels = (32, 64, 128)
    torch.manual_seed(4)
    base = Detect(nc=4, ch=channels).train()
    torch.manual_seed(4)
    head = BRDDetect(nc=4, config={"level_strengths": [1.0, 0.5, 0.0]}, ch=channels).train()
    features = [torch.randn(2, 32, 24, 24), torch.randn(2, 64, 12, 12), torch.randn(2, 128, 6, 6)]
    expected = base([feature.clone() for feature in features])
    actual = head([feature.clone() for feature in features])
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(expected, actual))
    assert isinstance(head.box_refine[2], nn.Identity)

    sum(output.square().mean() for output in actual).backward()
    adapters = [module for module in head.box_refine if isinstance(module, _BRDLogitAdapter)]
    assert sum(module.out.weight.grad.abs().sum().item() for module in adapters) > 0
    with torch.no_grad():
        for feature, module in zip(features, adapters):
            delta = module(feature).float()
            shaped = delta.reshape(delta.shape[0], 4, module.reg_max, *delta.shape[-2:])
            assert shaped.mean(dim=2).abs().max() < 2e-6
            assert delta.abs().max() <= module.max_logit_delta * module.level_strength + 1e-6
            _assert_finite("BRD delta", delta)


def test_configs_and_yaml_contracts() -> None:
    invalid = ({"reduction": 0}, {"detail_kernel": 2}, {"support_kernel": 2}, {"max_logit_delta": 0.0})
    for config in invalid:
        try:
            BRDDetect(nc=4, config={**config, "level_strengths": [0.0, 0.0, 0.0]}, ch=(16, 32, 64))
        except ValueError:
            continue
        raise AssertionError(f"Invalid BRD config accepted: {config}")
    expected = ((True, False, False), (False, True, False), (False, False, True), (True, True, False),
                (True, False, True), (False, True, True), (True, True, True))
    for path, flags in zip(TDR_YAMLS, expected):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        modules = [layer[2] for section in ("backbone", "head") for layer in payload[section]]
        actual = tuple(name in modules for name in ("TIERDCRAUp", "SADIP3Fuse", "BRDDetect"))
        assert actual == flags, (path, actual, flags)


def test_yaml_parser(imgsz: int) -> None:
    from ultralytics.nn.tasks import DetectionModel

    for path in TDR_YAMLS:
        model = DetectionModel(str(path), ch=3, nc=4, verbose=False).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, imgsz, imgsz))
        decoded = output[0] if isinstance(output, tuple) else output
        _assert_finite(path.name, decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", action="store_true", help="Build and forward all T1--T7 models.")
    parser.add_argument("--imgsz", type=int, default=128)
    args = parser.parse_args()
    test_tier_endpoint_and_fallback()
    test_sadi_identity_bound_and_gradient()
    test_brd_endpoint_distribution_and_gradient()
    test_configs_and_yaml_contracts()
    if args.yaml:
        test_yaml_parser(args.imgsz)
    print("ALL REVIEWED TDR-YOLOv13 BLOCKING TESTS PASSED")


if __name__ == "__main__":
    main()
