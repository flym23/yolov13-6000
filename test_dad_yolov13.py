#!/usr/bin/env python3
"""Blocking repository-level tests for DAD-YOLOv13 integration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ultralytics.nn.modules.block import DCRAUp, DDFCalib, TIERDCRAUp
from ultralytics.nn.modules.head import Detect, SDDCDetect, _SDDCTaskAdapter


DAD_CONFIGS = (
    (Path("ultralytics/cfg/models/v13/yolov13-dad-a1-ddfc.yaml"), True, False, False),
    (Path("ultralytics/cfg/models/v13/yolov13-dad-a2-tier.yaml"), False, True, False),
    (Path("ultralytics/cfg/models/v13/yolov13-dad-a3-sddc.yaml"), False, False, True),
    (Path("ultralytics/cfg/models/v13/yolov13-dad-a4-ddfc-tier.yaml"), True, True, False),
    (Path("ultralytics/cfg/models/v13/yolov13-dad-a5-tier-sddc.yaml"), False, True, True),
    (Path("ultralytics/cfg/models/v13/yolov13-dad-a6-ddfc-sddc.yaml"), True, False, True),
    (Path("ultralytics/cfg/models/v13/yolov13-dad-a7-full.yaml"), True, True, True),
)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values.")


def _inputs(channels=64, lateral_channels=32, height=7, width=9):
    return torch.randn(2, channels, height, width), torch.randn(2, lateral_channels, height * 2, width * 2)


def test_ddfc_identity_reload_and_two_step_gradient() -> None:
    torch.manual_seed(0)
    module = DDFCalib(64, 64).eval()
    x = torch.randn(2, 64, 31, 29)
    with torch.no_grad():
        assert torch.equal(module(x), x)
    clone = DDFCalib(64, 64).eval()
    clone.load_state_dict(module.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(module(x), clone(x))

    module = DDFCalib(32, 32).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    x = torch.randn(2, 32, 17, 19)
    module(x).square().mean().backward()
    scale_grad = module.low_scale.grad.abs().sum() + module.detail_scale.grad.abs().sum()
    branch_grad = sum(
        0.0 if parameter.grad is None else parameter.grad.abs().sum().item()
        for name, parameter in module.named_parameters()
        if "branch" in name
    )
    assert scale_grad.item() > 0.0 and branch_grad == 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(x).square().mean().backward()
    branch_grad = sum(
        0.0 if parameter.grad is None else parameter.grad.abs().sum().item()
        for name, parameter in module.named_parameters()
        if "branch" in name
    )
    assert branch_grad > 0.0


def test_tier_endpoint_reliability_and_two_step_gradient() -> None:
    torch.manual_seed(1)
    parent = DCRAUp(
        64,
        32,
        scale=2,
        kernel_size=3,
        reduction=4,
        temperature=0.20,
        residual_groups=4,
        use_entropy=True,
        use_lateral_guidance=True,
        detach_confidence=True,
        strict_scale=True,
    ).eval()
    tier = TIERDCRAUp(64, 32, {"reliability_mode": "entropy", "temperature": 0.20}).eval()
    tier.load_state_dict(parent.state_dict(), strict=True)
    deep, lateral = _inputs()
    with torch.no_grad():
        assert torch.equal(parent([deep, lateral]), tier([deep, lateral]))

    tri = TIERDCRAUp(64, 32, {"reliability_mode": "tri"}).eval()
    with torch.no_grad():
        base, residual, weights, reliability = tri._compute_alignment(deep, lateral)
        output = tri([deep, lateral])
    assert base.shape == residual.shape == (2, 64, 14, 18)
    assert weights.shape == (2, 9, 14, 18) and reliability.shape == (2, 1, 14, 18)
    assert reliability.min().item() >= 0.0 and reliability.max().item() <= 1.0
    _assert_finite("TIER residual", residual)
    _assert_finite("TIER reliability", reliability)
    assert torch.equal(output, F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float())

    tri = TIERDCRAUp(64, 32, {"reliability_mode": "tri"}).train()
    optimizer = torch.optim.SGD(tri.parameters(), lr=0.05)
    tri([deep, lateral]).square().mean().backward()
    assert tri.residual_out.weight.grad is not None and torch.count_nonzero(tri.residual_out.weight.grad).item() > 0
    assert tri.key_proj.weight.grad is None or torch.count_nonzero(tri.key_proj.weight.grad).item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    tri([deep, lateral]).square().mean().backward()
    assert torch.count_nonzero(tri.key_proj.weight.grad).item() > 0
    assert torch.count_nonzero(tri.query_proj.weight.grad).item() > 0


def test_sddc_endpoint_bound_and_two_step_gradient() -> None:
    torch.manual_seed(2)
    channels = (32, 64, 128)
    base = Detect(nc=4, ch=channels).train()
    head = SDDCDetect(nc=4, ch=channels).train()
    head.cv2.load_state_dict(base.cv2.state_dict(), strict=True)
    head.cv3.load_state_dict(base.cv3.state_dict(), strict=True)
    features = [torch.randn(2, 32, 24, 24), torch.randn(2, 64, 12, 12), torch.randn(2, 128, 6, 6)]
    expected = base([feature.clone() for feature in features])
    actual = head([feature.clone() for feature in features])
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))

    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    sum(output.square().mean() for output in head([feature.clone() for feature in features])).backward()
    adapters = list(head.box_adapters) + list(head.cls_adapters)
    assert sum(adapter.pointwise.weight.grad.abs().sum().item() for adapter in adapters) > 0.0
    assert sum(
        0.0 if adapter.depthwise.weight.grad is None else adapter.depthwise.weight.grad.abs().sum().item()
        for adapter in adapters
    ) == 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sum(output.square().mean() for output in head([feature.clone() for feature in features])).backward()
    assert sum(adapter.depthwise.weight.grad.abs().sum().item() for adapter in adapters) > 0.0

    adapter = _SDDCTaskAdapter(16, "detail", gain=1.0, max_residual=0.10).eval()
    with torch.no_grad():
        adapter.pointwise.weight.normal_(0.0, 0.5)
        adapter.pointwise.bias.normal_(0.0, 0.5)
        x = torch.randn(2, 16, 13, 17)
        delta = adapter(x) - x
        delta_rms = delta.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
        feature_rms = x.float().square().mean(dim=(2, 3), keepdim=True).add(adapter.eps).sqrt()
        assert torch.all(delta_rms <= adapter.max_residual * feature_rms + 1e-6)


def test_invalid_configs_and_yaml_contracts() -> None:
    for kwargs in ({"reduction": 0}, {"context_kernel": 2}, {"max_gain": 0.0}):
        try:
            DDFCalib(16, 16, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Invalid DDFCalib config was accepted: {kwargs}")
    for config in (
        {"reliability_mode": "invalid"},
        {"margin_power": 0.0},
        {"consensus_kernel": 2},
        {"detail_kernel": 4},
        {"unknown": 1},
    ):
        try:
            TIERDCRAUp(32, 16, config)
        except ValueError:
            continue
        raise AssertionError(f"Invalid TIER config was accepted: {config}")

    for path, expect_ddfc, expect_tier, expect_sddc in DAD_CONFIGS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        modules = [layer[2] for section in ("backbone", "head") for layer in payload[section]]
        assert ("DDFCalib" in modules) is expect_ddfc, path
        assert ("TIERDCRAUp" in modules) is expect_tier, path
        assert ("SDDCDetect" in modules) is expect_sddc, path
        if expect_tier:
            tier_layer = next(layer for layer in payload["head"] if layer[2] == "TIERDCRAUp")
            assert set(tier_layer[3][0]) == set(TIERDCRAUp._DEFAULT_CONFIG)


def test_yaml_parser(imgsz: int) -> None:
    from ultralytics.nn.tasks import DetectionModel

    for path, _, _, _ in DAD_CONFIGS:
        model = DetectionModel(str(path), ch=3, nc=4, verbose=False).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, imgsz, imgsz))
        decoded = output[0] if isinstance(output, tuple) else output
        _assert_finite(path.name, decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", action="store_true", help="Build and forward all seven YAML ablations.")
    parser.add_argument("--imgsz", type=int, default=128)
    args = parser.parse_args()
    test_ddfc_identity_reload_and_two_step_gradient()
    test_tier_endpoint_reliability_and_two_step_gradient()
    test_sddc_endpoint_bound_and_two_step_gradient()
    test_invalid_configs_and_yaml_contracts()
    if args.yaml:
        test_yaml_parser(args.imgsz)
    print("ALL DAD-YOLOv13 BLOCKING TESTS PASSED")


if __name__ == "__main__":
    main()
