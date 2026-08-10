"""Unit and construction gates for the POD-YOLOv13 production modules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules.block import LGPDDown, OCFConcat
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, UDQDetect, _DistributionGuidedQuality


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"
VARIANT_CONFIGS = {
    "d0": "yolov13n-pod-d0.yaml",
    "p": "yolov13n-pod-p.yaml",
    "o": "yolov13n-pod-o.yaml",
    "d": "yolov13n-pod-d.yaml",
    "po": "yolov13n-pod-po.yaml",
    "pd": "yolov13n-pod-pd.yaml",
    "od": "yolov13n-pod-od.yaml",
    "pod": "yolov13n-pod.yaml",
}


def _centered_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float() - x.float().mean(dim=(2, 3), keepdim=True)
    return x.square().mean(dim=(2, 3), keepdim=True).add(eps).sqrt()


def test_lgpd_exact_endpoint_and_odd_shape() -> None:
    torch.manual_seed(1)
    module = LGPDDown(32, 32, {"base_groups": 4}).eval()
    x = torch.randn(2, 32, 81, 79)
    with torch.no_grad():
        assert torch.equal(module(x), module.base(x))
    assert module(x).shape[-2:] == (41, 40)


def test_ocf_exact_concat_endpoint() -> None:
    module = OCFConcat(32, 48).eval()
    deep, lateral = torch.randn(2, 32, 17, 15), torch.randn(2, 48, 17, 15)
    with torch.no_grad():
        assert torch.equal(module([deep, lateral]), torch.cat((deep, lateral), dim=1))


def test_two_step_gradients_open_for_zero_start_branches() -> None:
    lgpd = LGPDDown(32, 32, {"base_groups": 4}).train()
    optimizer = torch.optim.SGD(lgpd.parameters(), lr=0.1)
    gradients = []
    for _ in range(2):
        optimizer.zero_grad()
        lgpd(torch.randn(2, 32, 32, 32)).square().mean().backward()
        gradients.append(lgpd.branch_in[0].weight.grad.abs().sum().item())
        optimizer.step()
    assert gradients[0] == 0.0 and gradients[1] > 0.0

    ocf = OCFConcat(32, 48).train()
    optimizer = torch.optim.SGD(ocf.parameters(), lr=0.1)
    gradients = []
    for _ in range(2):
        optimizer.zero_grad()
        ocf([torch.randn(2, 32, 16, 16), torch.randn(2, 48, 16, 16)]).square().mean().backward()
        gradients.append(ocf.deep_proj[0].weight.grad.abs().sum().item())
        optimizer.step()
    assert gradients[0] == 0.0 and gradients[1] > 0.0

    quality = _DistributionGuidedQuality(32, 16).train()
    optimizer = torch.optim.SGD(quality.parameters(), lr=0.1)
    gradients = []
    for _ in range(2):
        optimizer.zero_grad()
        logits = quality(torch.randn(2, 32, 8, 8), torch.randn(2, 64, 8, 8))
        F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
        gradients.append(quality.feature_path[0].conv.weight.grad.abs().sum().item())
        optimizer.step()
    assert gradients[0] == 0.0 and gradients[1] > 0.0


def test_rng_isolation_and_rms_bounds() -> None:
    torch.manual_seed(777)
    _ = Conv(32, 32, 3, 2, 1, 4)
    control = torch.rand(8)
    torch.manual_seed(777)
    _ = LGPDDown(32, 32, {"base_groups": 4})
    assert torch.equal(control, torch.rand(8))

    torch.manual_seed(888)
    control = torch.rand(8)
    torch.manual_seed(888)
    _ = OCFConcat(32, 48)
    assert torch.equal(control, torch.rand(8))

    lgpd = LGPDDown(32, 32, {"base_groups": 4, "max_residual_ratio": 0.1}).eval()
    torch.nn.init.normal_(lgpd.branch_out.weight, 0, 1)
    x = torch.randn(2, 32, 32, 32)
    with torch.no_grad():
        base, output = lgpd.base(x), lgpd(x)
    assert (_centered_rms(output - base) / _centered_rms(base)).max().item() <= 0.101

    ocf = OCFConcat(32, 48, {"max_residual_ratio": 0.1}).eval()
    torch.nn.init.normal_(ocf.out.weight, 0, 1)
    deep, lateral = torch.randn(2, 32, 16, 16), torch.randn(2, 48, 16, 16)
    with torch.no_grad():
        output = ocf([deep, lateral])[:, :32]
    assert (_centered_rms(output - deep) / _centered_rms(deep)).max().item() <= 0.101


def test_distribution_stats_are_shift_invariant_and_detached() -> None:
    quality = _DistributionGuidedQuality(32, 16).train()
    box_logits = torch.randn(2, 64, 8, 8, requires_grad=True)
    common_shift = torch.randn(2, 4, 1, 8, 8).expand(-1, -1, 16, -1, -1).reshape_as(box_logits)
    stats_a, stats_b = quality._distribution_stats(box_logits), quality._distribution_stats(box_logits + common_shift)
    assert (stats_a - stats_b).abs().max().item() < 1e-5
    quality_loss = F.binary_cross_entropy_with_logits(
        quality(torch.randn(2, 32, 8, 8), box_logits), torch.rand(2, 1, 8, 8)
    )
    quality_loss.backward()
    assert box_logits.grad is None or box_logits.grad.abs().sum().item() == 0.0


def test_udq_preserves_detect_box_and_class_logits_at_initialization() -> None:
    features = [torch.randn(1, 64, 8, 8), torch.randn(1, 128, 4, 4), torch.randn(1, 256, 2, 2)]
    torch.manual_seed(20260809)
    reference = Detect(nc=4, ch=(64, 128, 256)).train()
    torch.manual_seed(20260809)
    candidate = UDQDetect(nc=4, ch=(64, 128, 256)).train()
    reference_outputs = reference([feature.clone() for feature in features])
    candidate_outputs = candidate([feature.clone() for feature in features])
    for reference_output, candidate_output in zip(reference_outputs, candidate_outputs):
        assert torch.equal(reference_output, candidate_output[:, :-1])


def test_cpu_bfloat16_autocast_is_finite() -> None:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = (
            LGPDDown(32, 32, {"base_groups": 4}).eval()(torch.randn(2, 32, 32, 32)),
            OCFConcat(32, 48).eval()([torch.randn(2, 32, 16, 16), torch.randn(2, 48, 16, 16)]),
            _DistributionGuidedQuality(32, 16).eval()(torch.randn(2, 32, 16, 16), torch.randn(2, 64, 16, 16)),
        )
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: LGPDDown(31, 32, {"base_groups": 4}),
        lambda: LGPDDown(32, 32, {"low_kernel": 4}),
        lambda: OCFConcat(32, 48, {"support_kernel": 4}),
        lambda: _DistributionGuidedQuality(32, 1),
    ),
)
def test_invalid_configs_fail_fast(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_all_factorial_configs_build() -> None:
    for name in VARIANT_CONFIGS.values():
        model = YOLO(str(CONFIG_DIR / name)).model
        assert model.model[-1].nc == 4
