from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules.block import SADIP3Fuse, TIERDCRAUp
from ultralytics.nn.modules.head import Detect, EBDRDetect, _EBDRLogitAdapter


def _centered_rms(x, eps=1e-6):
    x = x.float() - x.float().mean(dim=(2, 3), keepdim=True)
    return x.square().mean(dim=(2, 3), keepdim=True).add(eps).sqrt()


def test_tier_zero_start_is_exact_nearest():
    torch.manual_seed(7)
    module = TIERDCRAUp(64, 32)
    deep, lateral = torch.randn(2, 64, 8, 8), torch.randn(2, 32, 16, 16)
    torch.testing.assert_close(module([deep, lateral]), F.interpolate(deep, size=lateral.shape[-2:], mode="nearest"), rtol=0.0, atol=0.0)


def test_sadi_zero_start_is_exact_p3_identity():
    torch.manual_seed(7)
    module = SADIP3Fuse(32, 64)
    p2, p3 = torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16)
    torch.testing.assert_close(module([p2, p3]), p3, rtol=0.0, atol=0.0)


def test_ebdr_zero_start_and_shift_invariant_uncertainty():
    torch.manual_seed(7)
    module = _EBDRLogitAdapter(64, 64)
    feature, logits = torch.randn(2, 64, 16, 16), torch.randn(2, 64, 16, 16)
    torch.testing.assert_close(module(feature, logits), torch.zeros_like(logits), rtol=0.0, atol=0.0)
    torch.testing.assert_close(module._uncertainty(logits), module._uncertainty(logits + 9.75), rtol=1e-5, atol=1e-6)


def test_ebdr_head_is_detect_equivalent_at_initialization():
    torch.manual_seed(11)
    baseline, improved = Detect(nc=4, ch=(64, 128, 256)), EBDRDetect(nc=4, config={}, ch=(64, 128, 256))
    improved.load_state_dict(baseline.state_dict(), strict=False)
    baseline.train(), improved.train()
    features = [torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8), torch.randn(2, 256, 4, 4)]
    for expected, actual in zip(baseline([x.clone() for x in features]), improved([x.clone() for x in features])):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_module_construction_does_not_perturb_global_rng():
    for constructor in (lambda: TIERDCRAUp(64, 32), lambda: SADIP3Fuse(32, 64), lambda: _EBDRLogitAdapter(64, 64)):
        torch.manual_seed(12345)
        state = torch.random.get_rng_state()
        constructor()
        observed = torch.rand(8)
        torch.random.set_rng_state(state)
        torch.testing.assert_close(observed, torch.rand(8), rtol=0.0, atol=0.0)


def test_zero_start_final_projections_receive_gradients_and_open_upstream_branches():
    torch.manual_seed(13)
    tier, sadi, ebdr = TIERDCRAUp(64, 32), SADIP3Fuse(32, 64), _EBDRLogitAdapter(64, 64)
    optimizer = torch.optim.SGD([*tier.parameters(), *sadi.parameters(), *ebdr.parameters()], lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = torch.randn(2, 64, 8, 8), torch.randn(2, 32, 16, 16)
        p2, p3 = torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16)
        feature, logits = torch.randn(2, 64, 16, 16), torch.randn(2, 64, 16, 16)
        y_t, y_s, delta = tier([deep, lateral]), sadi([p2, p3]), ebdr(feature, logits)
        loss = (
            (y_t * torch.randn_like(y_t)).mean()
            + (y_s * torch.randn_like(y_s)).mean()
            + ((logits + delta) * torch.randn_like(logits)).mean()
        )
        loss.backward()
        optimizer.step()
    assert tier.residual_out.weight.grad is not None and tier.residual_out.weight.grad.abs().sum() > 0
    assert sadi.detail_out.weight.grad is not None and sadi.detail_out.weight.grad.abs().sum() > 0
    assert ebdr.out.weight.grad is not None and ebdr.out.weight.grad.abs().sum() > 0
    assert tier.query_proj.weight.grad is not None and tier.query_proj.weight.grad.abs().sum() > 0
    assert sadi.p2_proj[0].weight.grad is not None and sadi.p2_proj[0].weight.grad.abs().sum() > 0
    assert ebdr.depthwise.weight.grad is not None and ebdr.depthwise.weight.grad.abs().sum() > 0


def test_rms_bounds_and_invalid_shapes():
    tier, sadi = TIERDCRAUp(64, 32), SADIP3Fuse(32, 64)
    torch.nn.init.normal_(tier.residual_out.weight, std=2.0)
    torch.nn.init.normal_(sadi.detail_out.weight, std=2.0)
    deep, lateral = torch.randn(2, 64, 8, 8), torch.randn(2, 32, 16, 16)
    p2, p3 = torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16)
    assert (_centered_rms(tier([deep, lateral]) - F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")) / _centered_rms(deep)).max() <= 0.1501
    assert (_centered_rms(sadi([p2, p3]) - p3) / _centered_rms(p3)).max() <= 0.1201
    with pytest.raises(ValueError):
        tier([torch.randn(1, 64, 8, 8), torch.randn(1, 32, 15, 16)])
    with pytest.raises(ValueError):
        sadi([torch.randn(1, 32, 31, 32), torch.randn(1, 64, 16, 16)])


@pytest.mark.slow
def test_full_tdr_yaml_build_and_forward():
    path = Path("ultralytics/cfg/models/v13/yolov13-tdr.yaml")
    model = YOLO(str(path)).model.train()
    outputs = model(torch.randn(1, 3, 640, 640))
    assert isinstance(outputs, list) and len(outputs) == 3 and all(torch.isfinite(x).all() for x in outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP validation.")
def test_cuda_amp_forward_backward_is_finite():
    tier, sadi, ebdr = TIERDCRAUp(64, 32).cuda().train(), SADIP3Fuse(32, 64).cuda().train(), _EBDRLogitAdapter(64, 64).cuda().train()
    with torch.autocast("cuda", dtype=torch.float16):
        y_t = tier([torch.randn(2, 64, 8, 8, device="cuda"), torch.randn(2, 32, 16, 16, device="cuda")])
        y_s = sadi([torch.randn(2, 32, 32, 32, device="cuda"), torch.randn(2, 64, 16, 16, device="cuda")])
        delta = ebdr(torch.randn(2, 64, 16, 16, device="cuda"), torch.randn(2, 64, 16, 16, device="cuda"))
        loss = y_t.square().mean() + y_s.square().mean() + delta.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
