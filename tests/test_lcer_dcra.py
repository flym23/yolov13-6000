"""Mechanism, endpoint, parser, and numerical tests for LCER-DCRA."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import LCERDCRAUp, MEDCRAUp, URRDCRAUp


MODEL_YAMLS = {
    "l0_baseline": "ultralytics/cfg/models/v13/yolov13-lcer-dcra-l0.yaml",
    "l1_strict_rho020": "ultralytics/cfg/models/v13/yolov13-lcer-dcra-l1-strict.yaml",
    "l2_channel_power20": "ultralytics/cfg/models/v13/yolov13-lcer-dcra-l2-channel.yaml",
    "l3_local_consensus": "ultralytics/cfg/models/v13/yolov13-lcer-dcra-l3-local.yaml",
}


def make_inputs(seed=0, batch=2, deep_size=(3, 4), device="cpu"):
    generator = torch.Generator(device=device).manual_seed(seed)
    deep = torch.randn(batch, 16, *deep_size, generator=generator, device=device)
    lateral = torch.randn(batch, 8, deep_size[0] * 2, deep_size[1] * 2, generator=generator, device=device)
    return deep, lateral


def grad_sum(parameter):
    return 0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum())


def module(config=None):
    return LCERDCRAUp(c_deep=16, c_lateral=8, config=config).eval()


def endpoint_modules():
    torch.manual_seed(123)
    m2 = MEDCRAUp(c_deep=16, c_lateral=8, use_energy_bound=True, max_residual_ratio=0.20).eval()
    m5 = MEDCRAUp(c_deep=16, c_lateral=8, use_energy_bound=False, max_residual_ratio=0.20).eval()
    strict = module({"release_mode": "strict", "strict_ratio": 0.20})
    none = module({"release_mode": "none", "strict_ratio": 0.20})
    with torch.no_grad():
        strict.residual_out.weight.normal_(mean=0.0, std=0.20)
    state = strict.state_dict()
    m2.load_state_dict(state, strict=True)
    m5.load_state_dict(state, strict=True)
    none.load_state_dict(state, strict=True)
    return m2, m5, strict, none


def test_parameter_buffer_and_state_key_compatibility():
    medcra = MEDCRAUp(c_deep=16, c_lateral=8)
    lcer = module()
    assert list(medcra.state_dict()) == list(lcer.state_dict())
    assert sum(parameter.numel() for parameter in medcra.parameters()) == sum(parameter.numel() for parameter in lcer.parameters())
    assert dict(medcra.named_buffers()).keys() == dict(lcer.named_buffers()).keys()


def test_initial_output_is_exact_nearest():
    lcer = module()
    deep, lateral = make_inputs(seed=1)
    with torch.no_grad():
        output = lcer([deep, lateral])
    assert torch.equal(output, F.interpolate(deep, size=lateral.shape[-2:], mode="nearest"))


def test_strict_endpoint_is_exact_medcra_m2():
    m2, _, strict, _ = endpoint_modules()
    deep, lateral = make_inputs(seed=2)
    with torch.no_grad():
        assert torch.equal(m2([deep, lateral]), strict([deep, lateral]))


def test_none_endpoint_is_exact_medcra_m5():
    _, m5, _, none = endpoint_modules()
    deep, lateral = make_inputs(seed=3)
    with torch.no_grad():
        assert torch.equal(m5([deep, lateral]), none([deep, lateral]))


def test_channel_endpoint_is_exact_urr_power2():
    torch.manual_seed(124)
    urr = URRDCRAUp(c_deep=16, c_lateral=8, release_mode="adaptive", strict_ratio=0.10, release_power=2.0).eval()
    lcer = module({"release_mode": "channel", "strict_ratio": 0.10, "channel_power": 2.0})
    lcer.load_state_dict(urr.state_dict(), strict=True)
    with torch.no_grad():
        lcer.residual_out.weight.normal_(mean=0.0, std=0.20)
    urr.load_state_dict(lcer.state_dict(), strict=True)
    deep, lateral = make_inputs(seed=4)
    with torch.no_grad():
        assert torch.equal(urr([deep, lateral]), lcer([deep, lateral]))


def test_local_release_is_between_strict_and_channel_per_element():
    strict = module({"release_mode": "strict"})
    channel = module({"release_mode": "channel"})
    local = module({"release_mode": "local"})
    base = torch.randn(2, 16, 6, 8)
    correction = torch.randn_like(base)
    confidence = torch.rand(2, 1, 6, 8)
    with torch.no_grad():
        strict_correction = strict._route_output_correction(base, correction, confidence)
        channel_correction = channel._route_output_correction(base, correction, confidence)
        local_correction = local._route_output_correction(base, correction, confidence)
    assert torch.all(local_correction.abs() + 2e-6 >= strict_correction.abs())
    assert torch.all(local_correction.abs() <= channel_correction.abs() + 2e-6)


def test_confidence_extremes_recover_local_endpoints():
    _, _, strict, _ = endpoint_modules()
    channel = module({"release_mode": "channel", "strict_ratio": 0.20})
    local = module({"release_mode": "local", "strict_ratio": 0.20})
    channel.load_state_dict(strict.state_dict(), strict=True)
    local.load_state_dict(strict.state_dict(), strict=True)
    deep, lateral = make_inputs(seed=5)
    with torch.no_grad():
        base, residual, _, confidence = local._compute_alignment(deep, lateral)
        correction = local.residual_out(local._moment_preserving_residual(base, residual))
        zero = local._route_output_correction(base, correction, torch.zeros_like(confidence))
        one = local._route_output_correction(base, correction, torch.ones_like(confidence))
        strict_correction = strict._route_output_correction(base, correction, confidence)
        channel_correction = channel._route_output_correction(base, correction, confidence)
    assert torch.equal(zero, strict_correction)
    assert torch.equal(one, channel_correction)


def test_local_consensus_suppresses_an_isolated_confident_pixel():
    lcer = module({"release_mode": "local", "consensus_kernel": 3})
    isolated = torch.zeros(1, 1, 7, 7)
    isolated[..., 3, 3] = 1.0
    block = torch.zeros(1, 1, 7, 7)
    block[..., 2:5, 2:5] = 1.0
    reference = torch.zeros(1, 16, 7, 7)
    isolated_release = lcer._compute_spatial_consensus(isolated, reference)
    block_release = lcer._compute_spatial_consensus(block, reference)
    assert isolated_release[..., 3, 3] < block_release[..., 3, 3]
    assert torch.isclose(block_release[..., 3, 3], torch.ones(1)).all()


def test_two_step_gradient_activation():
    torch.manual_seed(7)
    lcer = LCERDCRAUp(c_deep=16, c_lateral=8).train()
    optimizer = torch.optim.SGD(lcer.parameters(), lr=0.20, momentum=0.0, weight_decay=0.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs(seed=8 + step)
        deep.requires_grad_(True)
        lateral.requires_grad_(True)
        output = lcer([deep, lateral])
        (output * torch.randn_like(output)).mean().backward()
        assert grad_sum(lcer.residual_out.weight) > 0.0
        if step == 0:
            assert grad_sum(lcer.key_proj.weight) == 0.0
            assert grad_sum(lcer.query_proj.weight) == 0.0
        else:
            assert grad_sum(lcer.key_proj.weight) > 0.0
            assert grad_sum(lcer.query_proj.weight) > 0.0
            assert lateral.grad is not None and torch.isfinite(lateral.grad).all() and lateral.grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize(
    "config",
    (
        {"release_mode": "invalid"},
        {"strict_ratio": 0.0},
        {"strict_ratio": 1.1},
        {"channel_power": 0.0},
        {"spatial_power": 0.0},
        {"consensus_kernel": 2},
        {"unknown": True},
    ),
)
def test_invalid_constructor_config(config):
    with pytest.raises(ValueError):
        module(config)


@pytest.mark.parametrize("stage,yaml_path", tuple(MODEL_YAMLS.items()))
def test_yaml_build_and_topology(stage, yaml_path):
    model = YOLO(yaml_path).model.eval()
    if stage == "l0_baseline":
        assert isinstance(model.model[15], nn.Upsample)
        assert sum(isinstance(item, LCERDCRAUp) for item in model.modules()) == 0
    else:
        assert isinstance(model.model[15], LCERDCRAUp)
        assert model.model[15].f == [14, 12]
        assert sum(isinstance(item, LCERDCRAUp) for item in model.modules()) == 1
    # Layer 18 remains FullPAD_Tunnel; P4→P3 nearest upsample is unchanged at layer 19.
    assert isinstance(model.model[19], nn.Upsample)
    assert model.model[-1].f == [23, 27, 31]
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    tensors = output if isinstance(output, (list, tuple)) else (output,)
    assert all(torch.isfinite(tensor).all() for tensor in tensors if isinstance(tensor, torch.Tensor))


def test_l1_strict_model_state_compatibility():
    m2 = YOLO("ultralytics/cfg/models/v13/yolov13-medcra-rho020.yaml").model.eval()
    l1 = YOLO(MODEL_YAMLS["l1_strict_rho020"]).model.eval()
    l1.load_state_dict(m2.state_dict(), strict=True)
    assert sum(parameter.numel() for parameter in m2.parameters()) == sum(parameter.numel() for parameter in l1.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP validation.")
def test_cuda_amp_two_step_and_strict_reload(tmp_path: Path):
    torch.manual_seed(9)
    lcer = LCERDCRAUp(c_deep=16, c_lateral=8).cuda().train()
    optimizer = torch.optim.SGD(lcer.parameters(), lr=0.20, momentum=0.0, weight_decay=0.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs(seed=10 + step, device="cuda")
        with torch.cuda.amp.autocast(enabled=True):
            output = lcer([deep, lateral])
            loss = (output.float() * torch.randn_like(output.float())).mean()
        assert torch.isfinite(output).all() and torch.isfinite(loss)
        loss.backward()
        assert grad_sum(lcer.residual_out.weight) > 0.0
        optimizer.step()
    checkpoint = tmp_path / "lcer_dcra_state.pt"
    torch.save(lcer.state_dict(), checkpoint)
    clone = LCERDCRAUp(c_deep=16, c_lateral=8).cuda().eval()
    clone.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    deep, lateral = make_inputs(seed=12, device="cuda")
    with torch.no_grad():
        assert torch.equal(lcer.eval()([deep, lateral]), clone([deep, lateral]))
