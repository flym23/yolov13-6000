"""Mechanism, endpoint, parser, and numerical tests for URR-DCRA."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import MEDCRAUp, URRDCRAUp


MODEL_YAMLS = (
    "ultralytics/cfg/models/v13/yolov13-urr-dcra.yaml",
    "ultralytics/cfg/models/v13/yolov13-urr-dcra-mean.yaml",
    "ultralytics/cfg/models/v13/yolov13-urr-dcra-power05.yaml",
    "ultralytics/cfg/models/v13/yolov13-urr-dcra-power20.yaml",
    "ultralytics/cfg/models/v13/yolov13-urr-dcra-strict.yaml",
    "ultralytics/cfg/models/v13/yolov13-urr-dcra-none.yaml",
)


def make_inputs(seed=0, batch=2, deep_size=(3, 4), device="cpu"):
    generator = torch.Generator(device=device).manual_seed(seed)
    deep = torch.randn(batch, 16, *deep_size, generator=generator, device=device)
    lateral = torch.randn(batch, 8, deep_size[0] * 2, deep_size[1] * 2, generator=generator, device=device)
    return deep, lateral


def spatial_rms(x):
    return x.float().square().mean(dim=(2, 3), keepdim=True).sqrt()


def grad_sum(parameter):
    return 0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum())


def endpoint_modules():
    torch.manual_seed(123)
    m2 = MEDCRAUp(c_deep=16, c_lateral=8, use_energy_bound=True, max_residual_ratio=0.10).eval()
    m5 = MEDCRAUp(c_deep=16, c_lateral=8, use_energy_bound=False, max_residual_ratio=0.10).eval()
    strict = URRDCRAUp(c_deep=16, c_lateral=8, release_mode="strict", strict_ratio=0.10).eval()
    none = URRDCRAUp(c_deep=16, c_lateral=8, release_mode="none", strict_ratio=0.10).eval()
    with torch.no_grad():
        strict.residual_out.weight.normal_(mean=0.0, std=0.20)
    state = strict.state_dict()
    m2.load_state_dict(state, strict=True)
    m5.load_state_dict(state, strict=True)
    none.load_state_dict(state, strict=True)
    return m2, m5, strict, none


def test_parameter_and_buffer_compatibility():
    medcra = MEDCRAUp(c_deep=16, c_lateral=8)
    urr = URRDCRAUp(c_deep=16, c_lateral=8)
    assert list(medcra.state_dict()) == list(urr.state_dict())
    assert sum(parameter.numel() for parameter in medcra.parameters()) == sum(
        parameter.numel() for parameter in urr.parameters()
    )
    assert dict(medcra.named_buffers()).keys() == dict(urr.named_buffers()).keys()


def test_initial_output_is_exact_nearest():
    module = URRDCRAUp(c_deep=16, c_lateral=8).eval()
    deep, lateral = make_inputs(seed=1)
    with torch.no_grad():
        output = module([deep, lateral])
    assert torch.equal(output, F.interpolate(deep, size=lateral.shape[-2:], mode="nearest"))


def test_strict_endpoint_is_exact_m2():
    m2, _, strict, _ = endpoint_modules()
    deep, lateral = make_inputs(seed=2)
    with torch.no_grad():
        assert torch.equal(m2([deep, lateral]), strict([deep, lateral]))


def test_none_endpoint_is_exact_m5():
    _, m5, _, none = endpoint_modules()
    deep, lateral = make_inputs(seed=3)
    with torch.no_grad():
        assert torch.equal(m5([deep, lateral]), none([deep, lateral]))


def test_confidence_extremes_recover_endpoints():
    _, _, strict, none = endpoint_modules()
    adaptive = URRDCRAUp(c_deep=16, c_lateral=8, release_mode="adaptive").eval()
    adaptive.load_state_dict(strict.state_dict(), strict=True)
    deep, lateral = make_inputs(seed=4)
    with torch.no_grad():
        base, residual, _, confidence = adaptive._compute_alignment(deep, lateral)
        correction = adaptive.residual_out(adaptive._moment_preserving_residual(base, residual))
        zero = adaptive._route_output_correction(base, correction, torch.zeros_like(confidence))
        one = adaptive._route_output_correction(base, correction, torch.ones_like(confidence))
        strict_correction = strict._route_output_correction(base, correction, confidence)
        none_correction = none._route_output_correction(base, correction, confidence)
    assert torch.equal(zero, strict_correction)
    assert torch.equal(one, none_correction)


def test_energy_weighted_reliability_uses_active_region():
    module = URRDCRAUp(c_deep=2, c_lateral=2, energy_weighted_release=True).eval()
    confidence = torch.zeros(1, 1, 4, 4)
    confidence[..., :2, :2] = 1.0
    correction = torch.zeros(1, 2, 4, 4)
    correction[:, 0, :2, :2] = 2.0
    correction[:, 1, 2:, 2:] = 2.0
    reliability = module._compute_channel_reliability(confidence, correction)
    assert reliability.shape == (1, 2, 1, 1)
    assert reliability[0, 0, 0, 0] > 0.99
    assert reliability[0, 1, 0, 0] < 0.01


@pytest.mark.parametrize("release_power", (0.5, 1.0, 2.0))
def test_adaptive_correction_stays_between_endpoints(release_power):
    _, _, strict, none = endpoint_modules()
    adaptive = URRDCRAUp(c_deep=16, c_lateral=8, release_mode="adaptive", release_power=release_power).eval()
    adaptive.load_state_dict(strict.state_dict(), strict=True)
    deep, lateral = make_inputs(seed=5)
    base = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    with torch.no_grad():
        adaptive_rms = spatial_rms(adaptive([deep, lateral]) - base)
        strict_rms = spatial_rms(strict([deep, lateral]) - base)
        none_rms = spatial_rms(none([deep, lateral]) - base)
    assert torch.all(adaptive_rms + 2e-5 >= strict_rms)
    assert torch.all(adaptive_rms <= none_rms + 2e-5)


def test_centered_correction_has_zero_spatial_mean():
    module = URRDCRAUp(c_deep=16, c_lateral=8, center_correction=True).eval()
    with torch.no_grad():
        module.residual_out.weight.normal_(mean=0.0, std=0.20)
    deep, lateral = make_inputs(seed=6)
    with torch.no_grad():
        correction = module([deep, lateral]) - F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    assert correction.float().mean(dim=(2, 3)).abs().max() < 2e-5


def test_release_power_ordering():
    confidence = torch.full((1, 1, 4, 4), 0.25)
    correction = torch.ones(1, 2, 4, 4)
    reliabilities = [
        URRDCRAUp(c_deep=2, c_lateral=2, release_power=power)
        ._compute_channel_reliability(confidence, correction)
        .mean()
        for power in (0.5, 1.0, 2.0)
    ]
    assert reliabilities[0] > reliabilities[1] > reliabilities[2]


def test_two_step_gradient_activation():
    torch.manual_seed(7)
    module = URRDCRAUp(c_deep=16, c_lateral=8).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.20, momentum=0.0, weight_decay=0.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs(seed=8 + step)
        deep.requires_grad_(True)
        lateral.requires_grad_(True)
        output = module([deep, lateral])
        (output * torch.randn_like(output)).mean().backward()
        assert grad_sum(module.residual_out.weight) > 0.0
        if step == 0:
            assert grad_sum(module.key_proj.weight) == 0.0
            assert grad_sum(module.query_proj.weight) == 0.0
        else:
            assert grad_sum(module.key_proj.weight) > 0.0
            assert grad_sum(module.query_proj.weight) > 0.0
            assert lateral.grad is not None and torch.isfinite(lateral.grad).all() and lateral.grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize(
    "kwargs",
    ({"release_mode": "invalid"}, {"strict_ratio": 0.0}, {"strict_ratio": 1.1}, {"release_power": 0.0}),
)
def test_invalid_constructor_arguments(kwargs):
    with pytest.raises(ValueError):
        URRDCRAUp(c_deep=16, c_lateral=8, **kwargs)


@pytest.mark.parametrize("yaml_path", MODEL_YAMLS)
def test_yaml_build_and_topology(yaml_path):
    model = YOLO(yaml_path).model.eval()
    assert isinstance(model.model[15], URRDCRAUp)
    # Layer 18 remains FullPAD_Tunnel; the unmodified P4→P3 nearest upsample is layer 19.
    assert isinstance(model.model[19], nn.Upsample)
    assert model.model[15].f == [14, 12]
    assert model.model[-1].f == [23, 27, 31]
    assert sum(isinstance(module, URRDCRAUp) for module in model.modules()) == 1
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    tensors = output if isinstance(output, (list, tuple)) else (output,)
    assert all(torch.isfinite(tensor).all() for tensor in tensors if isinstance(tensor, torch.Tensor))


def test_full_model_state_compatibility():
    m2 = YOLO("ultralytics/cfg/models/v13/yolov13-medcra.yaml").model.eval()
    m5 = YOLO("ultralytics/cfg/models/v13/yolov13-medcra-no-bound.yaml").model.eval()
    strict = YOLO("ultralytics/cfg/models/v13/yolov13-urr-dcra-strict.yaml").model.eval()
    none = YOLO("ultralytics/cfg/models/v13/yolov13-urr-dcra-none.yaml").model.eval()
    strict.load_state_dict(m2.state_dict(), strict=True)
    none.load_state_dict(m5.state_dict(), strict=True)
    assert sum(parameter.numel() for parameter in m2.parameters()) == sum(
        parameter.numel() for parameter in strict.parameters()
    )
    assert sum(parameter.numel() for parameter in m5.parameters()) == sum(
        parameter.numel() for parameter in none.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP validation.")
def test_cuda_amp_two_step_and_strict_reload(tmp_path):
    torch.manual_seed(9)
    module = URRDCRAUp(c_deep=16, c_lateral=8).cuda().train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.20, momentum=0.0, weight_decay=0.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs(seed=10 + step, device="cuda")
        with torch.cuda.amp.autocast(enabled=True):
            output = module([deep, lateral])
            loss = (output.float() * torch.randn_like(output.float())).mean()
        assert torch.isfinite(output).all() and torch.isfinite(loss)
        loss.backward()
        assert grad_sum(module.residual_out.weight) > 0.0
        optimizer.step()
    checkpoint = tmp_path / "urr_dcra_state.pt"
    torch.save(module.state_dict(), checkpoint)
    clone = URRDCRAUp(c_deep=16, c_lateral=8).cuda().eval()
    clone.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    deep, lateral = make_inputs(seed=12, device="cuda")
    with torch.no_grad():
        assert torch.equal(module.eval()([deep, lateral]), clone([deep, lateral]))
