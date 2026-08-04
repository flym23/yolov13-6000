from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import DCRAUp, MEDCRAUp


MODEL_YAMLS = [
    "ultralytics/cfg/models/v13/yolov13-medcra.yaml",
    "ultralytics/cfg/models/v13/yolov13-medcra-no-moment.yaml",
    "ultralytics/cfg/models/v13/yolov13-medcra-no-center.yaml",
    "ultralytics/cfg/models/v13/yolov13-medcra-no-bound.yaml",
    "ultralytics/cfg/models/v13/yolov13-medcra-rho005.yaml",
    "ultralytics/cfg/models/v13/yolov13-medcra-rho020.yaml",
]
A4_YAML = "ultralytics/cfg/models/v13/yolov13-dcra-tau020.yaml"


def make_inputs(device="cpu"):
    deep = torch.randn(1, 16, 4, 4, device=device)
    lateral = torch.randn(1, 8, 8, 8, device=device)
    return deep, lateral


def centered_rms(x, eps):
    centered = x.float() - x.float().mean(dim=(2, 3), keepdim=True)
    return centered.square().mean(dim=(2, 3)).add(eps).sqrt()


def grad_sum(parameter):
    return 0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum())


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    return []


def test_medcra_adds_no_parameters_over_dcra():
    dcra = DCRAUp(c_deep=16, c_lateral=8, temperature=0.20)
    medcra = MEDCRAUp(c_deep=16, c_lateral=8, temperature=0.20)
    assert list(dcra.state_dict()) == list(medcra.state_dict())
    assert sum(p.numel() for p in dcra.parameters()) == sum(p.numel() for p in medcra.parameters())
    assert not list(medcra.buffers())


def test_initial_output_is_exact_nearest():
    torch.manual_seed(0)
    module = MEDCRAUp(c_deep=16, c_lateral=8).eval()
    deep, lateral = make_inputs()
    with torch.no_grad():
        output = module([deep, lateral])
    assert torch.equal(output, F.interpolate(deep, size=lateral.shape[-2:], mode="nearest"))


def test_constraints_disabled_are_exactly_a4_dcra():
    torch.manual_seed(1)
    dcra = DCRAUp(c_deep=16, c_lateral=8, temperature=0.20).eval()
    medcra = MEDCRAUp(
        c_deep=16,
        c_lateral=8,
        temperature=0.20,
        preserve_moments=False,
        center_correction=False,
        use_energy_bound=False,
    ).eval()
    with torch.no_grad():
        dcra.residual_out.weight.normal_(mean=0.0, std=0.05)
    medcra.load_state_dict(dcra.state_dict(), strict=True)
    deep, lateral = make_inputs()
    with torch.no_grad():
        assert torch.equal(dcra([deep, lateral]), medcra([deep, lateral]))


def test_moment_matching_in_controlled_non_clamped_case():
    torch.manual_seed(2)
    module = MEDCRAUp(c_deep=16, c_lateral=8, moment_scale_max=4.0).eval()
    base = torch.randn(2, 16, 8, 8)
    matched = module._moment_preserving_residual(base, 0.10 * torch.randn_like(base))
    base_mean, _, base_rms = module._spatial_mean_and_centered_rms(base, module.eps)
    candidate_mean, _, candidate_rms = module._spatial_mean_and_centered_rms(base + matched, module.eps)
    assert torch.allclose(base_mean, candidate_mean, atol=2e-6, rtol=0.0)
    assert torch.allclose(base_rms, candidate_rms, atol=2e-5, rtol=2e-5)


def test_zero_mean_and_energy_bound_after_projection():
    torch.manual_seed(3)
    module = MEDCRAUp(c_deep=16, c_lateral=8, max_residual_ratio=0.10).eval()
    with torch.no_grad():
        module.residual_out.weight.normal_(mean=0.0, std=0.20)
    deep, lateral = make_inputs()
    with torch.no_grad():
        output = module([deep, lateral])
    base = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    correction = output - base
    assert correction.float().mean(dim=(2, 3)).abs().max() < 2e-5
    assert torch.all(centered_rms(correction, module.eps) <= module.max_residual_ratio * centered_rms(base, module.eps) + 2e-5)


def test_no_bound_preserves_only_centering():
    torch.manual_seed(4)
    module = MEDCRAUp(c_deep=16, c_lateral=8, use_energy_bound=False).eval()
    with torch.no_grad():
        module.residual_out.weight.normal_(mean=0.0, std=0.20)
    deep, lateral = make_inputs()
    with torch.no_grad():
        correction = module([deep, lateral]) - F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    assert correction.float().mean(dim=(2, 3)).abs().max() < 2e-5


def test_two_step_gradient_activation():
    torch.manual_seed(5)
    module = MEDCRAUp(c_deep=16, c_lateral=8).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.20, momentum=0.0, weight_decay=0.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs()
        deep.requires_grad_(True)
        lateral.requires_grad_(True)
        loss = (module([deep, lateral]) * torch.randn(1, 16, 8, 8)).mean()
        loss.backward()
        assert grad_sum(module.residual_out.weight) > 0.0
        if step == 0:
            assert grad_sum(module.key_proj.weight) == 0.0
            assert grad_sum(module.query_proj.weight) == 0.0
        else:
            assert grad_sum(module.key_proj.weight) > 0.0
            assert grad_sum(module.query_proj.weight) > 0.0
            assert lateral.grad is not None and torch.isfinite(lateral.grad).all() and lateral.grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize("yaml_path", MODEL_YAMLS)
def test_model_yaml_build_and_topology(yaml_path):
    assert Path(yaml_path).is_file()
    model = YOLO(yaml_path).model.eval()
    assert sum(isinstance(module, MEDCRAUp) for module in model.modules()) == 1
    assert isinstance(model.model[15], MEDCRAUp)
    # A4's untouched FullPAD_Tunnel remains at layer 18; its original nearest P4->P3 upsample is layer 19.
    assert isinstance(model.model[19], nn.Upsample)
    assert model.model[15].f == [14, 12]
    assert model.model[-1].f == [23, 27, 31]
    with torch.no_grad():
        assert all(torch.isfinite(value).all() for value in tensors(model(torch.randn(1, 3, 128, 128))))


def test_full_model_state_compatibility_with_a4():
    a4_model = YOLO(A4_YAML).model.eval()
    me_model = YOLO(MODEL_YAMLS[0]).model.eval()
    me_model.load_state_dict(a4_model.state_dict(), strict=True)
    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        a4_output, me_output = a4_model(sample), me_model(sample)
    assert len(tensors(a4_output)) == len(tensors(me_output))
    assert all(torch.equal(a, b) for a, b in zip(tensors(a4_output), tensors(me_output)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP validation.")
def test_cuda_amp_two_step_and_strict_reload(tmp_path):
    torch.manual_seed(6)
    module = MEDCRAUp(c_deep=16, c_lateral=8).cuda().train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.20, momentum=0.0, weight_decay=0.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs(device="cuda")
        with torch.cuda.amp.autocast(enabled=True):
            output = module([deep, lateral])
            loss = (output.float() * torch.randn_like(output.float())).mean()
        assert torch.isfinite(output).all() and torch.isfinite(loss)
        loss.backward()
        assert grad_sum(module.residual_out.weight) > 0.0
        if step:
            assert grad_sum(module.key_proj.weight) > 0.0
        optimizer.step()
    checkpoint = tmp_path / "medcra_state.pt"
    torch.save(module.state_dict(), checkpoint)
    clone = MEDCRAUp(c_deep=16, c_lateral=8).cuda().eval()
    clone.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    deep, lateral = make_inputs(device="cuda")
    with torch.no_grad():
        assert torch.equal(module.eval()([deep, lateral]), clone([deep, lateral]))
