from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import DCRAUp
from ultralytics.utils.torch_utils import initialize_weights


BASELINE_YAML = "ultralytics/cfg/models/v13/yolov13-original.yaml"

MODEL_YAMLS = [
    "ultralytics/cfg/models/v13/yolov13-dcra.yaml",
    "ultralytics/cfg/models/v13/yolov13-dcra-no-entropy.yaml",
    "ultralytics/cfg/models/v13/yolov13-dcra-deep-only.yaml",
    "ultralytics/cfg/models/v13/yolov13-dcra-tau020.yaml",
    "ultralytics/cfg/models/v13/yolov13-dcra-k5.yaml",
]


def make_inputs(batch=2, c_deep=64, c_lateral=32, deep_size=(20, 20), scale=2, device="cpu"):
    deep_h, deep_w = deep_size
    deep = torch.randn(batch, c_deep, deep_h, deep_w, device=device)
    lateral = torch.randn(batch, c_lateral, deep_h * scale, deep_w * scale, device=device)
    return deep, lateral


def flatten_tensors(value):
    tensors = []
    if torch.is_tensor(value):
        tensors.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_tensors(item))
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_tensors(item))
    return tensors


def test_initial_output_is_exact_nearest_and_lateral_is_unchanged():
    torch.manual_seed(0)
    module = DCRAUp(c_deep=64, c_lateral=32).eval()
    deep, lateral = make_inputs()
    lateral_before = lateral.clone()
    with torch.no_grad():
        output = module([deep, lateral])
        reference = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    assert output.shape == reference.shape
    assert torch.equal(output, reference)
    assert torch.equal(lateral, lateral_before)


def test_phase_optimized_path_matches_explicit_reference():
    torch.manual_seed(1)
    module = DCRAUp(c_deep=16, c_lateral=8, kernel_size=3).eval()
    deep, lateral = make_inputs(batch=1, c_deep=16, c_lateral=8, deep_size=(6, 5))
    with torch.no_grad():
        key_low = module.key_proj(deep)
        key_patches = module._extract_local_patches(key_low)
        value_patches = module._extract_local_patches(deep)
        query = module.query_proj(lateral)
        optimized_values, optimized_weights = module._phase_correlate_and_reassemble(
            query,
            key_patches,
            value_patches,
        )

        key_patches_high = module._resize_patch_tensor(key_patches, lateral.shape[-2:])
        query_fp32 = F.normalize(query.float(), p=2, dim=1, eps=module.eps)
        keys_fp32 = F.normalize(key_patches_high.float(), p=2, dim=1, eps=module.eps)
        reference_logits = (query_fp32.unsqueeze(2) * keys_fp32).sum(dim=1) / module.temperature
        reference_weights = torch.softmax(reference_logits, dim=1)
        value_patches_high = module._resize_patch_tensor(value_patches, lateral.shape[-2:])
        reference_values = torch.einsum(
            "bckhw,bkhw->bchw",
            value_patches_high.float(),
            reference_weights,
        )

    assert torch.allclose(optimized_weights, reference_weights, atol=2e-6, rtol=2e-6)
    assert torch.allclose(optimized_values, reference_values, atol=2e-6, rtol=2e-6)


def test_exact_scale_forward_does_not_use_high_resolution_candidate_fallback(monkeypatch):
    module = DCRAUp(c_deep=16, c_lateral=8).eval()
    deep, lateral = make_inputs(batch=1, c_deep=16, c_lateral=8, deep_size=(6, 5))

    def reject_fallback(*_args, **_kwargs):
        raise AssertionError("Exact-scale path entered _resize_patch_tensor fallback.")

    monkeypatch.setattr(module, "_resize_patch_tensor", reject_fallback)
    with torch.no_grad():
        output = module([deep, lateral])
    assert output.shape == (1, 16, 12, 10)


@pytest.mark.parametrize(
    "use_entropy,use_lateral_guidance,kernel_size,temperature",
    [
        (True, True, 3, 0.10),
        (False, True, 3, 0.10),
        (True, False, 3, 0.10),
        (True, True, 3, 0.20),
        (True, True, 5, 0.10),
    ],
)
def test_all_ablation_variants_forward(use_entropy, use_lateral_guidance, kernel_size, temperature):
    module = DCRAUp(
        c_deep=64,
        c_lateral=32,
        use_entropy=use_entropy,
        use_lateral_guidance=use_lateral_guidance,
        kernel_size=kernel_size,
        temperature=temperature,
    ).eval()
    deep, lateral = make_inputs()
    with torch.no_grad():
        output = module([deep, lateral])
    assert output.shape == (2, 64, 40, 40)
    assert torch.isfinite(output).all()
    assert (module.query_proj is not None) == use_lateral_guidance


@pytest.mark.parametrize("kernel_size", [3, 5])
def test_correlation_weights_and_confidence(kernel_size):
    torch.manual_seed(1)
    module = DCRAUp(c_deep=64, c_lateral=32, kernel_size=kernel_size, use_entropy=True).eval()
    deep, lateral = make_inputs()
    with torch.no_grad():
        _, _, weights, confidence = module._compute_alignment(deep, lateral)
    assert weights.dtype == torch.float32
    assert confidence.dtype == torch.float32
    assert weights.shape == (2, kernel_size**2, 40, 40)
    assert confidence.shape == (2, 1, 40, 40)
    assert torch.isfinite(weights).all()
    assert torch.isfinite(confidence).all()
    assert (weights.sum(dim=1) - 1.0).abs().max() < 1e-5
    assert confidence.min() >= 0.0
    assert confidence.max() <= 1.0
    assert confidence.requires_grad is False


def test_no_entropy_returns_unit_confidence():
    module = DCRAUp(c_deep=64, c_lateral=32, use_entropy=False).eval()
    deep, lateral = make_inputs()
    with torch.no_grad():
        _, _, _, confidence = module._compute_alignment(deep, lateral)
    assert torch.equal(confidence, torch.ones_like(confidence))


def test_initialize_weights_keeps_zero_residual_projection():
    module = DCRAUp(c_deep=64, c_lateral=32)
    initialize_weights(module)
    assert torch.count_nonzero(module.residual_out.weight) == 0


def test_rng_state_is_preserved_and_local_init_is_repeatable():
    seed = 12345
    torch.manual_seed(seed)
    expected_a = nn.Conv2d(3, 8, 3)
    expected_b = nn.Conv2d(8, 8, 3)
    expected_b_weight = expected_b.weight.detach().clone()

    torch.manual_seed(seed)
    actual_a = nn.Conv2d(3, 8, 3)
    first_dcra = DCRAUp(c_deep=64, c_lateral=32)
    actual_b = nn.Conv2d(8, 8, 3)

    torch.manual_seed(seed)
    _ = nn.Conv2d(3, 8, 3)
    second_dcra = DCRAUp(c_deep=64, c_lateral=32)
    assert torch.equal(expected_a.weight, actual_a.weight)
    assert torch.equal(expected_b_weight, actual_b.weight)
    assert torch.equal(first_dcra.key_proj.weight, second_dcra.key_proj.weight)
    assert torch.equal(first_dcra.query_proj.weight, second_dcra.query_proj.weight)


def test_two_step_gradient_activation():
    torch.manual_seed(7)
    module = DCRAUp(c_deep=64, c_lateral=32, use_entropy=True, use_lateral_guidance=True).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.10, momentum=0.0, weight_decay=0.0)

    deep, lateral = make_inputs()
    deep.requires_grad_(True)
    lateral.requires_grad_(True)
    output = module([deep, lateral])
    loss = (output * torch.randn_like(output)).mean()
    loss.backward()

    residual_grad_first = module.residual_out.weight.grad
    assert residual_grad_first is not None
    assert torch.isfinite(residual_grad_first).all()
    assert residual_grad_first.abs().sum() > 0
    assert module.key_proj.weight.grad is not None
    assert module.query_proj.weight.grad is not None
    assert module.key_proj.weight.grad.abs().sum() == 0
    assert module.query_proj.weight.grad.abs().sum() == 0
    assert lateral.grad is not None
    assert lateral.grad.abs().sum() == 0

    optimizer.step()
    assert torch.count_nonzero(module.residual_out.weight) > 0
    optimizer.zero_grad(set_to_none=True)
    deep_second, lateral_second = make_inputs()
    deep_second.requires_grad_(True)
    lateral_second.requires_grad_(True)
    output_second = module([deep_second, lateral_second])
    loss_second = (output_second * torch.randn_like(output_second)).mean()
    loss_second.backward()

    for name, parameter in (
        ("residual_out.weight", module.residual_out.weight),
        ("key_proj.weight", module.key_proj.weight),
        ("query_proj.weight", module.query_proj.weight),
    ):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert lateral_second.grad is not None
    assert torch.isfinite(lateral_second.grad).all()
    assert lateral_second.grad.abs().sum() > 0


def test_state_dict_strict_round_trip():
    torch.manual_seed(8)
    module = DCRAUp(c_deep=64, c_lateral=32).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    deep, lateral = make_inputs()
    output = module([deep, lateral])
    (output * torch.randn_like(output)).mean().backward()
    optimizer.step()

    restored = DCRAUp(c_deep=64, c_lateral=32)
    restored.load_state_dict(module.state_dict(), strict=True)
    module.eval()
    restored.eval()
    with torch.no_grad():
        first = module([deep, lateral])
        second = restored([deep, lateral])
    assert torch.equal(first, second)


def test_invalid_inputs_raise():
    module = DCRAUp(c_deep=64, c_lateral=32)
    deep = torch.randn(2, 64, 20, 20)
    wrong_channels = torch.randn(2, 31, 40, 40)
    wrong_scale = torch.randn(2, 32, 39, 40)
    with pytest.raises(ValueError, match="Lateral-channel mismatch"):
        module([deep, wrong_channels])
    with pytest.raises(ValueError, match="spatial scale mismatch"):
        module([deep, wrong_scale])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast validation.")
def test_cuda_half_model_validation_forward():
    module = DCRAUp(c_deep=64, c_lateral=32).cuda().half().eval()
    deep, lateral = make_inputs(device="cuda")
    deep = deep.half()
    lateral = lateral.half()
    with torch.no_grad():
        output = module([deep, lateral])
        reference = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float()
    assert output.dtype == torch.float16
    assert torch.equal(output, reference.half())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast validation.")
def test_cuda_amp_two_step_backward():
    torch.manual_seed(9)
    torch.cuda.manual_seed_all(9)
    module = DCRAUp(c_deep=64, c_lateral=32).cuda().train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.10, momentum=0.0, weight_decay=0.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs(device="cuda")
        deep.requires_grad_(True)
        lateral.requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=True):
            output = module([deep, lateral])
            loss = (output.float() * torch.randn_like(output.float())).mean()
        assert torch.isfinite(output).all()
        assert torch.isfinite(loss)
        loss.backward()
        assert module.residual_out.weight.grad is not None
        assert torch.isfinite(module.residual_out.weight.grad).all()
        assert module.residual_out.weight.grad.abs().sum() > 0
        if step == 1:
            assert module.key_proj.weight.grad is not None
            assert module.query_proj.weight.grad is not None
            assert module.key_proj.weight.grad.abs().sum() > 0
            assert module.query_proj.weight.grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize("yaml_path", MODEL_YAMLS)
def test_model_yaml_build_and_topology(yaml_path):
    path = Path(yaml_path)
    assert path.is_file(), yaml_path
    model = YOLO(str(path)).model.eval()
    dcra_modules = [module for module in model.modules() if isinstance(module, DCRAUp)]
    assert len(dcra_modules) == 1
    assert isinstance(model.model[15], DCRAUp)
    assert model.model[18].__class__.__name__ == "FullPAD_Tunnel"
    assert isinstance(model.model[19], nn.Upsample)
    assert model.model[15].f == [14, 12]
    assert model.model[-1].f == [23, 27, 31]

    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(sample)

    tensors = []

    def collect(value):
        if torch.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(output)
    assert tensors
    assert all(torch.isfinite(tensor).all() for tensor in tensors)


def test_full_model_initial_equivalence_to_d0():
    seed = 20260713
    torch.manual_seed(seed)
    baseline = YOLO(BASELINE_YAML).model.eval()
    torch.manual_seed(seed)
    dcra = YOLO(MODEL_YAMLS[0]).model.eval()

    baseline_state = baseline.state_dict()
    dcra_state = dcra.state_dict()
    for name, tensor in baseline_state.items():
        assert name in dcra_state, name
        assert tensor.shape == dcra_state[name].shape, name
        assert torch.equal(tensor, dcra_state[name]), name

    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        baseline_output = flatten_tensors(baseline(sample))
        dcra_output = flatten_tensors(dcra(sample))
    assert len(baseline_output) == len(dcra_output)
    for baseline_tensor, dcra_tensor in zip(baseline_output, dcra_output):
        assert baseline_tensor.shape == dcra_tensor.shape
        assert torch.equal(baseline_tensor, dcra_tensor)
