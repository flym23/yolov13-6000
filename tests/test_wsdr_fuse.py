import copy

import pytest
import torch

from ultralytics.nn.modules import FAARUp, WSDRFuse


@pytest.mark.parametrize("shape", [(2, 8, 40, 40), (2, 8, 41, 39), (1, 3, 1, 1), (1, 3, 2, 3)])
def test_haar_round_trip_fp32(shape):
    x = torch.randn(*shape, dtype=torch.float32)
    ll, details, original_size = WSDRFuse._haar_decompose(x)
    y = WSDRFuse._haar_reconstruct(ll, details, original_size)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


def test_avgpool_round_trip_fp32():
    x = torch.randn(2, 8, 41, 39)
    low, details, original_size = WSDRFuse._avgpool_decompose(x)
    y = WSDRFuse._avgpool_reconstruct(low, details, original_size)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "adaptive,use_hf_energy,decomposition,hf_reweight",
    [
        (False, True, "haar", False),
        (True, True, "haar", False),
        (True, False, "haar", False),
        (True, True, "avgpool", False),
        (True, True, "haar", True),
    ],
)
def test_wsdr_shape_and_backward(adaptive, use_hf_energy, decomposition, hf_reweight):
    module = WSDRFuse(
        c_deep=64,
        c_lat=32,
        adaptive=adaptive,
        use_hf_energy=use_hf_energy,
        decomposition=decomposition,
        hf_reweight=hf_reweight,
    ).train()
    deep = torch.randn(2, 64, 20, 20, requires_grad=True)
    lateral = torch.randn(2, 32, 40, 40, requires_grad=True)
    output = module([deep, lateral])
    assert output.shape == (2, 96, 40, 40)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert deep.grad is not None and lateral.grad is not None
    assert torch.isfinite(deep.grad).all() and torch.isfinite(lateral.grad).all()
    assert deep.grad.abs().sum() > 0 and lateral.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
        assert parameter.grad.abs().sum() > 0, f"Zero gradient: {name}"


def test_wsdr_odd_spatial_shape():
    module = WSDRFuse(c_deep=64, c_lat=32, adaptive=True, decomposition="haar").eval()
    deep = torch.randn(2, 64, 20, 19)
    lateral = torch.randn(2, 32, 41, 39)
    with torch.no_grad():
        output = module([deep, lateral])
    assert output.shape == (2, 96, 41, 39)
    assert torch.isfinite(output).all()


def test_fixed_zero_gate_preserves_lateral_exactly():
    module = WSDRFuse(
        c_deep=64,
        c_lat=32,
        adaptive=False,
        fixed_gate=0.0,
        decomposition="haar",
        hf_reweight=False,
    ).eval()
    deep = torch.randn(2, 64, 20, 20)
    lateral = torch.randn(2, 32, 40, 40)
    with torch.no_grad():
        output = module([deep, lateral])
    assert torch.allclose(output[:, 64:], lateral, atol=1e-6, rtol=1e-6)


def test_main_model_preserves_haar_high_frequency():
    module = WSDRFuse(
        c_deep=64,
        c_lat=32,
        adaptive=True,
        use_hf_energy=True,
        decomposition="haar",
        hf_reweight=False,
    ).eval()
    deep = torch.randn(2, 64, 20, 20)
    lateral = torch.randn(2, 32, 40, 40)
    with torch.no_grad():
        lateral_output = module([deep, lateral])[:, 64:]
    _, details_before, _ = WSDRFuse._haar_decompose(lateral)
    _, details_after, _ = WSDRFuse._haar_decompose(lateral_output)
    for before, after in zip(details_before, details_after):
        assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)


def test_gate_bounds_and_shape():
    module = WSDRFuse(c_deep=64, c_lat=32, g_max=0.25, adaptive=True, use_hf_energy=True).eval()
    deep = torch.randn(2, 64, 20, 20)
    lateral = torch.randn(2, 32, 40, 40)
    with torch.no_grad():
        low, details, _ = module._decompose(lateral)
        gate = module._compute_gate(module.deep_proj(deep), low, details)
    assert gate.shape == (2, 1, 20, 20)
    assert torch.isfinite(gate).all()
    assert gate.min() >= 0.0 and gate.max() <= module.g_max


def test_fixed_gate_is_exact_constant():
    module = WSDRFuse(c_deep=64, c_lat=32, g_max=0.25, adaptive=False, fixed_gate=0.125).eval()
    deep = torch.randn(2, 64, 20, 20)
    lateral = torch.randn(2, 32, 40, 40)
    with torch.no_grad():
        low, details, _ = module._decompose(lateral)
        gate = module._compute_gate(module.deep_proj(deep), low, details)
    assert torch.equal(gate, torch.full_like(gate, 0.125))


def test_internal_faar_path_matches_standalone_faar():
    reference = FAARUp(c1=64, scale=2, mode="semantic", groups=4).eval()
    module = WSDRFuse(c_deep=64, c_lat=32, adaptive=False, fixed_gate=0.125, faar_groups=4).eval()
    module.deep_up.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)
    deep = torch.randn(2, 64, 20, 20)
    lateral = torch.randn(2, 32, 40, 40)
    with torch.no_grad():
        expected = reference(deep)
        actual = module([deep, lateral])[:, :64]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16 autocast validation.")
def test_wsdr_cuda_amp():
    module = WSDRFuse(
        c_deep=64,
        c_lat=32,
        adaptive=True,
        use_hf_energy=True,
        decomposition="haar",
        hf_reweight=False,
    ).cuda().train()
    deep = torch.randn(2, 64, 20, 20, device="cuda", requires_grad=True)
    lateral = torch.randn(2, 32, 40, 40, device="cuda", requires_grad=True)
    with torch.cuda.amp.autocast(enabled=True):
        output = module([deep, lateral])
        loss = output.float().square().mean()
    loss.backward()
    assert output.shape == (2, 96, 40, 40)
    assert torch.isfinite(output).all()
    assert torch.isfinite(deep.grad).all() and torch.isfinite(lateral.grad).all()
