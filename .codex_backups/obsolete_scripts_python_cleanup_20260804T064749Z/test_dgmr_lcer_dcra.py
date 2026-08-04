"""Blocking repository-level tests for DGMR-LCER-DCRA integration."""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ultralytics.nn.modules.block import DGMRLCERDCRAUp, LSMRLCERDCRAUp


DGMR_YAMLS = (
    Path("ultralytics/cfg/models/v13/yolov13-dgmr-lcer-dcra-d1-matched.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-dgmr-lcer-dcra-d2-global.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-dgmr-lcer-dcra-d3-local.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-dgmr-lcer-dcra-d4-dual.yaml"),
)


def _set_nonzero_residual(module: torch.nn.Module, std: float = 0.05) -> None:
    with torch.no_grad():
        module.residual_out.weight.normal_(mean=0.0, std=std)


def _copy_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)


def _config(**overrides) -> dict:
    config = {
        "release_mode": "local",
        "strict_ratio": 0.20,
        "channel_power": 2.0,
        "spatial_power": 1.0,
        "consensus_kernel": 3,
        "energy_weighted_channel": True,
        "detach_release": True,
    }
    config.update(overrides)
    return config


def _inputs(channels=64, lateral_channels=32, height=5, width=7):
    return torch.randn(2, channels, height, width), torch.randn(2, lateral_channels, height * 2, width * 2)


def test_initial_nearest_and_state() -> None:
    torch.manual_seed(0)
    dgmr = DGMRLCERDCRAUp(64, 32, {}).eval()
    lsmr = LSMRLCERDCRAUp(64, 32, _config(moment_mode="local")).eval()
    deep, lateral = _inputs()
    output = dgmr([deep, lateral])
    nearest = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float()
    assert torch.equal(output, nearest), (output - nearest).abs().max().item()
    assert set(dgmr.state_dict()) == set(lsmr.state_dict())
    assert sum(parameter.numel() for parameter in dgmr.parameters()) == sum(
        parameter.numel() for parameter in lsmr.parameters()
    )
    assert len(list(dgmr.buffers())) == len(list(lsmr.buffers()))


def test_completed_endpoints_are_exact() -> None:
    torch.manual_seed(1)
    deep, lateral = _inputs()
    for mode in ("matched", "raw", "global", "local"):
        cfg = _config(
            moment_mode=mode,
            moment_relax_max=0.50,
            support_reference=0.10,
            support_power=1.0,
            local_support_kernel=5,
        )
        lsmr = LSMRLCERDCRAUp(64, 32, cfg).eval()
        dgmr = DGMRLCERDCRAUp(64, 32, cfg).eval()
        _copy_state(lsmr, dgmr)
        _set_nonzero_residual(lsmr)
        with torch.no_grad():
            dgmr.residual_out.weight.copy_(lsmr.residual_out.weight)
        parent_output, child_output = lsmr([deep, lateral]), dgmr([deep, lateral])
        assert torch.equal(parent_output, child_output), (mode, (parent_output - child_output).abs().max().item())


def test_cell_lower_envelope_even_and_odd() -> None:
    module = DGMRLCERDCRAUp(8, 4, _config(moment_mode="dual", moment_relax_max=0.50)).eval()
    even = torch.tensor(
        [[[[0.40, 0.20, 0.30, 0.10], [0.35, 0.25, 0.15, 0.05],
           [0.45, 0.42, 0.32, 0.31], [0.44, 0.43, 0.29, 0.28]]]], dtype=torch.float32
    )
    shared = module._cell_consistent_lower_envelope(even)
    expected = torch.tensor(
        [[[[0.20, 0.20, 0.05, 0.05], [0.20, 0.20, 0.05, 0.05],
           [0.42, 0.42, 0.28, 0.28], [0.42, 0.42, 0.28, 0.28]]]], dtype=torch.float32
    )
    assert torch.equal(shared, expected)
    assert torch.all(shared <= even + 1e-7)
    odd = torch.rand(2, 3, 5, 7) * 0.50
    odd_shared = module._cell_consistent_lower_envelope(odd)
    assert odd_shared.shape == odd.shape
    assert torch.isfinite(odd_shared).all()
    assert torch.all((odd_shared >= 0.0) & (odd_shared <= 0.50))


def test_dual_gate_is_bounded_by_both_routes() -> None:
    module = DGMRLCERDCRAUp(4, 2, _config(moment_mode="dual", moment_relax_max=0.50)).eval()
    raw, matched, confidence = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4), torch.ones(1, 1, 4, 4)
    global_gate = torch.tensor([[[[0.30]], [[0.20]], [[0.45]], [[0.10]]]], dtype=torch.float32)
    local_gate = torch.tensor(
        [[[[0.40, 0.35, 0.30, 0.25], [0.32, 0.31, 0.24, 0.22],
           [0.20, 0.18, 0.42, 0.41], [0.15, 0.14, 0.40, 0.39]]]], dtype=torch.float32
    ).repeat(1, 4, 1, 1)

    def fake_global(self, residual, conf):
        return global_gate.to(residual.device)

    def fake_local(self, delta, conf):
        return local_gate.to(delta.device)

    module._compute_global_relaxation = types.MethodType(fake_global, module)
    module._compute_local_relaxation = types.MethodType(fake_local, module)
    dual = module._compute_dual_relaxation(raw, matched, confidence)
    cell_local = module._cell_consistent_lower_envelope(local_gate)
    assert dual.shape == raw.shape
    assert torch.all(dual <= global_gate + 1e-7)
    assert torch.all(dual <= cell_local + 1e-7)
    assert torch.all((dual >= 0.0) & (dual <= module.moment_relax_max))


def test_nonfinite_gate_is_conservatively_closed() -> None:
    module = DGMRLCERDCRAUp(4, 2, _config(moment_mode="dual", moment_relax_max=0.50)).eval()
    reference = torch.randn(1, 4, 4, 4)
    local = torch.full_like(reference, 0.25)
    local[:, :, 0, 0] = float("nan")
    local[:, :, 1, 1] = float("inf")
    sanitized = module._validate_relaxation_map(local, reference, "local", spatial=True)
    assert torch.isfinite(sanitized).all()
    assert torch.equal(sanitized[:, :, 0, 0], torch.zeros_like(sanitized[:, :, 0, 0]))
    assert torch.equal(sanitized[:, :, 1, 1], torch.zeros_like(sanitized[:, :, 1, 1]))


def test_one_unsupported_subpixel_protects_cell() -> None:
    module = DGMRLCERDCRAUp(4, 2, _config(moment_mode="dual", moment_relax_max=0.50)).eval()
    local = torch.full((1, 4, 4, 4), 0.40)
    local[:, :, 1, 1] = 0.0
    shared = module._cell_consistent_lower_envelope(local)
    assert torch.equal(shared[:, :, :2, :2], torch.zeros_like(shared[:, :, :2, :2]))
    assert torch.equal(shared[:, :, 2:, 2:], torch.full_like(shared[:, :, 2:, 2:], 0.40))


def test_dual_residual_formula_and_zero_relaxation() -> None:
    torch.manual_seed(2)
    module = DGMRLCERDCRAUp(8, 4, _config(moment_mode="dual", moment_relax_max=0.50)).eval()
    base, raw, confidence = torch.randn(2, 8, 6, 10), torch.randn(2, 8, 6, 10), torch.rand(2, 1, 6, 10)
    matched = super(DGMRLCERDCRAUp, module)._moment_preserving_residual(base, raw)
    gate = module._compute_dual_relaxation(raw, matched, confidence)
    expected = matched.float() + gate * (raw.float() - matched.float())
    assert torch.allclose(module._local_support_moment_residual(base, raw, confidence).float(), expected, atol=1e-6, rtol=1e-6)

    matched_endpoint = DGMRLCERDCRAUp(32, 16, _config(moment_mode="matched", moment_relax_max=0.50)).eval()
    dual_zero = DGMRLCERDCRAUp(32, 16, _config(moment_mode="dual", moment_relax_max=0.0)).eval()
    _copy_state(matched_endpoint, dual_zero)
    _set_nonzero_residual(matched_endpoint)
    with torch.no_grad():
        dual_zero.residual_out.weight.copy_(matched_endpoint.residual_out.weight)
    deep, lateral = _inputs(32, 16)
    assert torch.equal(matched_endpoint([deep, lateral]), dual_zero([deep, lateral]))


def test_shapes_reload_autocast_and_two_step_gradient() -> None:
    torch.manual_seed(3)
    module = DGMRLCERDCRAUp(32, 16, _config(moment_mode="dual")).eval()
    _set_nonzero_residual(module)
    for height, width in ((3, 5), (4, 4), (7, 9)):
        deep, lateral = _inputs(32, 16, height, width)
        output = module([deep, lateral])
        assert output.shape == (2, 32, height * 2, width * 2) and torch.isfinite(output).all()
    clone = DGMRLCERDCRAUp(32, 16, _config(moment_mode="dual")).eval()
    clone.load_state_dict(module.state_dict(), strict=True)
    deep, lateral = _inputs(32, 16)
    assert torch.equal(module([deep, lateral]), clone([deep, lateral]))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.isfinite(module([deep, lateral])).all()

    gradient_module = DGMRLCERDCRAUp(32, 16, _config(moment_mode="dual")).train()
    optimizer = torch.optim.SGD(gradient_module.parameters(), lr=0.01)
    deep = torch.randn(2, 32, 5, 7, requires_grad=True)
    lateral = torch.randn(2, 16, 10, 14, requires_grad=True)
    gradient_module([deep, lateral]).square().mean().backward()
    assert gradient_module.residual_out.weight.grad is not None
    assert torch.count_nonzero(gradient_module.residual_out.weight.grad).item() > 0
    assert gradient_module.key_proj.weight.grad is None or torch.count_nonzero(gradient_module.key_proj.weight.grad).item() == 0
    assert gradient_module.query_proj.weight.grad is None or torch.count_nonzero(gradient_module.query_proj.weight.grad).item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    gradient_module([deep, lateral]).square().mean().backward()
    assert gradient_module.key_proj.weight.grad is not None and torch.count_nonzero(gradient_module.key_proj.weight.grad).item() > 0
    assert gradient_module.query_proj.weight.grad is not None and torch.count_nonzero(gradient_module.query_proj.weight.grad).item() > 0


def test_invalid_configs_and_yaml_contracts() -> None:
    for config in (
        {"unknown_key": 1}, {"moment_mode": "invalid"}, {"moment_relax_max": -0.1},
        {"moment_relax_max": 1.1}, {"support_reference": 0.0}, {"support_power": 0.0},
        {"local_support_kernel": 2}, {"local_support_kernel": 4}, {"preserve_moments": False},
    ):
        try:
            DGMRLCERDCRAUp(32, 16, config)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"Invalid config was accepted: {config}")
    common = None
    for path, expected_mode in zip(DGMR_YAMLS, ("matched", "global", "local", "dual")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        layer = payload["head"][6]
        assert layer[0] == [14, 12] and layer[2] == "DGMRLCERDCRAUp"
        config = layer[3][0]
        assert set(config) == set(DGMRLCERDCRAUp._DEFAULT_CONFIG)
        assert config["moment_mode"] == expected_mode
        assert payload["head"][7] == [[-1, 12], 1, "Concat", [1]]
        assert payload["head"][-1] == [[23, 27, 31], 1, "Detect", ["nc"]]
        without_mode = {key: value for key, value in config.items() if key != "moment_mode"}
        if common is None:
            common = without_mode
        else:
            assert without_mode == common


def test_yaml_parser(yaml_path: str) -> None:
    from ultralytics.nn.tasks import DetectionModel

    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    model = DetectionModel(str(path), ch=3, nc=4, verbose=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 640, 640)) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default=None, help="Optional DGMR YAML path for a 640x640 parser/forward test.")
    args = parser.parse_args()
    test_initial_nearest_and_state()
    test_completed_endpoints_are_exact()
    test_cell_lower_envelope_even_and_odd()
    test_dual_gate_is_bounded_by_both_routes()
    test_nonfinite_gate_is_conservatively_closed()
    test_one_unsupported_subpixel_protects_cell()
    test_dual_residual_formula_and_zero_relaxation()
    test_shapes_reload_autocast_and_two_step_gradient()
    test_invalid_configs_and_yaml_contracts()
    if args.yaml:
        test_yaml_parser(args.yaml)
    print("ALL DGMR-LCER-DCRA BLOCKING TESTS PASSED")


if __name__ == "__main__":
    main()
