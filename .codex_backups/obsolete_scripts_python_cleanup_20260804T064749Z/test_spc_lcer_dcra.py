"""Blocking repository-level tests for the SPC-LCER-DCRA integration.

Run from the repository root:
    python test_spc_lcer_dcra.py
    python test_spc_lcer_dcra.py --yaml ultralytics/cfg/models/v13/yolov13-spc-p3-main.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ultralytics.nn.modules.block import LCERDCRAUp, SPCLCERDCRAUp


SPC_YAMLS = (
    Path("ultralytics/cfg/models/v13/yolov13-spc-p1-lcer-endpoint.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-spc-p2-naive.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-spc-p3-main.yaml"),
)


def _set_nonzero_residual(module: torch.nn.Module, std: float = 0.05) -> None:
    with torch.no_grad():
        module.residual_out.weight.normal_(mean=0.0, std=std)


def _copy_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)


def _random_phase_weights(batch=2, candidates=9, scale=2, h=5, w=7):
    return torch.softmax(torch.randn(batch, candidates, scale, scale, h, w), dim=1)


def _lcer_l3_config() -> dict:
    return {
        "release_mode": "local",
        "strict_ratio": 0.20,
        "channel_power": 2.0,
        "spatial_power": 1.0,
        "consensus_kernel": 3,
    }


def test_initial_nearest_and_state_dict() -> None:
    torch.manual_seed(0)
    module = SPCLCERDCRAUp(64, 32, {}).eval()
    lcer = LCERDCRAUp(64, 32, _lcer_l3_config()).eval()
    deep = torch.randn(2, 64, 5, 7)
    lateral = torch.randn(2, 32, 10, 14)
    output = module([deep, lateral])
    nearest = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float()
    assert torch.equal(output, nearest), (output - nearest).abs().max().item()
    assert set(module.state_dict()) == set(lcer.state_dict())
    assert sum(parameter.numel() for parameter in module.parameters()) == sum(
        parameter.numel() for parameter in lcer.parameters()
    )
    assert len(list(module.buffers())) == len(list(lcer.buffers()))


def _assert_spc_lcer_endpoint(spc_config: dict) -> None:
    torch.manual_seed(1)
    lcer = LCERDCRAUp(64, 32, _lcer_l3_config()).eval()
    spc = SPCLCERDCRAUp(64, 32, {**_lcer_l3_config(), **spc_config}).eval()
    _copy_state(lcer, spc)
    _set_nonzero_residual(lcer)
    with torch.no_grad():
        spc.residual_out.weight.copy_(lcer.residual_out.weight)
    deep = torch.randn(2, 64, 5, 7)
    lateral = torch.randn(2, 32, 10, 14)
    assert torch.equal(lcer([deep, lateral]), spc([deep, lateral]))


def test_alpha_zero_exactly_equals_lcer_l3() -> None:
    _assert_spc_lcer_endpoint({"phase_alpha_max": 0.0})


def test_disabled_phase_consistency_exactly_equals_lcer_l3() -> None:
    _assert_spc_lcer_endpoint({"use_phase_consistency": False})


def test_phase_distribution_properties() -> None:
    torch.manual_seed(2)
    module = SPCLCERDCRAUp(32, 16, {"phase_alpha_max": 0.35, "phase_borrow_power": 1.0}).eval()
    weights = _random_phase_weights()
    stabilized = module._phase_consistent_weights(weights)
    assert stabilized.shape == weights.shape
    assert torch.isfinite(stabilized).all()
    assert torch.all(stabilized >= 0)
    assert torch.allclose(stabilized.sum(dim=1), torch.ones_like(stabilized.sum(dim=1)), atol=1e-6, rtol=1e-6)


def test_confident_phases_are_protected() -> None:
    module = SPCLCERDCRAUp(16, 8, {"phase_alpha_max": 0.35, "phase_borrow_power": 1.0}).eval()
    weights = torch.zeros(1, 9, 2, 2, 3, 3)
    weights[:, 1, 0, 0] = 1.0
    weights[:, 3, 0, 1] = 1.0
    weights[:, 5, 1, 0] = 1.0
    weights[:, 7, 1, 1] = 1.0
    assert torch.equal(module._phase_consistent_weights(weights), weights)


def test_uncertain_phase_borrows_from_confident_siblings() -> None:
    module = SPCLCERDCRAUp(16, 8, {"phase_alpha_max": 0.50, "phase_borrow_power": 1.0, "phase_proto_floor": 0.01}).eval()
    weights = torch.zeros(1, 9, 2, 2, 1, 1)
    weights[:, 4, 0, 0] = 1.0
    weights[:, 4, 0, 1] = 1.0
    weights[:, 4, 1, 0] = 1.0
    weights[:, :, 1, 1] = 1.0 / 9.0
    stabilized = module._phase_consistent_weights(weights)
    assert stabilized[:, 4, 1, 1].item() > weights[:, 4, 1, 1].item()


def test_exact_phase_method_shapes() -> None:
    torch.manual_seed(3)
    module = SPCLCERDCRAUp(32, 16, {}).eval()
    deep = torch.randn(2, 32, 5, 7)
    lateral = torch.randn(2, 16, 10, 14)
    module._validate_inputs(deep, lateral)
    key_low = module._project_for_fp32_path(module.key_proj, deep)
    query = module._project_for_fp32_path(module.query_proj, lateral)
    key_patches = module._extract_local_patches(key_low)
    value_patches = module._extract_local_patches(deep)
    reassembled, weights = module._phase_correlate_and_reassemble(query, key_patches, value_patches)
    assert reassembled.shape == (2, 32, 10, 14)
    assert weights.shape == (2, 9, 10, 14)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 10, 14), atol=1e-5, rtol=1e-5)


def test_shapes_reload_and_cpu_autocast() -> None:
    torch.manual_seed(4)
    module = SPCLCERDCRAUp(32, 16, {}).eval()
    _set_nonzero_residual(module)
    for height, width in ((3, 5), (4, 4), (7, 9)):
        deep = torch.randn(2, 32, height, width)
        lateral = torch.randn(2, 16, height * 2, width * 2)
        output = module([deep, lateral])
        assert output.shape == (2, 32, height * 2, width * 2)
        assert torch.isfinite(output).all()
    clone = SPCLCERDCRAUp(32, 16, {}).eval()
    clone.load_state_dict(module.state_dict(), strict=True)
    deep = torch.randn(2, 32, 5, 7)
    lateral = torch.randn(2, 16, 10, 14)
    assert torch.equal(module([deep, lateral]), clone([deep, lateral]))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.isfinite(module([deep, lateral])).all()


def test_two_step_gradient() -> None:
    torch.manual_seed(5)
    module = SPCLCERDCRAUp(32, 16, {}).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    deep = torch.randn(2, 32, 5, 7, requires_grad=True)
    lateral = torch.randn(2, 16, 10, 14, requires_grad=True)
    module([deep, lateral]).square().mean().backward()
    assert module.residual_out.weight.grad is not None and torch.count_nonzero(module.residual_out.weight.grad).item() > 0
    assert module.key_proj.weight.grad is None or torch.count_nonzero(module.key_proj.weight.grad).item() == 0
    assert module.query_proj.weight.grad is None or torch.count_nonzero(module.query_proj.weight.grad).item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module([deep, lateral]).square().mean().backward()
    assert module.key_proj.weight.grad is not None and torch.count_nonzero(module.key_proj.weight.grad).item() > 0
    assert module.query_proj.weight.grad is not None and torch.count_nonzero(module.query_proj.weight.grad).item() > 0


def test_invalid_configs() -> None:
    invalid = (
        {"unknown_key": 1},
        {"phase_alpha_max": -0.1},
        {"phase_alpha_max": 1.1},
        {"phase_borrow_power": -0.1},
        {"phase_proto_floor": -0.1},
        {"phase_proto_floor": 0.0},
        {"phase_proto_floor": 1.1},
    )
    for config in invalid:
        try:
            SPCLCERDCRAUp(32, 16, config)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"Invalid config was accepted: {config}")


def test_yaml_contracts() -> None:
    common = None
    for path in SPC_YAMLS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        layer = payload["head"][6]
        assert layer[0] == [14, 12] and layer[2] == "SPCLCERDCRAUp"
        config = layer[3][0]
        assert set(config) == set(SPCLCERDCRAUp._DEFAULT_CONFIG)
        assert payload["head"][7] == [[-1, 12], 1, "Concat", [1]]
        assert payload["head"][10][2] == "nn.Upsample"
        assert payload["head"][-1] == [[23, 27, 31], 1, "Detect", ["nc"]]
        without_ablation = {key: value for key, value in config.items() if key not in {"phase_alpha_max", "phase_borrow_power"}}
        if common is None:
            common = without_ablation
        else:
            assert without_ablation == common


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
    parser.add_argument("--yaml", default=None, help="Optional SPC YAML path for a 640x640 parser/forward test.")
    args = parser.parse_args()
    test_initial_nearest_and_state_dict()
    test_alpha_zero_exactly_equals_lcer_l3()
    test_disabled_phase_consistency_exactly_equals_lcer_l3()
    test_phase_distribution_properties()
    test_confident_phases_are_protected()
    test_uncertain_phase_borrows_from_confident_siblings()
    test_exact_phase_method_shapes()
    test_shapes_reload_and_cpu_autocast()
    test_two_step_gradient()
    test_invalid_configs()
    test_yaml_contracts()
    if args.yaml:
        test_yaml_parser(args.yaml)
    print("ALL SPC-LCER-DCRA BLOCKING TESTS PASSED")


if __name__ == "__main__":
    main()
