"""Blocking repository-level tests for SAMR-LCER-DCRA integration.

Run from the repository root:
    python test_samr_lcer_dcra.py
    python test_samr_lcer_dcra.py --yaml ultralytics/cfg/models/v13/yolov13-samr-lcer-dcra-s3-adaptive.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ultralytics.nn.modules.block import LCERDCRAUp, SAMRLCERDCRAUp


SAMR_YAMLS = (
    Path("ultralytics/cfg/models/v13/yolov13-samr-lcer-dcra-s1-matched.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-samr-lcer-dcra-s2-raw.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-samr-lcer-dcra-s3-adaptive.yaml"),
)


def _set_nonzero_residual(module: torch.nn.Module, std: float = 0.05) -> None:
    with torch.no_grad():
        module.residual_out.weight.normal_(mean=0.0, std=std)


def _copy_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)


def _lcer_l3_config(**overrides) -> dict:
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


def test_initial_nearest_and_state_dict() -> None:
    torch.manual_seed(0)
    module = SAMRLCERDCRAUp(64, 32, {}).eval()
    lcer = LCERDCRAUp(64, 32, _lcer_l3_config()).eval()
    deep, lateral = _inputs()
    output = module([deep, lateral])
    nearest = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float()
    assert torch.equal(output, nearest), (output - nearest).abs().max().item()
    assert set(module.state_dict()) == set(lcer.state_dict())
    assert sum(parameter.numel() for parameter in module.parameters()) == sum(
        parameter.numel() for parameter in lcer.parameters()
    )
    assert len(list(module.buffers())) == len(list(lcer.buffers()))


def test_matched_endpoint_exactly_equals_lcer_l3() -> None:
    torch.manual_seed(1)
    lcer = LCERDCRAUp(64, 32, _lcer_l3_config()).eval()
    samr = SAMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="matched", moment_relax_max=0.0)).eval()
    _copy_state(lcer, samr)
    _set_nonzero_residual(lcer)
    with torch.no_grad():
        samr.residual_out.weight.copy_(lcer.residual_out.weight)
    deep, lateral = _inputs()
    assert torch.equal(lcer([deep, lateral]), samr([deep, lateral]))


def test_raw_endpoint_exactly_equals_no_moment_lcer() -> None:
    torch.manual_seed(2)
    raw_lcer = LCERDCRAUp(64, 32, _lcer_l3_config(preserve_moments=False)).eval()
    samr = SAMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="raw", moment_relax_max=1.0)).eval()
    _copy_state(raw_lcer, samr)
    _set_nonzero_residual(raw_lcer)
    with torch.no_grad():
        samr.residual_out.weight.copy_(raw_lcer.residual_out.weight)
    deep, lateral = _inputs()
    assert torch.equal(raw_lcer([deep, lateral]), samr([deep, lateral]))


def test_zero_relaxation_exactly_equals_matched_endpoint() -> None:
    torch.manual_seed(3)
    matched = SAMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="matched", moment_relax_max=0.0)).eval()
    adaptive_zero = SAMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="adaptive", moment_relax_max=0.0)).eval()
    _copy_state(matched, adaptive_zero)
    _set_nonzero_residual(matched)
    with torch.no_grad():
        adaptive_zero.residual_out.weight.copy_(matched.residual_out.weight)
    deep, lateral = _inputs()
    assert torch.equal(matched([deep, lateral]), adaptive_zero([deep, lateral]))


def test_effective_support_and_relaxation_properties() -> None:
    module = SAMRLCERDCRAUp(
        16,
        8,
        _lcer_l3_config(moment_mode="adaptive", moment_relax_max=0.50, support_reference=0.10),
    ).eval()
    confidence = torch.ones(1, 1, 4, 4)
    uniform = torch.ones(1, 2, 4, 4)
    compact = torch.zeros(1, 2, 4, 4)
    compact[:, :, 0, 0] = 1.0
    empty = torch.zeros(1, 2, 4, 4)
    uniform_support = module._compute_effective_support(uniform, confidence)
    compact_support = module._compute_effective_support(compact, confidence)
    empty_support = module._compute_effective_support(empty, confidence)
    assert torch.allclose(uniform_support, torch.ones_like(uniform_support), atol=1e-6, rtol=1e-6)
    assert torch.allclose(compact_support, torch.full_like(compact_support, 1.0 / 16.0), atol=1e-6, rtol=1e-6)
    assert torch.equal(empty_support, torch.zeros_like(empty_support))
    assert torch.allclose(module._compute_moment_relaxation(uniform, confidence), torch.full((1, 2, 1, 1), 0.50))
    assert torch.all(module._compute_moment_relaxation(compact, confidence) < 0.50)


def test_shapes_reload_and_cpu_autocast() -> None:
    torch.manual_seed(4)
    module = SAMRLCERDCRAUp(32, 16, {}).eval()
    _set_nonzero_residual(module)
    for height, width in ((3, 5), (4, 4), (7, 9)):
        deep, lateral = _inputs(32, 16, height, width)
        output = module([deep, lateral])
        assert output.shape == (2, 32, height * 2, width * 2)
        assert torch.isfinite(output).all()
    clone = SAMRLCERDCRAUp(32, 16, {}).eval()
    clone.load_state_dict(module.state_dict(), strict=True)
    deep, lateral = _inputs(32, 16)
    assert torch.equal(module([deep, lateral]), clone([deep, lateral]))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.isfinite(module([deep, lateral])).all()


def test_two_step_gradient() -> None:
    torch.manual_seed(5)
    module = SAMRLCERDCRAUp(32, 16, {}).train()
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
        {"moment_mode": "unknown"},
        {"moment_relax_max": -0.1},
        {"moment_relax_max": 1.1},
        {"support_reference": 0.0},
        {"support_reference": 1.1},
        {"support_power": 0.0},
        {"preserve_moments": False},
    )
    for config in invalid:
        try:
            SAMRLCERDCRAUp(32, 16, config)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"Invalid config was accepted: {config}")


def test_yaml_contracts() -> None:
    common = None
    expected = {
        "yolov13-samr-lcer-dcra-s1-matched.yaml": ("matched", 0.0),
        "yolov13-samr-lcer-dcra-s2-raw.yaml": ("raw", 1.0),
        "yolov13-samr-lcer-dcra-s3-adaptive.yaml": ("adaptive", 0.50),
    }
    for path in SAMR_YAMLS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        layer = payload["head"][6]
        assert layer[0] == [14, 12] and layer[2] == "SAMRLCERDCRAUp"
        config = layer[3][0]
        assert set(config) == set(SAMRLCERDCRAUp._DEFAULT_CONFIG)
        assert (config["moment_mode"], float(config["moment_relax_max"])) == expected[path.name]
        assert payload["head"][7] == [[-1, 12], 1, "Concat", [1]]
        assert payload["head"][10][2] == "nn.Upsample"
        assert payload["head"][-1] == [[23, 27, 31], 1, "Detect", ["nc"]]
        without_ablation = {key: value for key, value in config.items() if key not in {"moment_mode", "moment_relax_max"}}
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
    parser.add_argument("--yaml", default=None, help="Optional SAMR YAML path for a 640x640 parser/forward test.")
    args = parser.parse_args()
    test_initial_nearest_and_state_dict()
    test_matched_endpoint_exactly_equals_lcer_l3()
    test_raw_endpoint_exactly_equals_no_moment_lcer()
    test_zero_relaxation_exactly_equals_matched_endpoint()
    test_effective_support_and_relaxation_properties()
    test_shapes_reload_and_cpu_autocast()
    test_two_step_gradient()
    test_invalid_configs()
    test_yaml_contracts()
    if args.yaml:
        test_yaml_parser(args.yaml)
    print("ALL SAMR-LCER-DCRA BLOCKING TESTS PASSED")


if __name__ == "__main__":
    main()
