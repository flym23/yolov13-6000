"""Blocking repository-level tests for LSMR-LCER-DCRA integration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ultralytics.nn.modules.block import LCERDCRAUp, LSMRLCERDCRAUp, SAMRLCERDCRAUp


LSMR_YAMLS = (
    Path("ultralytics/cfg/models/v13/yolov13-lsmr-lcer-dcra-r1-matched.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-lsmr-lcer-dcra-r2-global.yaml"),
    Path("ultralytics/cfg/models/v13/yolov13-lsmr-lcer-dcra-r3-local.yaml"),
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
    lsmr = LSMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="local")).eval()
    lcer = LCERDCRAUp(64, 32, _lcer_l3_config()).eval()
    deep, lateral = _inputs()
    output = lsmr([deep, lateral])
    nearest = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float()
    assert torch.equal(output, nearest), (output - nearest).abs().max().item()
    assert set(lsmr.state_dict()) == set(lcer.state_dict())
    assert sum(parameter.numel() for parameter in lsmr.parameters()) == sum(
        parameter.numel() for parameter in lcer.parameters()
    )
    assert len(list(lsmr.buffers())) == len(list(lcer.buffers()))


def test_matched_and_raw_endpoints() -> None:
    torch.manual_seed(1)
    lcer = LCERDCRAUp(64, 32, _lcer_l3_config()).eval()
    matched = LSMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="matched")).eval()
    _copy_state(lcer, matched)
    _set_nonzero_residual(lcer)
    with torch.no_grad():
        matched.residual_out.weight.copy_(lcer.residual_out.weight)
    deep, lateral = _inputs()
    assert torch.equal(lcer([deep, lateral]), matched([deep, lateral]))

    raw_lcer = LCERDCRAUp(64, 32, _lcer_l3_config(preserve_moments=False)).eval()
    raw = LSMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="raw")).eval()
    _copy_state(raw_lcer, raw)
    _set_nonzero_residual(raw_lcer)
    with torch.no_grad():
        raw.residual_out.weight.copy_(raw_lcer.residual_out.weight)
    assert torch.equal(raw_lcer([deep, lateral]), raw([deep, lateral]))


def test_zero_relaxation_exactly_equals_lcer() -> None:
    torch.manual_seed(2)
    lcer = LCERDCRAUp(64, 32, _lcer_l3_config()).eval()
    zero = LSMRLCERDCRAUp(64, 32, _lcer_l3_config(moment_mode="local", moment_relax_max=0.0)).eval()
    _copy_state(lcer, zero)
    _set_nonzero_residual(lcer)
    with torch.no_grad():
        zero.residual_out.weight.copy_(lcer.residual_out.weight)
    deep, lateral = _inputs()
    assert torch.equal(lcer([deep, lateral]), zero([deep, lateral]))


def test_global_endpoint_exactly_equals_samr() -> None:
    torch.manual_seed(3)
    samr = SAMRLCERDCRAUp(
        64, 32, _lcer_l3_config(moment_mode="adaptive", moment_relax_max=0.50, support_reference=0.10)
    ).eval()
    global_lsmr = LSMRLCERDCRAUp(
        64, 32, _lcer_l3_config(moment_mode="global", moment_relax_max=0.50, support_reference=0.10)
    ).eval()
    _copy_state(samr, global_lsmr)
    _set_nonzero_residual(samr)
    with torch.no_grad():
        global_lsmr.residual_out.weight.copy_(samr.residual_out.weight)
    deep, lateral = _inputs()
    assert torch.equal(samr([deep, lateral]), global_lsmr([deep, lateral]))


def test_local_support_and_interpolation_properties() -> None:
    module = LSMRLCERDCRAUp(4, 4, _lcer_l3_config(moment_mode="local", moment_relax_max=0.50)).eval()
    confidence = torch.ones(1, 1, 9, 9)
    compact = torch.zeros(1, 4, 9, 9)
    compact[:, :, 4, 4] = 1.0
    broad = torch.ones(1, 4, 9, 9)
    compact_relax = module._compute_local_relaxation(compact, confidence)
    broad_relax = module._compute_local_relaxation(broad, confidence)
    assert compact_relax.shape == compact.shape and broad_relax.shape == broad.shape
    assert torch.isfinite(compact_relax).all() and torch.isfinite(broad_relax).all()
    assert compact_relax.min() >= 0.0 and broad_relax.min() >= 0.0
    assert compact_relax.max() <= module.moment_relax_max + 1e-7
    assert broad_relax.max() <= module.moment_relax_max + 1e-7
    assert broad_relax.mean() > compact_relax.mean()

    torch.manual_seed(4)
    base = torch.randn(2, 4, 10, 14)
    residual = torch.randn(2, 4, 10, 14)
    random_confidence = torch.rand(2, 1, 10, 14)
    matched = module._moment_preserving_residual(base, residual)
    relaxation = module._compute_local_relaxation(residual.float() - matched.float(), random_confidence)
    expected = matched.float() + relaxation * (residual.float() - matched.float())
    assert torch.equal(module._local_support_moment_residual(base, residual, random_confidence).float(), expected)


def test_shapes_reload_autocast_and_two_step_gradient() -> None:
    torch.manual_seed(5)
    module = LSMRLCERDCRAUp(32, 16, _lcer_l3_config(moment_mode="local")).eval()
    _set_nonzero_residual(module)
    for height, width in ((3, 5), (4, 4), (7, 9)):
        deep, lateral = _inputs(32, 16, height, width)
        output = module([deep, lateral])
        assert output.shape == (2, 32, height * 2, width * 2)
        assert torch.isfinite(output).all()
    clone = LSMRLCERDCRAUp(32, 16, _lcer_l3_config(moment_mode="local")).eval()
    clone.load_state_dict(module.state_dict(), strict=True)
    deep, lateral = _inputs(32, 16)
    assert torch.equal(module([deep, lateral]), clone([deep, lateral]))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.isfinite(module([deep, lateral])).all()

    gradient_module = LSMRLCERDCRAUp(32, 16, _lcer_l3_config(moment_mode="local")).train()
    optimizer = torch.optim.SGD(gradient_module.parameters(), lr=0.01)
    deep = torch.randn(2, 32, 5, 7, requires_grad=True)
    lateral = torch.randn(2, 16, 10, 14, requires_grad=True)
    gradient_module([deep, lateral]).square().mean().backward()
    assert gradient_module.residual_out.weight.grad is not None and torch.count_nonzero(gradient_module.residual_out.weight.grad).item() > 0
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
            LSMRLCERDCRAUp(32, 16, config)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"Invalid config was accepted: {config}")
    expected_modes = ("matched", "global", "local")
    common = None
    for path, expected_mode in zip(LSMR_YAMLS, expected_modes):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        layer = payload["head"][6]
        assert layer[0] == [14, 12] and layer[2] == "LSMRLCERDCRAUp"
        config = layer[3][0]
        assert set(config) == set(LSMRLCERDCRAUp._DEFAULT_CONFIG)
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
    parser.add_argument("--yaml", default=None, help="Optional LSMR YAML path for a 640x640 parser/forward test.")
    args = parser.parse_args()
    test_initial_nearest_and_state_dict()
    test_matched_and_raw_endpoints()
    test_zero_relaxation_exactly_equals_lcer()
    test_global_endpoint_exactly_equals_samr()
    test_local_support_and_interpolation_properties()
    test_shapes_reload_autocast_and_two_step_gradient()
    test_invalid_configs_and_yaml_contracts()
    if args.yaml:
        test_yaml_parser(args.yaml)
    print("ALL LSMR-LCER-DCRA BLOCKING TESTS PASSED")


if __name__ == "__main__":
    main()
