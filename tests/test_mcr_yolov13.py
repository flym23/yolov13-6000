"""Focused integration tests for the production MCR modules and parser registration."""

from pathlib import Path

import torch
import yaml

from ultralytics import YOLO
from ultralytics.nn.modules.block import MCDRBlock, RCCFConcat, _mcr_centered_rms
from ultralytics.nn.modules.head import RCQDetect, _RCQDistributionQuality
from tools.run_mcr_matrix import STRUCTURES, VARIANTS, variant_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_mcdr_identity_flat_gate_and_two_step_opening():
    module = MCDRBlock(32).train()
    x = torch.randn(2, 32, 17, 19)
    with torch.no_grad():
        assert torch.equal(module(x), x)
    anis, _, _, _, support = module._structure_gates(torch.ones(2, module.hidden, 17, 19))
    assert anis.abs().max().item() <= 1e-7
    assert support.abs().max().item() <= 1e-7
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    opening = []
    for _ in range(2):
        optimizer.zero_grad()
        module(torch.randn(2, 32, 17, 19)).square().mean().backward()
        opening.append(module.in_proj[0].weight.grad.abs().sum().item())
        optimizer.step()
    assert opening[0] == 0.0 and opening[1] > 0.0


def test_mcdr_rms_bound():
    module = MCDRBlock(32, {"max_residual_ratio": 0.08}).eval()
    torch.nn.init.normal_(module.out.weight)
    x = torch.randn(2, 32, 21, 23)
    with torch.no_grad():
        output = module(x)
    ratio = _mcr_centered_rms(output - x, 1e-6) / _mcr_centered_rms(x, 1e-6)
    assert ratio.max().item() <= 0.081


def test_mcdr_structure_gates_stay_fp32_after_model_half():
    """Fixed Sobel buffers must be recast because ``Module.half`` casts buffers too."""
    module = MCDRBlock(32).half().eval()
    gates = module._structure_gates(torch.randn(1, module.hidden, 11, 13).half())
    assert all(gate.dtype is torch.float32 for gate in gates)


def test_rccf_exact_concat_lateral_preservation_and_amp_promotion():
    module = RCCFConcat(32, 48).eval()
    deep, lateral = torch.randn(2, 32, 13, 15), torch.randn(2, 48, 13, 15)
    with torch.no_grad():
        output = module([deep, lateral])
    assert torch.equal(output, torch.cat((deep, lateral), 1))
    assert torch.equal(output[:, 32:], lateral)
    with torch.no_grad():
        promoted = module([deep, lateral.half()])
    assert promoted.dtype == deep.dtype
    assert torch.equal(promoted[:, 32:], lateral.half().float())


def test_rcq_dfl_detach_and_exact_neutral_endpoint():
    quality = _RCQDistributionQuality(32).train()
    box = torch.randn(2, 64, 8, 8, requires_grad=True)
    feature = torch.randn(2, 32, 8, 8)
    quality(feature, box).square().mean().backward()
    assert box.grad is None or box.grad.abs().sum().item() == 0.0
    head = RCQDetect(4, {}, ch=(32, 64, 128)).eval()
    cls = torch.full((1, 4, 2, 2), -3.0)
    cls[:, 0] = 4.0
    certain_box = torch.full((1, 64, 2, 2), -6.0)
    certain_box.view(1, 4, 16, 2, 2)[:, :, 4] = 6.0
    neutral = torch.full((1, 1, 2, 2), head.quality_prior_logit)
    assert torch.allclose(head._calibrate_level_scores(0, cls, neutral, certain_box), cls.sigmoid(), atol=1e-6)


def test_mcr_full_yaml_builds_registered_modules():
    model = YOLO(str(ROOT / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n-mcr.yaml")).model
    module_types = {type(module).__name__ for module in model.modules()}
    assert {"MCDRBlock", "RCCFConcat", "RCQDetect"}.issubset(module_types)
    assert model.model[-1].no == 69


def test_all_factorial_yaml_snapshots_build_and_match_flags(tmp_path):
    for variant in VARIANTS:
        path = tmp_path / f"{variant}.yaml"
        path.write_text(yaml.safe_dump(variant_yaml(STRUCTURES[variant]), sort_keys=False), encoding="utf-8")
        model = YOLO(str(path)).model
        types = {type(module).__name__ for module in model.modules()}
        for name, enabled in STRUCTURES[variant].items():
            assert (name in types) is enabled
