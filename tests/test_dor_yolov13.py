"""Focused production checks for the DOR modules, loss and all factorial topologies."""

from pathlib import Path

import torch
import yaml

from ultralytics import YOLO
from ultralytics.nn.modules.block import DCPR, OCARFuse, _dor_rms
from ultralytics.nn.modules.head import RQDDetect
from ultralytics.utils.loss import rqd_objectness_recall_loss
from tools.run_dor_matrix import STRUCTURES, VARIANTS, variant_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dcpr_identity_bound_and_two_step_release():
    module = DCPR(32, 64).train()
    shallow, deep = torch.randn(2, 32, 17, 19), torch.randn(2, 64, 5, 6)
    with torch.no_grad():
        assert torch.equal(module([shallow, deep]), deep)

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    basis_gradients = []
    for _ in range(2):
        optimizer.zero_grad()
        output = module([torch.randn_like(shallow), torch.randn_like(deep)])
        output.square().mean().backward()
        basis_gradients.append(float(module.degradation_bases.grad.abs().sum()))
        optimizer.step()
    assert basis_gradients[0] == 0.0 and basis_gradients[1] > 0.0

    with torch.no_grad():
        module.basis_gain.fill_(1.0)
        changed = module([shallow, deep])
    ratio = _dor_rms(changed - deep, module.eps) / _dor_rms(deep, module.eps).clamp_min(module.eps)
    assert torch.isfinite(changed).all() and ratio.max().item() <= module.max_scale + 1e-5


def test_ocar_identity_bound_and_two_step_release():
    module = OCARFuse(32, 48, 64).train()
    deep, lateral, base = torch.randn(2, 32, 13, 15), torch.randn(2, 48, 13, 15), torch.randn(2, 64, 13, 15)
    with torch.no_grad():
        assert torch.equal(module([deep, lateral, base]), base)

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    projection_gradients = []
    for _ in range(2):
        optimizer.zero_grad()
        output = module([torch.randn_like(deep), torch.randn_like(lateral), torch.randn_like(base)])
        output.square().mean().backward()
        projection_gradients.append(float(module.deep_proj.weight.grad.abs().sum()))
        optimizer.step()
    assert projection_gradients[0] == 0.0 and projection_gradients[1] > 0.0

    torch.nn.init.normal_(module.refine[-1].weight)
    with torch.no_grad():
        changed = module([deep, lateral, base])
    ratio = _dor_rms(changed - base, module.eps) / _dor_rms(base, module.eps).clamp_min(module.eps)
    assert torch.isfinite(changed).all() and ratio.max().item() <= module.max_residual_ratio + 1e-4


def test_dor_modules_stay_finite_after_half_cast():
    if not torch.cuda.is_available():
        return  # CPU Conv2d has no FP16 kernel; CUDA FP16 is enforced by dor_preflight.py.
    device = torch.device("cuda:0")
    dcpr = DCPR(32, 64).to(device).half().eval()
    ocar = OCARFuse(32, 48, 64).to(device).half().eval()
    with torch.no_grad():
        assert torch.isfinite(dcpr([torch.randn(1, 32, 9, 11, device=device).half(), torch.randn(1, 64, 3, 4, device=device).half()])).all()
        assert torch.isfinite(ocar([torch.randn(1, 32, 7, 9, device=device).half(), torch.randn(1, 48, 7, 9, device=device).half(), torch.randn(1, 64, 7, 9, device=device).half()])).all()


def test_rqd_bounded_score_and_objectness_loss_gradients():
    head = RQDDetect(4, {}, ch=(32, 64, 128)).eval()
    cls, quality = torch.randn(2, 4, 35), torch.randn(2, 1, 35)
    objectness = torch.randn(2, 1, 35)
    levels = torch.ones(1, 1, 35)
    output = head._calibrate_scores(cls, quality, objectness, levels)
    raw = cls.sigmoid()
    udq = head._calibrate_scores(cls, quality, torch.full_like(objectness, -100.0), levels)
    assert torch.all(output >= udq) and torch.all(output <= raw)

    logits = torch.randn(2, 31, 1, requires_grad=True)
    loss = rqd_objectness_recall_loss(logits, torch.rand(2, 31) > 0.75)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_dor_full_yaml_builds_registered_modules():
    model = YOLO(str(ROOT / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n-dor.yaml")).model
    module_types = {type(module).__name__ for module in model.modules()}
    assert {"DCPR", "OCARFuse", "RQDDetect"}.issubset(module_types)
    assert model.model[-1].no == 70


def test_all_factorial_yaml_snapshots_build_and_match_flags(tmp_path):
    for variant in VARIANTS:
        path = tmp_path / f"{variant}.yaml"
        path.write_text(yaml.safe_dump(variant_yaml(STRUCTURES[variant]), sort_keys=False), encoding="utf-8")
        model = YOLO(str(path)).model
        module_types = {type(module).__name__ for module in model.modules()}
        for name, enabled in STRUCTURES[variant].items():
            assert (name in module_types) is enabled
