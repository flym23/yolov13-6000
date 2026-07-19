#!/usr/bin/env python3
"""Run mandatory AG-DSB head-gate, stability, parser and model audits."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import AGDSBAdaHGConv, AGDSBHyperACE, HyperACE
from ultralytics.nn.modules.block import AdaHGConv
from ultralytics.utils.torch_utils import capture_fp32_parameters, restore_fp32_parameters, strip_optimizer


CONFIGS = {
    "yolov13_ag_dsb_dense_fixed.yaml": (0, 1, False, True),
    "yolov13_ag_dsb_dense.yaml": (0, 1, True, True),
    "yolov13_ag_dsb_topk2.yaml": (2, 1, True, True),
    "yolov13_ag_dsb_topk3.yaml": (3, 1, True, True),
    "yolov13_ag_dsb_topk2_no_norm.yaml": (2, 1, True, False),
    "yolov13_ag_dsb_topk2_both.yaml": (2, 2, True, True),
}


def full_hyperace_tests():
    seed = 1234
    torch.manual_seed(seed)
    base = HyperACE(c1=64, c2=256, n=2, num_hyperedges=8)
    base_next_random = torch.rand(8)
    torch.manual_seed(seed)
    candidate = AGDSBHyperACE(
        c1=64, c2=256, n=2, num_hyperedges=8, dsb_topk=2, learnable_eta=False
    )
    candidate_next_random = torch.rand(8)
    assert torch.equal(base_next_random, candidate_next_random)
    candidate_state = candidate.state_dict()
    for key, value in base.state_dict().items():
        assert key in candidate_state and torch.equal(value, candidate_state[key]), key

    modules = [module for module in candidate.modules() if isinstance(module, AGDSBAdaHGConv)]
    assert len(modules) == 1 and tuple(modules[0].effective_eta().shape) == (1,)
    modules[0].eta_fixed.zero_()
    base.eval()
    candidate.eval()
    inputs = [torch.randn(2, 64, 16, 16), torch.randn(2, 64, 8, 8), torch.randn(2, 128, 4, 4)]
    with torch.no_grad():
        output_base = base(inputs)
        output_candidate = candidate(inputs)
    max_diff = (output_base - output_candidate).abs().max().item()
    assert torch.allclose(output_base, output_candidate, atol=1e-6, rtol=1e-6)
    print(f"HyperACE RNG/fixed-zero compatibility passed; max_diff={max_diff:.3e}")


def module_tests():
    torch.manual_seed(0)
    module = AGDSBAdaHGConv(embed_dim=64, num_hyperedges=4, num_heads=4, topk=2)
    assert module.gate_mode == "head_aligned_direct_ste_v1"
    assert tuple(module.eta_head_bias.shape) == (4,)
    assert tuple(module.effective_eta().shape) == (4,)
    assert torch.allclose(module.effective_eta(), torch.full((4,), 0.05), atol=1e-7, rtol=0)

    module.zero_grad(set_to_none=True)
    module.effective_eta().sum().backward()
    assert torch.allclose(module.eta_head_bias.grad, torch.ones(4), atol=1e-7, rtol=0)
    with torch.no_grad():
        module.eta_head_bias.copy_(torch.tensor([-1.0, 0.02, 0.08, 1.0]))
    bounded = module.effective_eta()
    assert torch.allclose(bounded, torch.tensor([0.01, 0.02, 0.08, 0.10]), atol=1e-7, rtol=0)
    module.zero_grad(set_to_none=True)
    bounded.sum().backward()
    assert torch.allclose(module.eta_head_bias.grad, torch.ones(4), atol=1e-7, rtol=0)

    with torch.no_grad():
        module.eta_head_bias.copy_(torch.tensor([0.02, 0.04, 0.06, 0.08]))
    mapped = module._apply_head_aligned_eta(torch.ones(1, 3, 64)).reshape(1, 3, 4, 16)
    assert torch.allclose(mapped.mean((0, 1, 3)), torch.tensor([0.02, 0.04, 0.06, 0.08]))

    x = torch.randn(2, 100, 64, requires_grad=True)
    weights = torch.cat(tuple(torch.full((16,), float(i)) for i in range(1, 5))).reshape(1, 1, 64)
    module.zero_grad(set_to_none=True)
    (module(x) * weights).square().mean().backward()
    gradient = module.eta_head_bias.grad.detach().float()
    assert torch.isfinite(gradient).all() and (gradient.abs() > 0).sum().item() >= 2
    assert gradient.std().item() > 0

    with torch.no_grad():
        logits = module.edge_generator(x.detach())
        a_v2e = module._dense_v2e_weights(logits)
        a_e2v = module._build_e2v_weights(logits)
        assert torch.allclose(a_v2e.float().sum(1), torch.ones_like(a_v2e.float().sum(1)), atol=1e-5)
        assert torch.allclose(a_e2v.float().sum(2), torch.ones_like(a_e2v.float().sum(2)), atol=1e-5)
        nonzero = (a_e2v.float() > 0).sum(2)
        assert int(nonzero.min()) == int(nonzero.max()) == 2
        edge = module.edge_proj(torch.bmm(a_v2e.transpose(1, 2), x.detach()))
        base = torch.bmm(a_v2e, edge)
        delta = module._normalize_message_delta(base, torch.bmm(a_e2v, edge) - base)
        base_power = base.float().square().mean(-1)
        delta_power = delta.float().square().mean(-1)
        assert torch.all(delta_power <= base_power + 1e-8)
        ratio = torch.sqrt(delta_power / (base_power + module.eps)).max().item()
        assert ratio <= 1.001
        zero_delta = module._normalize_message_delta(torch.zeros_like(base), torch.randn_like(base))
        assert torch.allclose(zero_delta, torch.zeros_like(zero_delta), atol=1e-7, rtol=0)

    calls = {"count": 0}
    handle = module.node_proj.register_forward_hook(
        lambda _module, _inputs, _output: calls.__setitem__("count", calls["count"] + 1)
    )
    with torch.no_grad():
        module(torch.randn(2, 100, 64))
    handle.remove()
    assert calls["count"] == 1

    fixed = AGDSBAdaHGConv(
        embed_dim=64, num_hyperedges=4, num_heads=4, topk=2, learnable_eta=False
    )
    base_module = AdaHGConv(embed_dim=64, num_hyperedges=4, num_heads=4)
    result = fixed.load_state_dict(base_module.state_dict(), strict=False)
    assert set(result.missing_keys) == {"eta_fixed"} and not result.unexpected_keys
    assert tuple(fixed.effective_eta().shape) == (1,)
    delta = torch.randn(2, 100, 64)
    assert torch.allclose(fixed._apply_head_aligned_eta(delta), delta * 0.05)

    try:
        AGDSBAdaHGConv(embed_dim=64, num_hyperedges=4, num_heads=4, topk=4)
    except ValueError:
        pass
    else:
        raise AssertionError("topk == num_hyperedges must be rejected")
    print(f"Module head gate/gradient/RMS/top-k/single-projection tests passed; gradient={gradient.tolist()}")


def active_gate_30_steps():
    torch.manual_seed(0)
    module = AGDSBAdaHGConv(embed_dim=64, num_hyperedges=4, num_heads=4, topk=0)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    before = module.effective_eta().detach().clone()
    gradient_steps = torch.zeros(module.num_heads, dtype=torch.int64)
    channel_weights = torch.cat(tuple(torch.full((16,), 1.0 + 0.5 * i) for i in range(4))).reshape(1, 1, 64)
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = (module(torch.randn(2, 100, 64)) * channel_weights).square().mean()
        loss.backward()
        gradient = module.eta_head_bias.grad.detach().float()
        assert torch.isfinite(gradient).all()
        gradient_steps += (gradient.abs() > 0).to(torch.int64)
        optimizer.step()
    after = module.effective_eta().detach()
    move = (after - before).abs()
    assert (gradient_steps > 0).sum().item() >= 2
    assert move.max().item() > 1e-6 and (move >= 1e-7).sum().item() >= 2
    assert (after >= 0.01).all() and (after <= 0.10).all()
    print(f"30-step head gate passed; before={before.tolist()} after={after.tolist()}")


def checkpoint_precision_tests():
    module = AGDSBAdaHGConv(embed_dim=64, num_hyperedges=4, num_heads=4, topk=2).eval()
    with torch.no_grad():
        module.eta_head_bias.add_(torch.tensor([2e-5, -3e-5, 4e-5, -5e-5]))
    expected = module.eta_head_bias.detach().clone()
    controls = capture_fp32_parameters(module, suffixes=("eta_bias", "eta_head_bias"))
    module.half()
    module.float()
    restore_fp32_parameters(module, controls)
    assert module.eta_head_bias.dtype is torch.float32 and torch.equal(module.eta_head_bias, expected)

    checkpoint_model = deepcopy(module)
    controls = capture_fp32_parameters(checkpoint_model, suffixes=("eta_bias", "eta_head_bias"))
    checkpoint_model.half()
    restore_fp32_parameters(checkpoint_model, controls)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gate_precision.pt"
        torch.save({"model": checkpoint_model, "ema": None, "train_args": {}}, path)
        strip_optimizer(path)
        restored = torch.load(path, map_location="cpu", weights_only=False)["model"].eta_head_bias
    assert restored.dtype is torch.float32 and torch.equal(restored, expected)
    print("FP32 head-gate validation/save/strip preservation passed")


def model_audits(run_forward=False):
    cfg_root = ROOT / "ultralytics/cfg/models/v13"
    norm_types = tuple(value for key, value in nn.__dict__.items() if "Norm" in key)
    for filename, (topk, branches, learnable, normalize) in CONFIGS.items():
        model = YOLO(str(cfg_root / filename)).model
        modules = [module for module in model.modules() if isinstance(module, AGDSBAdaHGConv)]
        assert len(modules) == branches
        for module in modules:
            assert module.gate_mode == "head_aligned_direct_ste_v1"
            assert module.num_hyperedges == 4 and module.topk == topk
            assert module.learnable_eta is learnable and module.normalize_delta is normalize
            expected_shape = (module.num_heads,) if learnable else (1,)
            assert tuple(module.effective_eta().shape) == expected_shape
        if learnable:
            gate_names = [name for name, _ in model.named_parameters() if name.endswith("eta_head_bias")]
            no_decay, decay = [], []
            for module_name, child in model.named_modules():
                for param_name, _ in child.named_parameters(recurse=False):
                    fullname = f"{module_name}.{param_name}" if module_name else param_name
                    (no_decay if "bias" in fullname or isinstance(child, norm_types) else decay).append(fullname)
            assert len(gate_names) == branches
            assert all(name in no_decay and name not in decay for name in gate_names)
        if run_forward:
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            sample = torch.randn(1, 3, 640, 640, device=device)
            with torch.no_grad():
                output = model(sample)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        amp_output = model(sample)
                    assert amp_output is not None
            tensors = output if isinstance(output, (list, tuple)) else [output]
            assert all(torch.isfinite(item).all() for item in tensors if torch.is_tensor(item))
        print(f"{filename}: E=4 topk={topk} branches={branches} learnable={learnable} PASS")


if __name__ == "__main__":
    full_hyperace_tests()
    module_tests()
    active_gate_30_steps()
    checkpoint_precision_tests()
    model_audits(run_forward=True)
    print("All AG-DSB head-aligned verification tests passed.")
