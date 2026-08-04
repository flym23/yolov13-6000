#!/usr/bin/env python3
"""Dependency-free DCRA preflight validation for the training server."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import DCRAUp
from ultralytics.utils.torch_utils import initialize_weights


BASELINE_YAML = ROOT / "ultralytics/cfg/models/v13/yolov13-original.yaml"
MODEL_YAMLS = [
    ROOT / "ultralytics/cfg/models/v13/yolov13-dcra-tau020.yaml",
]


def make_inputs(device, batch=2, c_deep=64, c_lateral=32, deep_size=(20, 20)):
    deep_h, deep_w = deep_size
    deep = torch.randn(batch, c_deep, deep_h, deep_w, device=device)
    lateral = torch.randn(batch, c_lateral, deep_h * 2, deep_w * 2, device=device)
    return deep, lateral


def flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in flatten_tensors(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in flatten_tensors(item)]
    return []


def validate_module_math(device):
    torch.manual_seed(0)
    module = DCRAUp(c_deep=64, c_lateral=32).to(device).eval()
    deep, lateral = make_inputs(device)
    lateral_before = lateral.clone()
    with torch.no_grad():
        output = module([deep, lateral])
        nearest = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    if not torch.equal(output, nearest) or not torch.equal(lateral, lateral_before):
        raise AssertionError("Initial DCRA output/lateral invariant failed.")

    reference_module = DCRAUp(c_deep=16, c_lateral=8).to(device).eval()
    deep, lateral = make_inputs(device, batch=1, c_deep=16, c_lateral=8, deep_size=(6, 5))
    with torch.no_grad():
        key_low = reference_module.key_proj(deep.float())
        key_patches = reference_module._extract_local_patches(key_low)
        value_patches = reference_module._extract_local_patches(deep)
        query = reference_module.query_proj(lateral.float())
        optimized_values, optimized_weights = reference_module._phase_correlate_and_reassemble(
            query, key_patches, value_patches
        )
        key_high = reference_module._resize_patch_tensor(key_patches, lateral.shape[-2:])
        query_fp32 = F.normalize(query.float(), p=2, dim=1, eps=reference_module.eps)
        keys_fp32 = F.normalize(key_high.float(), p=2, dim=1, eps=reference_module.eps)
        reference_weights = torch.softmax(
            (query_fp32.unsqueeze(2) * keys_fp32).sum(dim=1) / reference_module.temperature,
            dim=1,
        )
        value_high = reference_module._resize_patch_tensor(value_patches, lateral.shape[-2:])
        reference_values = torch.einsum("bckhw,bkhw->bchw", value_high.float(), reference_weights)
    if not torch.allclose(optimized_weights, reference_weights, atol=2e-6, rtol=2e-6):
        raise AssertionError("Phase-aware weights differ from explicit reference.")
    if not torch.allclose(optimized_values, reference_values, atol=2e-6, rtol=2e-6):
        raise AssertionError("Phase-aware values differ from explicit reference.")

    _, _, weights, confidence = module._compute_alignment(*make_inputs(device))
    if weights.dtype != torch.float32 or confidence.dtype != torch.float32:
        raise AssertionError("Weights/confidence are not FP32.")
    if (weights.sum(dim=1) - 1.0).abs().max().item() >= 1e-5:
        raise AssertionError("Candidate softmax does not sum to one.")
    if confidence.min().item() < 0.0 or confidence.max().item() > 1.0:
        raise AssertionError("Entropy confidence is outside [0, 1].")

    if device.type == "cuda":
        half_module = DCRAUp(c_deep=64, c_lateral=32).to(device).half().eval()
        deep, lateral = make_inputs(device)
        deep, lateral = deep.half(), lateral.half()
        with torch.no_grad():
            half_output = half_module([deep, lateral])
            half_reference = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float()
        if half_output.dtype != torch.float16 or not torch.equal(half_output, half_reference.half()):
            raise AssertionError("FP16 validator-model nearest invariant failed.")


def validate_initialization():
    module = DCRAUp(c_deep=64, c_lateral=32)
    initialize_weights(module)
    if torch.count_nonzero(module.residual_out.weight).item() != 0:
        raise AssertionError("initialize_weights changed zero residual_out.")

    seed = 12345
    torch.manual_seed(seed)
    expected_a = nn.Conv2d(3, 8, 3)
    expected_b = nn.Conv2d(8, 8, 3)
    torch.manual_seed(seed)
    actual_a = nn.Conv2d(3, 8, 3)
    first = DCRAUp(c_deep=64, c_lateral=32)
    actual_b = nn.Conv2d(8, 8, 3)
    torch.manual_seed(seed)
    _ = nn.Conv2d(3, 8, 3)
    second = DCRAUp(c_deep=64, c_lateral=32)
    if not torch.equal(expected_a.weight, actual_a.weight) or not torch.equal(expected_b.weight, actual_b.weight):
        raise AssertionError("DCRA construction advanced the global CPU RNG.")
    if not torch.equal(first.key_proj.weight, second.key_proj.weight):
        raise AssertionError("DCRA module-local initialization is not repeatable.")


def validate_two_step_gradients(device):
    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    module = DCRAUp(c_deep=64, c_lateral=32).to(device).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.10)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        deep, lateral = make_inputs(device)
        deep.requires_grad_(True)
        lateral.requires_grad_(True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = module([deep, lateral])
            loss = (output.float() * torch.randn_like(output.float())).mean()
        loss.backward()
        if module.residual_out.weight.grad is None or module.residual_out.weight.grad.abs().sum().item() == 0.0:
            raise AssertionError(f"Step {step}: residual_out gradient is zero.")
        if step == 0:
            if module.key_proj.weight.grad.abs().sum().item() != 0.0:
                raise AssertionError("First-step key gradient must be blocked by zero residual_out.")
        else:
            if module.key_proj.weight.grad.abs().sum().item() == 0.0:
                raise AssertionError("Second-step key gradient is zero.")
            if module.query_proj.weight.grad.abs().sum().item() == 0.0:
                raise AssertionError("Second-step query gradient is zero.")
        optimizer.step()


def validate_models():
    profiles = []
    for yaml_path in MODEL_YAMLS:
        model = YOLO(str(yaml_path)).model.eval()
        if not isinstance(model.model[15], DCRAUp) or model.model[15].f != [14, 12]:
            raise AssertionError(f"Invalid DCRA layer 15 in {yaml_path.name}.")
        if model.model[18].__class__.__name__ != "FullPAD_Tunnel":
            raise AssertionError(f"Original layer 18 FullPAD_Tunnel changed in {yaml_path.name}.")
        if not isinstance(model.model[19], nn.Upsample):
            raise AssertionError(f"Original P4->P3 layer 19 Upsample changed in {yaml_path.name}.")
        if model.model[-1].f != [23, 27, 31]:
            raise AssertionError(f"Detect inputs changed in {yaml_path.name}.")
        if sum(isinstance(module, DCRAUp) for module in model.modules()) != 1:
            raise AssertionError(f"Expected one DCRAUp in {yaml_path.name}.")
        profiles.append({"yaml": yaml_path.name, "parameters": sum(p.numel() for p in model.parameters())})

    seed = 20260713
    torch.manual_seed(seed)
    baseline = YOLO(str(BASELINE_YAML)).model.eval()
    torch.manual_seed(seed)
    dcra = YOLO(str(MODEL_YAMLS[0])).model.eval()
    dcra_state = dcra.state_dict()
    for name, tensor in baseline.state_dict().items():
        if name not in dcra_state or not torch.equal(tensor, dcra_state[name]):
            raise AssertionError(f"D0 shared initial state differs at {name}.")
    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        baseline_output = flatten_tensors(baseline(sample))
        dcra_output = flatten_tensors(dcra(sample))
    if len(baseline_output) != len(dcra_output):
        raise AssertionError("D0/DCRA output tensor counts differ.")
    for index, (left, right) in enumerate(zip(baseline_output, dcra_output)):
        if not torch.equal(left, right):
            raise AssertionError(f"D0/DCRA initial output differs at tensor {index}.")
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    validate_initialization()
    validate_module_math(device)
    validate_two_step_gradients(device)
    profiles = validate_models()
    print(json.dumps({"status": "passed", "device": str(device), "models": profiles}, indent=2))


if __name__ == "__main__":
    main()
