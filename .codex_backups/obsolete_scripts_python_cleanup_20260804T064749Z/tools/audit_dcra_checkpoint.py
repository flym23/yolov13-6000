#!/usr/bin/env python3
"""Audit trained DCRA checkpoint topology, numerical invariants, and learned residual activation."""

import argparse
import json

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import DCRAUp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights).model.to(args.device).eval()
    modules = [module for module in model.modules() if isinstance(module, DCRAUp)]
    if len(modules) != 1:
        raise AssertionError(f"Expected one DCRAUp, found {len(modules)}.")
    if not isinstance(model.model[15], DCRAUp) or model.model[15].f != [14, 12]:
        raise AssertionError("Layer 15 is not the required DCRAUp([14, 12]).")
    if model.model[18].__class__.__name__ != "FullPAD_Tunnel":
        raise AssertionError("Layer 18 is not the original P4 FullPAD_Tunnel.")
    if model.model[19].__class__.__name__ != "Upsample":
        raise AssertionError("Layer 19 is not the original P4->P3 nn.Upsample.")
    if model.model[-1].f != [23, 27, 31]:
        raise AssertionError(f"Unexpected Detect inputs: {model.model[-1].f}.")

    dcra = modules[0]
    captured = {}

    def pre_hook(_module, inputs):
        captured["inputs"] = inputs[0]

    handle = dcra.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        output = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device))
    handle.remove()
    if "inputs" not in captured:
        raise AssertionError("DCRAUp forward hook did not capture inputs.")
    deep, lateral = captured["inputs"]
    with torch.no_grad():
        _, _, weights, confidence = dcra._compute_alignment(deep, lateral)

    if weights.dtype != torch.float32 or confidence.dtype != torch.float32:
        raise AssertionError("DCRA weights/confidence are not FP32.")
    if not torch.isfinite(weights).all() or not torch.isfinite(confidence).all():
        raise AssertionError("DCRA weights/confidence contain non-finite values.")
    weight_sum_error = (weights.sum(dim=1) - 1.0).abs().max().item()
    if weight_sum_error > 1e-5:
        raise AssertionError(f"DCRA softmax sum error is too large: {weight_sum_error}.")
    residual_weight_norm = dcra.residual_out.weight.float().norm().item()
    if residual_weight_norm == 0.0:
        raise AssertionError("Trained DCRA residual_out is still exactly zero.")

    def finite_tensors(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tensors(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tensors(item) for item in value)
        return True

    if not finite_tensors(output):
        raise AssertionError("Model output contains non-finite values.")
    print(
        json.dumps(
            {
                "weights": args.weights,
                "kernel_size": dcra.kernel_size,
                "temperature": dcra.temperature,
                "use_entropy": dcra.use_entropy,
                "use_lateral_guidance": dcra.use_lateral_guidance,
                "residual_out_norm": residual_weight_norm,
                "weight_sum_max_error": weight_sum_error,
                "confidence_min": confidence.min().item(),
                "confidence_max": confidence.max().item(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
