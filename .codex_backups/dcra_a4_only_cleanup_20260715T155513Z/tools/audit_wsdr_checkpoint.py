import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import WSDRFuse


def rms(x):
    return x.float().square().mean().sqrt().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = YOLO(args.weights).model.to(device).float().eval()
    modules = [module for module in model.modules() if isinstance(module, WSDRFuse)]
    if len(modules) != 1:
        raise AssertionError(f"Expected one WSDRFuse, found {len(modules)}.")
    wsdr = modules[0]
    captured = {}

    def pre_hook(module, inputs):
        captured["x"] = inputs[0]

    handle = wsdr.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))
    handle.remove()
    if "x" not in captured:
        raise AssertionError("Failed to capture WSDRFuse inputs.")
    deep, lateral = captured["x"]
    with torch.no_grad():
        low, details, original_size = wsdr._decompose(lateral)
        semantic = wsdr.deep_proj(deep)
        if semantic.shape[-2:] != low.shape[-2:]:
            semantic = F.interpolate(semantic, size=low.shape[-2:], mode="bilinear", align_corners=False)
        gate = wsdr._compute_gate(semantic, low, details)
        low_new = low + gate * (semantic - low)
        details_new = wsdr._reweight_details(details)
        lateral_base = wsdr._reconstruct(low, details, original_size)
        lateral_new = wsdr._reconstruct(low_new, details_new, original_size)
    if not torch.isfinite(gate).all():
        raise AssertionError("Gate contains NaN or Inf.")
    if gate.min().item() < 0.0:
        raise AssertionError("Gate is below zero.")
    if gate.max().item() > wsdr.g_max + 1e-6:
        raise AssertionError("Gate exceeds g_max.")
    delta_rms = rms(lateral_new - lateral_base)
    if delta_rms <= 1e-8:
        raise AssertionError("WSDR semantic injection is numerically inactive.")
    report = {
        "adaptive": wsdr.adaptive,
        "use_hf_energy": wsdr.use_hf_energy,
        "decomposition": wsdr.decomposition,
        "hf_reweight": wsdr.hf_reweight,
        "g_max": wsdr.g_max,
        "gate_min": gate.min().item(),
        "gate_mean": gate.mean().item(),
        "gate_std": gate.float().std(unbiased=False).item(),
        "gate_max": gate.max().item(),
        "semantic_rms": rms(semantic),
        "low_rms": rms(low),
        "injection_delta_rms": delta_rms,
    }
    if wsdr.decomposition == "haar" and not wsdr.hf_reweight:
        _, details_after, _ = wsdr._haar_decompose(lateral_new)
        hf_errors = [
            (before.float() - after.float()).abs().max().item()
            for before, after in zip(details, details_after)
        ]
        report["hf_identity_max_abs_error"] = max(hf_errors)
        if report["hf_identity_max_abs_error"] > 2e-4:
            raise AssertionError(
                "Main WSDR changed Haar high-frequency subbands: "
                f"{report['hf_identity_max_abs_error']}."
            )
    print(report)


if __name__ == "__main__":
    main()
