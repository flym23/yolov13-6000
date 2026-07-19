import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import WSDRFuse
from ultralytics.utils.torch_utils import get_flops


DEFAULT_YAMLS = [
    "ultralytics/cfg/models/v13/yolov13_wsdr_fixed.yaml",
    "ultralytics/cfg/models/v13/yolov13_wsdr.yaml",
    "ultralytics/cfg/models/v13/yolov13_wsdr_no_hf.yaml",
    "ultralytics/cfg/models/v13/yolov13_wsdr_avgpool.yaml",
    "ultralytics/cfg/models/v13/yolov13_wsdr_hf_reweight.yaml",
]

EXPECTED_LAYERS = {
    9: ("HyperACE", [4, 6, 8]),
    12: ("FullPAD_Tunnel", [6, 9]),
    13: ("FullPAD_Tunnel", [4, 10]),
    14: ("FullPAD_Tunnel", [8, 11]),
    15: ("WSDRFuse", [14, 12]),
    16: ("DSC3k2", -1),
    17: ("FullPAD_Tunnel", [-1, 9]),
    18: ("FAARUp", 16),
    19: ("Concat", [-1, 13]),
    20: ("DSC3k2", -1),
    21: ("Conv", 10),
    22: ("FullPAD_Tunnel", [20, 21]),
    23: ("Conv", -1),
    24: ("Concat", [-1, 17]),
    25: ("DSC3k2", -1),
    26: ("FullPAD_Tunnel", [-1, 9]),
    27: ("Conv", 25),
    28: ("Concat", [-1, 14]),
    29: ("DSC3k2", -1),
    30: ("FullPAD_Tunnel", [-1, 11]),
    31: ("Detect", [22, 26, 30]),
}


def flatten_tensors(value):
    tensors = []
    if torch.is_tensor(value):
        tensors.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_tensors(item))
    return tensors


def scalarize(value):
    tensors = flatten_tensors(value)
    if not tensors:
        raise RuntimeError(f"Model output contains no tensors: {type(value).__name__}")
    total = None
    for tensor in tensors:
        term = tensor.float().square().mean()
        total = term if total is None else total + term
    return total


def max_output_difference(first, second):
    first_tensors = flatten_tensors(first)
    second_tensors = flatten_tensors(second)
    if len(first_tensors) != len(second_tensors):
        raise AssertionError(
            "Output tensor-count mismatch after reload: "
            f"{len(first_tensors)} vs {len(second_tensors)}."
        )
    maximum = 0.0
    for index, (left, right) in enumerate(zip(first_tensors, second_tensors)):
        if left.shape != right.shape:
            raise AssertionError(
                f"Output shape mismatch at tensor {index}: "
                f"{tuple(left.shape)} vs {tuple(right.shape)}."
            )
        maximum = max(maximum, (left.float() - right.float()).abs().max().item())
    return maximum


def assert_core_gradients(wsdr):
    required_prefixes = ["deep_up.", "deep_proj."]
    if wsdr.adaptive:
        required_prefixes.append("gate_head.")
    if wsdr.hf_reweight:
        required_prefixes.append("hf_gate.")
    matched = {prefix: 0 for prefix in required_prefixes}
    for name, parameter in wsdr.named_parameters():
        for prefix in required_prefixes:
            if name.startswith(prefix):
                matched[prefix] += 1
                if parameter.grad is None:
                    raise AssertionError(f"Missing gradient: {name}")
                if not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f"Non-finite gradient: {name}")
                if parameter.grad.abs().sum().item() == 0.0:
                    raise AssertionError(f"Zero gradient: {name}")
    for prefix, count in matched.items():
        if count == 0:
            raise AssertionError(f"No parameters matched required prefix {prefix!r}.")


def assert_model_structure(model):
    if len(model.model) != 32:
        raise AssertionError(f"Expected 32 model layers, found {len(model.model)}.")
    for index, (expected_type, expected_from) in EXPECTED_LAYERS.items():
        module = model.model[index]
        actual_type = module.type.rsplit(".", 1)[-1]
        if actual_type != expected_type or module.f != expected_from:
            raise AssertionError(
                f"Layer {index} mismatch: got type={actual_type}, from={module.f}; "
                f"expected type={expected_type}, from={expected_from}."
            )


def validate_one(yaml_path, device, imgsz, batch):
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    print(f"\n=== validating {yaml_path} ===")
    model = YOLO(str(yaml_path)).model.to(device)
    assert_model_structure(model)
    wsdr_modules = [module for module in model.modules() if isinstance(module, WSDRFuse)]
    if len(wsdr_modules) != 1:
        raise AssertionError(f"Expected exactly one WSDRFuse, found {len(wsdr_modules)}.")
    wsdr = wsdr_modules[0]
    captured = {}

    def output_hook(module, inputs, output):
        captured["input"] = inputs[0]
        captured["output"] = output

    handle = wsdr.register_forward_hook(output_hook)
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(torch.randn(batch, 3, imgsz, imgsz, device=device))
    loss = scalarize(output)
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite FP32 loss.")
    loss.backward()
    assert_core_gradients(wsdr)
    if "input" not in captured or "output" not in captured:
        raise AssertionError("WSDRFuse hook did not run.")
    deep, lateral = captured["input"]
    wsdr_output = captured["output"]
    if wsdr_output.shape[1] != deep.shape[1] + lateral.shape[1]:
        raise AssertionError(
            f"WSDRFuse output-channel mismatch: got {wsdr_output.shape[1]}, "
            f"expected {deep.shape[1] + lateral.shape[1]}."
        )
    if wsdr_output.shape[-2:] != lateral.shape[-2:]:
        raise AssertionError(
            f"WSDRFuse output-spatial mismatch: output={tuple(wsdr_output.shape)}, "
            f"lateral={tuple(lateral.shape)}."
        )
    handle.remove()

    if device.type == "cuda":
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            output_amp = model(torch.randn(batch, 3, imgsz, imgsz, device=device))
            loss_amp = scalarize(output_amp)
        if not torch.isfinite(loss_amp):
            raise AssertionError("Non-finite AMP loss.")
        loss_amp.backward()
        assert_core_gradients(wsdr)

    model.eval()
    x_eval = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        output_before = model(x_eval)
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "state.pt"
        torch.save(model.state_dict(), state_path)
        model_reloaded = YOLO(str(yaml_path)).model.to(device)
        state = torch.load(state_path, map_location=device)
        model_reloaded.load_state_dict(state, strict=True)
        model_reloaded.eval()
        with torch.no_grad():
            output_after = model_reloaded(x_eval)
    difference = max_output_difference(output_before, output_after)
    if difference > 1e-5:
        raise AssertionError(f"Reload output mismatch: max_abs_diff={difference}.")
    print(
        {
            "yaml": str(yaml_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "gflops_640": get_flops(model, imgsz=640),
            "wsdr_c_deep": wsdr.c_deep,
            "wsdr_c_lat": wsdr.c_lat,
            "adaptive": wsdr.adaptive,
            "use_hf_energy": wsdr.use_hf_energy,
            "decomposition": wsdr.decomposition,
            "hf_reweight": wsdr.hf_reweight,
            "reload_max_abs_diff": difference,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yamls", nargs="*", default=DEFAULT_YAMLS)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    for yaml_path in args.yamls:
        validate_one(yaml_path=yaml_path, device=device, imgsz=args.imgsz, batch=args.batch)
    print("\nAll WSDR model validations passed.")


if __name__ == "__main__":
    main()
