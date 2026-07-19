#!/usr/bin/env python3
"""Audit head-aligned AG-DSB gates and sparse configuration in a checkpoint."""

import argparse
from pathlib import Path

import torch


EXPECTED_GATE_MODE = "head_aligned_direct_ste_v1"


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_float_list(tensor):
    return tensor.detach().float().cpu().reshape(-1).tolist()


def output_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from output_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from output_tensors(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--require-headwise-moved", action="store_true")
    parser.add_argument("--min-mean-move", type=float, default=1e-6)
    parser.add_argument("--min-max-move", type=float, default=1e-5)
    parser.add_argument("--per-head-threshold", type=float, default=1e-6)
    parser.add_argument("--min-moved-heads", type=int, default=2)
    parser.add_argument("--check-forward", action="store_true")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    model = checkpoint.get("model")
    if model is None:
        model = checkpoint.get("ema")
    if model is None:
        raise RuntimeError("Checkpoint contains neither model nor EMA.")

    records = []
    for name, module in model.named_modules():
        if not hasattr(module, "effective_eta"):
            continue
        gate_mode = getattr(module, "gate_mode", None)
        if gate_mode != EXPECTED_GATE_MODE:
            raise RuntimeError(
                f"Unexpected or legacy gate mode at {name}: {gate_mode!r}. "
                "Do not audit old scalar-gate checkpoints as head-aligned results."
            )

        eta_tensor = module.effective_eta().detach().float().cpu().reshape(-1)
        eta_init = float(getattr(module, "eta_init", 0.05))
        move_tensor = (eta_tensor - eta_init).abs()
        learnable = hasattr(module, "eta_head_bias")
        raw_tensor = (
            module.eta_head_bias.detach().float().cpu().reshape(-1) if learnable else None
        )
        raw_dtype = str(module.eta_head_bias.dtype) if learnable else None
        num_heads = int(getattr(module, "num_heads", -1))
        if learnable and eta_tensor.numel() != num_heads:
            raise RuntimeError(
                f"Effective eta count does not equal num_heads at {name}: "
                f"{eta_tensor.numel()} vs {num_heads}."
            )
        if not learnable and eta_tensor.numel() != 1:
            raise RuntimeError(f"Fixed eta must remain scalar at {name}, got {eta_tensor.numel()}.")

        records.append(
            {
                "name": name,
                "gate_mode": gate_mode,
                "learnable": learnable,
                "eta": to_float_list(eta_tensor),
                "raw": to_float_list(raw_tensor) if raw_tensor is not None else None,
                "raw_dtype": raw_dtype,
                "eta_init": eta_init,
                "eta_min": float(getattr(module, "eta_min", 0.01)),
                "eta_max": float(getattr(module, "eta_max", 0.10)),
                "mean_move": float(move_tensor.mean()),
                "max_move": float(move_tensor.max()),
                "moved_heads": int((move_tensor >= args.per_head_threshold).sum()),
                "topk": int(getattr(module, "topk", -1)),
                "num_hyperedges": int(getattr(module, "num_hyperedges", -1)),
                "num_heads": num_heads,
            }
        )

    if not records:
        raise RuntimeError("No AG-DSB head-gated modules found.")

    for record in records:
        print(
            f"{record['name']}: mode={record['gate_mode']}, learnable={record['learnable']}, "
            f"eta={record['eta']}, raw={record['raw']}, raw_dtype={record['raw_dtype']}, "
            f"eta_init={record['eta_init']:.8f}, mean_move={record['mean_move']:.8e}, "
            f"max_move={record['max_move']:.8e}, moved_heads={record['moved_heads']}/"
            f"{record['num_heads']}, topk={record['topk']}, "
            f"num_hyperedges={record['num_hyperedges']}"
        )
        eta_tensor = torch.tensor(record["eta"], dtype=torch.float32)
        if not torch.isfinite(eta_tensor).all():
            raise RuntimeError(f"Non-finite eta at {record['name']}.")
        if not (
            (eta_tensor >= record["eta_min"]).all()
            and (eta_tensor <= record["eta_max"]).all()
        ):
            raise RuntimeError(f"Effective eta is out of bounds at {record['name']}: {record['eta']}.")
        if record["topk"] > 0 and not record["topk"] < record["num_hyperedges"]:
            raise RuntimeError(
                f"Invalid sparse configuration at {record['name']}: "
                f"topk={record['topk']}, num_hyperedges={record['num_hyperedges']}."
            )
        if args.require_headwise_moved and record["learnable"]:
            if record["raw_dtype"] != "torch.float32":
                raise RuntimeError(
                    f"Learnable head-gate precision was lost at {record['name']}: "
                    f"checkpoint dtype={record['raw_dtype']}."
                )
            if not (
                record["mean_move"] >= args.min_mean_move
                or record["max_move"] >= args.min_max_move
            ):
                raise RuntimeError(
                    f"Head gates did not move enough at {record['name']}: "
                    f"mean_move={record['mean_move']:.8e}, max_move={record['max_move']:.8e}."
                )
            if record["moved_heads"] < args.min_moved_heads:
                raise RuntimeError(
                    f"Too few head gates moved at {record['name']}: "
                    f"moved_heads={record['moved_heads']}, required={args.min_moved_heads}."
                )

    if args.check_forward:
        model = model.float().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 640, 640))
        tensors = list(output_tensors(output))
        if not tensors or not all(torch.isfinite(item).all() for item in tensors):
            raise RuntimeError("Checkpoint forward produced missing or non-finite tensors.")
        print("AG-DSB checkpoint FP32 forward passed.")

    print("AG-DSB head-aligned gate audit passed.")


if __name__ == "__main__":
    main()
