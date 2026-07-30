from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ultralytics

if PROJECT_ROOT not in Path(ultralytics.__file__).resolve().parents:
    raise RuntimeError(f"Expected repository ultralytics under {PROJECT_ROOT}, got {ultralytics.__file__}.")

from ultralytics import YOLO
from ultralytics.nn.modules import AMSCLCERDCRAUp, BGDRP3Fuse, LCERDCRAUp, UGDRDetect
from ultralytics.nn.modules.head import Detect


def assert_finite_tree(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), f"non-finite tensor: {tuple(value.shape)}"
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite_tree(item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite_tree(item)


def endpoint_checks():
    l3_cfg = dict(LCERDCRAUp._DEFAULT_CONFIG)
    endpoint_cfg = dict(l3_cfg)
    endpoint_cfg.update(
        evidence_mode="entropy",
        consensus_kernels=[3],
        consensus_weights=[1.0],
    )
    for seed in (0, 7, 19):
        torch.manual_seed(seed)
        parent = LCERDCRAUp(64, 32, l3_cfg).eval()
        child = AMSCLCERDCRAUp(64, 32, endpoint_cfg).eval()
        child.load_state_dict(parent.state_dict(), strict=True)
        deep = torch.randn(2, 64, 7, 9)
        lateral = torch.randn(2, 32, 14, 18)
        with torch.no_grad():
            assert torch.equal(parent([deep, lateral]), child([deep, lateral]))

    bgdr = BGDRP3Fuse(32, 48).eval()
    p2 = torch.randn(2, 32, 28, 36)
    p3 = torch.randn(2, 48, 14, 18)
    with torch.no_grad():
        assert torch.equal(bgdr([p2, p3]), p3)

    channels = (32, 64, 128)
    for seed in (1, 11, 29):
        torch.manual_seed(seed)
        base = Detect(4, channels).train()
        torch.manual_seed(seed)
        head = UGDRDetect(4, {"level_strengths": [1.0, 0.5, 0.0]}, channels).train()
        xs = [
            torch.randn(2, channels[0], 20, 20),
            torch.randn(2, channels[1], 10, 10),
            torch.randn(2, channels[2], 5, 5),
        ]
        base_out = base([x.clone() for x in xs])
        head_out = head([x.clone() for x in xs])
        assert all(torch.equal(a, b) for a, b in zip(base_out, head_out))


def yaml_and_model_checks(yaml_dir: Path, imgsz: int, device: str):
    paths = sorted(yaml_dir.glob("yolov13-l3cru-*.yaml"))
    assert len(paths) == 8, f"expected 8 YAMLs, got {len(paths)}"
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["nc"] == 4 and data["scale"] == "n"
        rows = data["backbone"] + data["head"]
        assert rows[16][0] == [-1, 12] and rows[16][2] == "Concat"
        model = YOLO(str(path), task="detect")
        module = model.model.to(device).eval()
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            output = module(x)
        assert_finite_tree(output)

        # Exercise fused inference. Custom plain Conv+BN branches may remain
        # unfused, but the model must remain executable and finite.
        fused = model.model.fuse().to(device).eval()
        with torch.no_grad():
            output_fused = fused(x)
        assert_finite_tree(output_fused)
        print(f"PASS {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    endpoint_checks()
    yaml_and_model_checks(args.yaml_dir, args.imgsz, args.device)
    print("REAL-REPO L3-CRU INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
