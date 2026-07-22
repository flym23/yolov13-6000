"""Canonical L0--L3 LCER-DCRA ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = (
    "l0_baseline",
    "l1_strict_rho020",
    "l2_channel_power20",
    "l3_local_consensus",
)

MODEL_FILES = {
    "l0_baseline": "yolov13-lcer-dcra-l0.yaml",
    "l1_strict_rho020": "yolov13-lcer-dcra-l1-strict.yaml",
    "l2_channel_power20": "yolov13-lcer-dcra-l2-channel.yaml",
    "l3_local_consensus": "yolov13-lcer-dcra-l3-local.yaml",
}

STRUCTURES = {
    "l0_baseline": "L0 / 原始 YOLOv13：P5→P4 为最近邻上采样（重跑 3 个 seed）",
    "l1_strict_rho020": "L1 / LCER-DCRA 严格端点：ME-DCRA 能量约束，rho=0.20",
    "l2_channel_power20": "L2 / LCER-DCRA 通道释放：能量加权置信度^2，rho=0.20，无空间共识",
    "l3_local_consensus": "L3 / LCER-DCRA 局部共识：能量加权置信度^2 + 3×3 空间共识，rho=0.20",
}


def resolve_model(root: Path, stage: str) -> Path:
    """Return a checked absolute model YAML for one preregistered LCER-DCRA stage."""
    try:
        filename = MODEL_FILES[stage]
    except KeyError as error:
        raise ValueError(f"unknown LCER-DCRA stage: {stage}") from error
    path = root / "ultralytics" / "cfg" / "models" / "v13" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
