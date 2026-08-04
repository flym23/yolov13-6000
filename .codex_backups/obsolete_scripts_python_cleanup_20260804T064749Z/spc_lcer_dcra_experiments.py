"""Canonical P0--P3 SPC-LCER-DCRA ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = ("p0_baseline", "p1_lcer_endpoint", "p2_naive_phase", "p3_spc_main")

MODEL_FILES = {
    "p0_baseline": "yolov13-spc-p0-baseline.yaml",
    "p1_lcer_endpoint": "yolov13-spc-p1-lcer-endpoint.yaml",
    "p2_naive_phase": "yolov13-spc-p2-naive.yaml",
    "p3_spc_main": "yolov13-spc-p3-main.yaml",
}

STRUCTURES = {
    "p0_baseline": "P0 / 原始 YOLOv13：P5→P4 最近邻上采样（同批次 3-seed 基线）",
    "p1_lcer_endpoint": "P1 / SPC-LCER 端点：alpha=0，逐元素等价于 LCER-DCRA L3",
    "p2_naive_phase": "P2 / SPC-LCER 朴素相位收缩：alpha=0.35，borrow_power=0",
    "p3_spc_main": "P3 / SPC-LCER 主方案：alpha=0.35，借用不确定且相位不一致的候选分布",
}


def resolve_model(root: Path, stage: str) -> Path:
    """Return a checked absolute YAML for one preregistered SPC stage."""
    try:
        filename = MODEL_FILES[stage]
    except KeyError as error:
        raise ValueError(f"unknown SPC-LCER-DCRA stage: {stage}") from error
    path = root / "ultralytics" / "cfg" / "models" / "v13" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
