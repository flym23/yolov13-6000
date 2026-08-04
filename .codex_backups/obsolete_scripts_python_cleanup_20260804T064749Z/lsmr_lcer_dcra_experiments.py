"""Canonical R1--R3 LSMR-LCER-DCRA ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = ("r1_matched_endpoint", "r2_global_endpoint", "r3_lsmr_local")

MODEL_FILES = {
    "r1_matched_endpoint": "yolov13-lsmr-lcer-dcra-r1-matched.yaml",
    "r2_global_endpoint": "yolov13-lsmr-lcer-dcra-r2-global.yaml",
    "r3_lsmr_local": "yolov13-lsmr-lcer-dcra-r3-local.yaml",
}

STRUCTURES = {
    "r1_matched_endpoint": "R1 / LSMR matched 端点：moment_mode=matched，严格等价 LCER-DCRA L3",
    "r2_global_endpoint": "R2 / LSMR global 端点：moment_mode=global，严格等价 SAMR-S3 的通道全局松弛",
    "r3_lsmr_local": "R3 / LSMR 主方案：moment_mode=local，矩差异的 5×5 局部支持与置信密度松弛",
}

REFERENCE_BASELINES = (
    "/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
)


def resolve_model(root: Path, stage: str) -> Path:
    """Return a checked absolute YAML for one preregistered LSMR stage."""
    try:
        filename = MODEL_FILES[stage]
    except KeyError as error:
        raise ValueError(f"unknown LSMR-LCER-DCRA stage: {stage}") from error
    path = root / "ultralytics" / "cfg" / "models" / "v13" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
