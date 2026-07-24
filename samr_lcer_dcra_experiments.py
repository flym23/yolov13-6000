"""Canonical S1--S3 SAMR-LCER-DCRA ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = ("s1_matched_endpoint", "s2_raw_endpoint", "s3_samr_main")

MODEL_FILES = {
    "s1_matched_endpoint": "yolov13-samr-lcer-dcra-s1-matched.yaml",
    "s2_raw_endpoint": "yolov13-samr-lcer-dcra-s2-raw.yaml",
    "s3_samr_main": "yolov13-samr-lcer-dcra-s3-adaptive.yaml",
}

STRUCTURES = {
    "s1_matched_endpoint": "S1 / SAMR matched 端点：moment_mode=matched，严格等价 LCER-DCRA L3",
    "s2_raw_endpoint": "S2 / SAMR raw 端点：moment_mode=raw，关闭残差矩保持",
    "s3_samr_main": "S3 / SAMR 主方案：支持域自适应矩松弛，relax_max=0.50，reference=0.10，power=1.0",
}

REFERENCE_BASELINES = (
    "/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
)


def resolve_model(root: Path, stage: str) -> Path:
    """Return the checked absolute YAML for one preregistered SAMR stage."""
    try:
        filename = MODEL_FILES[stage]
    except KeyError as error:
        raise ValueError(f"unknown SAMR-LCER-DCRA stage: {stage}") from error
    path = root / "ultralytics" / "cfg" / "models" / "v13" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
