"""Canonical D1--D4 DGMR-LCER-DCRA ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = (
    "d1_matched_endpoint",
    "d2_global_endpoint",
    "d3_local_endpoint",
    "d4_dgmr_dual",
)

MODEL_FILES = {
    "d1_matched_endpoint": "yolov13-dgmr-lcer-dcra-d1-matched.yaml",
    "d2_global_endpoint": "yolov13-dgmr-lcer-dcra-d2-global.yaml",
    "d3_local_endpoint": "yolov13-dgmr-lcer-dcra-d3-local.yaml",
    "d4_dgmr_dual": "yolov13-dgmr-lcer-dcra-d4-dual.yaml",
}

STRUCTURES = {
    "d1_matched_endpoint": "D1 / DGMR 类 matched 端点：严格等价 LCER-DCRA L3 的矩匹配残差。",
    "d2_global_endpoint": "D2 / DGMR 类 global 端点：严格等价已完成的 SAMR/LSMR 通道全局矩松弛。",
    "d3_local_endpoint": "D3 / DGMR 类 local 端点：严格等价已完成的 LSMR 5×5 局部支持矩松弛。",
    "d4_dgmr_dual": "D4 / DGMR 主方案：global 门与 2×2 单元一致 local 下包络取交集的双门控矩松弛。",
}

REFERENCE_BASELINES = (
    "/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
)


def resolve_model(root: Path, stage: str) -> Path:
    """Return a checked absolute YAML for one preregistered DGMR stage."""
    try:
        filename = MODEL_FILES[stage]
    except KeyError as error:
        raise ValueError(f"unknown DGMR-LCER-DCRA stage: {stage}") from error
    path = root / "ultralytics" / "cfg" / "models" / "v13" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
