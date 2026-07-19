"""Canonical U2--U8 URR-DCRA ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = (
    "u2_m7_rho020",
    "u3_adaptive",
    "u4_mean",
    "u5_power05",
    "u6_power20",
    "u7_strict",
    "u8_none",
)

MODEL_FILES = {
    "u2_m7_rho020": "yolov13-medcra-rho020.yaml",
    "u3_adaptive": "yolov13-urr-dcra.yaml",
    "u4_mean": "yolov13-urr-dcra-mean.yaml",
    "u5_power05": "yolov13-urr-dcra-power05.yaml",
    "u6_power20": "yolov13-urr-dcra-power20.yaml",
    "u7_strict": "yolov13-urr-dcra-strict.yaml",
    "u8_none": "yolov13-urr-dcra-none.yaml",
}

STRUCTURES = {
    "u2_m7_rho020": "U2 / ME-DCRA M7 fixed rho=0.20",
    "u3_adaptive": "U3 / URR-DCRA adaptive energy-weighted release, power=1.0",
    "u4_mean": "U4 / URR-DCRA spatial-mean confidence release",
    "u5_power05": "U5 / URR-DCRA adaptive energy-weighted release, power=0.5",
    "u6_power20": "U6 / URR-DCRA adaptive energy-weighted release, power=2.0",
    "u7_strict": "U7 / URR-DCRA strict endpoint (ME-DCRA M2 equivalent)",
    "u8_none": "U8 / URR-DCRA released endpoint (ME-DCRA M5 equivalent)",
}


def resolve_model(root: Path, stage: str) -> Path:
    """Return a checked absolute model YAML for one reviewed ablation stage."""
    try:
        filename = MODEL_FILES[stage]
    except KeyError as error:
        raise ValueError(f"unknown URR-DCRA stage: {stage}") from error
    path = root / "ultralytics" / "cfg" / "models" / "v13" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
