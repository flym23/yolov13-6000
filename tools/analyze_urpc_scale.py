
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import yaml


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def resolve_dataset_root(data_yaml: Path, cfg: dict) -> Path:
    root = Path(cfg.get("path", ""))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def collect_images(root: Path, spec) -> list[Path]:
    if isinstance(spec, (list, tuple)):
        images: list[Path] = []
        for item in spec:
            images.extend(collect_images(root, item))
        return sorted(set(images))

    p = Path(str(spec))
    if not p.is_absolute():
        p = root / p

    if p.is_dir():
        return sorted(
            x for x in p.rglob("*")
            if x.suffix.lower() in IMAGE_SUFFIXES
        )

    if p.is_file() and p.suffix.lower() == ".txt":
        out: list[Path] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            q = Path(line)
            if not q.is_absolute():
                # Ultralytics list files are commonly relative to the txt file.
                q = (p.parent / q).resolve()
            out.append(q)
        return out

    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
        return [p]

    raise FileNotFoundError(f"Cannot resolve image specification: {p}")


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    # Replace the last "images" directory if present.
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    # Fallback: sibling labels directory.
    return (
        image_path.parent.parent
        / "labels"
        / image_path.name
    ).with_suffix(".txt")


def letterbox_scale(width: int, height: int, imgsz: int) -> float:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    return min(float(imgsz) / width, float(imgsz) / height)


def read_yolo_wh(label_path: Path) -> list[tuple[float, float]]:
    if not label_path.exists():
        return []
    out: list[tuple[float, float]] = []
    for lineno, line in enumerate(
        label_path.read_text(
            encoding="utf-8",
        ).splitlines(),
        start=1,
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f'Expected 5 YOLO detection fields: {label_path}:{lineno}')
        try:
            w = float(fields[3])
            h = float(fields[4])
        except ValueError as exc:
            raise ValueError(
                f"invalid YOLO label {label_path}:{lineno}"
            ) from exc
        if not all(math.isfinite(float(x)) for x in fields) or not (0 < w <= 1 and 0 < h <= 1):
            raise ValueError(
                f"negative width/height {label_path}:{lineno}"
            )
        out.append((w, h))
    return out


def audit(
    data_yaml: str | Path,
    split: str = "train",
    imgsz: int = 640,
) -> dict:
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(
        data_yaml.read_text(encoding="utf-8")
    )
    if split not in cfg:
        raise KeyError(
            f"split '{split}' is not in {data_yaml}"
        )

    root = resolve_dataset_root(
        data_yaml,
        cfg,
    )
    images = collect_images(
        root,
        cfg[split],
    )

    counts = {
        "objects": 0,
        "min_side_lt_8": 0,
        "min_side_lt_16": 0,
        "area_lt_16sq": 0,
        "area_lt_32sq": 0,
        "area_32sq_to_96sq": 0,
        "area_ge_96sq": 0,
    }
    missing_labels = 0

    side_values: list[float] = []
    area_values: list[float] = []

    for image_path in images:
        label_path = image_to_label_path(
            image_path
        )
        if not label_path.exists():
            missing_labels += 1
            continue

        with Image.open(image_path) as im:
            width, height = im.size

        scale = letterbox_scale(
            width,
            height,
            imgsz,
        )

        for norm_w, norm_h in read_yolo_wh(
            label_path
        ):
            w_px = norm_w * width * scale
            h_px = norm_h * height * scale
            min_side = min(w_px, h_px)
            area = w_px * h_px

            counts["objects"] += 1
            counts["min_side_lt_8"] += int(
                min_side < 8.0
            )
            counts["min_side_lt_16"] += int(
                min_side < 16.0
            )
            counts["area_lt_16sq"] += int(
                area < 16.0 ** 2
            )
            counts["area_lt_32sq"] += int(
                area < 32.0 ** 2
            )
            counts[
                "area_32sq_to_96sq"
            ] += int(
                32.0 ** 2
                <= area
                < 96.0 ** 2
            )
            counts["area_ge_96sq"] += int(
                area >= 96.0 ** 2
            )

            side_values.append(min_side)
            area_values.append(area)

    n = counts["objects"]

    def ratio(key: str) -> float:
        return (
            counts[key] / n
            if n
            else 0.0
        )

    def percentile(
        values: list[float],
        q: float,
    ) -> float | None:
        if not values:
            return None
        values = sorted(values)
        idx = (len(values) - 1) * q
        lo = int(idx)
        hi = min(lo + 1, len(values) - 1)
        frac = idx - lo
        return (
            values[lo] * (1.0 - frac)
            + values[hi] * frac
        )

    return {
        "data_yaml": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "image_count": len(images),
        "missing_label_files": missing_labels,
        **counts,
        "ratios": {
            "min_side_lt_8": ratio(
                "min_side_lt_8"
            ),
            "min_side_lt_16": ratio(
                "min_side_lt_16"
            ),
            "area_lt_16sq": ratio(
                "area_lt_16sq"
            ),
            "area_lt_32sq": ratio(
                "area_lt_32sq"
            ),
            "area_32sq_to_96sq": ratio(
                "area_32sq_to_96sq"
            ),
            "area_ge_96sq": ratio(
                "area_ge_96sq"
            ),
        },
        "min_side_px_percentiles": {
            "p10": percentile(
                side_values, 0.10
            ),
            "p25": percentile(
                side_values, 0.25
            ),
            "p50": percentile(
                side_values, 0.50
            ),
            "p75": percentile(
                side_values, 0.75
            ),
            "p90": percentile(
                side_values, 0.90
            ),
        },
        "area_px2_percentiles": {
            "p10": percentile(
                area_values, 0.10
            ),
            "p25": percentile(
                area_values, 0.25
            ),
            "p50": percentile(
                area_values, 0.50
            ),
            "p75": percentile(
                area_values, 0.75
            ),
            "p90": percentile(
                area_values, 0.90
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Ultralytics data YAML",
    )
    parser.add_argument(
        "--split",
        default="train",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--out",
        default="urpc_scale_audit.json",
    )
    args = parser.parse_args()

    result = audit(
        args.data,
        split=args.split,
        imgsz=args.imgsz,
    )
    text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
    print(text)
    Path(args.out).write_text(
        text,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
