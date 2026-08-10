"""Create a non-destructive zero-based-label view of URPC2019 for Ultralytics."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


SOURCE = Path("/home/room305/ZZF/URPC2019")
DESTINATION = Path("/home/room305/ZZF/yolov13-6000/datasets/URPC2019_zero_based")
SPLITS = ("train", "val")
NAMES = {0: "echinus", 1: "starfish", 2: "holothurian", 3: "scallop"}


def link_images() -> None:
    for split in SPLITS:
        source = SOURCE / "images" / split
        destination = DESTINATION / "images" / split
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"unsafe image-link target: {destination}")
            continue
        destination.symlink_to(source, target_is_directory=True)


def remap_labels() -> Counter[int]:
    counts: Counter[int] = Counter()
    for split in SPLITS:
        source_root = SOURCE / "labels" / split
        destination_root = DESTINATION / "labels" / split
        for source in source_root.rglob("*.txt"):
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"invalid label row: {source}:{number}")
                class_id = int(fields[0])
                if not 1 <= class_id <= 4:
                    raise ValueError(f"expected one-based class 1..4: {source}:{number}={class_id}")
                mapped = class_id - 1
                counts[mapped] += 1
                rows.append(" ".join((str(mapped), *fields[1:])))
            temporary = destination.with_suffix(".tmp")
            temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            temporary.replace(destination)
    return counts


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    link_images()
    counts = remap_labels()
    if set(counts) != set(NAMES):
        raise RuntimeError(f"expected all four classes after remapping, got {dict(counts)}")
    data = {
        "path": str(DESTINATION),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": NAMES,
    }
    (DESTINATION / "data.yaml").write_text(
        "\n".join((
            f"path: {data['path']}",
            "train: images/train",
            "val: images/val",
            "test: images/val",
            "names:",
            *[f"  {index}: {name}" for index, name in NAMES.items()],
            "",
        )),
        encoding="utf-8",
    )
    (DESTINATION / "remap_manifest.json").write_text(
        json.dumps({"source": str(SOURCE), "class_counts": counts, "mapping": {str(index + 1): index for index in NAMES}}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"destination": str(DESTINATION), "class_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
