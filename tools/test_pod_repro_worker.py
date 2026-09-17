"""Test one locked URPC2019 reproduction checkpoint with scale-aware AP summaries."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import ultralytics

    package_path = Path(ultralytics.__file__).resolve()
    if ROOT not in package_path.parents:
        raise RuntimeError(f"Imported ultralytics outside project root: {package_path}")
    data_yaml = Path(os.environ.get("URPC2019_ROOT", "/home/room305/ZZF/URPC2019")) / "data.yaml"
    for path in (args.weights, data_yaml, ROOT / "test.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.project.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    command = [
        str(Path(sys.executable).resolve()), str(ROOT / "test.py"),
        "--weights", str(args.weights), "--data", str(data_yaml), "--project", str(args.project),
        "--name", args.name, "--device", "0", "--batch", "16", "--workers", "2", "--imgsz", "640", "--no-plots",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    summary_path = args.project / args.name / "summary_metrics.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)


if __name__ == "__main__":
    main()
