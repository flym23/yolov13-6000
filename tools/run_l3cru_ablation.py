from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


GROUPS = (
    "t0_baseline",
    "t1_amsc",
    "t2_bgdr",
    "t3_ugdr",
    "t4_amsc_bgdr",
    "t5_amsc_ugdr",
    "t6_bgdr_ugdr",
    "t7_full",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_local_yolo():
    """Import the repository package and reject an accidental site-packages fallback."""
    import ultralytics

    package_path = Path(ultralytics.__file__).resolve()
    if PROJECT_ROOT not in package_path.parents:
        raise RuntimeError(
            f"Expected repository ultralytics under {PROJECT_ROOT}, got {package_path}."
        )
    return ultralytics.YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-dir", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS[:4]))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=Path("runs/train/l3cru"))
    parser.add_argument("--run-id", default="", help="Optional stable subdirectory name for a worker invocation.")
    parser.add_argument("--test-command", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.project / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    state = {
        "protocol": vars(args) | {"yaml_dir": str(args.yaml_dir), "project": str(args.project)},
        "runs": [],
    }
    state_path = run_root / "state.json"

    # Run sequentially on one device. Parallel model training changes memory
    # pressure and weakens the paired-seed interpretation.
    for group in args.groups:
        yaml_path = args.yaml_dir / f"yolov13-l3cru-{group}.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(yaml_path)
        for seed in args.seeds:
            name = f"{group}_seed{seed}"
            record = {"group": group, "seed": seed, "yaml": str(yaml_path), "name": name}
            state["runs"].append(record)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.dry_run:
                print("DRY-RUN", record)
                continue

            YOLO = load_local_yolo()
            model = YOLO(str(yaml_path), task="detect")
            result = model.train(
                data=args.data,
                epochs=args.epochs,
                patience=args.patience,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                amp=False,
                deterministic=True,
                seed=seed,
                resume=False,
                plots=False,
                project=str(run_root),
                name=name,
                exist_ok=False,
            )
            save_dir = Path(result.save_dir)
            best = save_dir / "weights" / "best.pt"
            if not best.is_file():
                raise FileNotFoundError(best)
            record["save_dir"] = str(save_dir)
            record["best"] = str(best)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            # APS/APM/APL are project-specific in this repository. Reuse the
            # established evaluator rather than inventing a new metric path.
            # The command may reference {weights}, {data}, {device}, {imgsz},
            # {batch}, {workers}, {group}, {seed}, and {run_root}.
            if args.test_command:
                command = args.test_command.format(
                    weights=best,
                    data=args.data,
                    device=args.device,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workers=args.workers,
                    group=group,
                    seed=seed,
                    run_root=run_root,
                )
                subprocess.run(command, shell=True, check=True, cwd=Path.cwd())

    print(state_path)


if __name__ == "__main__":
    main()
