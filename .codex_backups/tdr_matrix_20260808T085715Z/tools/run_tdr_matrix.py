#!/usr/bin/env python3
"""Run the complete TDR factorial matrix with three controlled seed workers per variant."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIANTS = ("d0", "t", "s", "e", "ts", "te", "se", "tdr")
SEEDS = (0, 1, 2)
METRICS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_METRICS = ("APS", "APM", "APL")


class ChainFailure(RuntimeError):
    """Raised when a required matrix stage cannot be completed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--chain-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--upstream-status", choices=("completed", "failed", "cancelled"), required=True)
    parser.add_argument("--upstream-reason", default="")
    parser.add_argument("--resume-d0-root", type=Path, required=True)
    return parser.parse_args()


class TDRMatrixRunner:
    """Persisted fail-fast orchestration for the eight TDR variants."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.project_root.resolve()
        self.chain_root = args.chain_root.resolve()
        self.resume_d0_root = args.resume_d0_root.resolve()
        self.data_yaml = Path(os.environ.get("URPC2019_ROOT", "/home/room305/ZZF/URPC2019")) / "data.yaml"
        self.python = Path(sys.executable).resolve()
        self.state_path = self.chain_root / "state.json"
        self.active: dict[str, subprocess.Popen[bytes]] = {}
        self.completed: list[str] = []
        self.cancelled = False

    def state(self, status: str, phase: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "run_id": self.args.run_id,
            "status": status,
            "phase": phase,
            "updated_at": utc_now(),
            "project_root": str(self.root),
            "chain_root": str(self.chain_root),
            "upstream_state": str(self.args.upstream_state),
            "upstream_status": self.args.upstream_status,
            "upstream_failure_reason": self.args.upstream_reason,
            "resume_d0_root": str(self.resume_d0_root),
            "completed_variants": list(self.completed),
            "worker_pids": {name: process.pid for name, process in self.active.items() if process.poll() is None},
        }
        payload.update(extra)
        atomic_json(self.state_path, payload)

    def validate_paths(self) -> None:
        expected_chain_parent = (self.root / "runs" / "chain").resolve()
        required = (
            self.root / "tools" / "train_tdr_worker.py",
            self.root / "test.py",
            self.root / "yolov13n.pt",
            self.data_yaml,
            self.resume_d0_root,
        )
        if not self.root.is_absolute() or not self.chain_root.is_absolute() or self.chain_root.parent != expected_chain_parent:
            raise ChainFailure("project_root and chain_root must be absolute, with chain_root under runs/chain/.")
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
        for variant in VARIANTS:
            config = self.model_config(variant)
            if not config.is_file():
                raise FileNotFoundError(config)

    def model_config(self, variant: str) -> Path:
        return self.root / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13n-{variant}.yaml"

    def snapshot_protocol(self) -> None:
        self.chain_root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.chain_root / "training_parameters.json",
            {
                "run_id": self.args.run_id,
                "variants": list(VARIANTS),
                "seeds": list(SEEDS),
                "data": str(self.data_yaml),
                "initialization": "YOLO(...).load(yolov13n.pt)",
                "epochs": 300,
                "patience": 40,
                "device": 0,
                "workers": 2,
                "amp": False,
                "deterministic": True,
                "plots": False,
                "imgsz": 640,
                "batch": 16,
                "created_at": utc_now(),
            },
        )
        snapshot_dir = self.chain_root / "model_yaml"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            shutil.copy2(self.model_config(variant), snapshot_dir / self.model_config(variant).name)

    def preflight(self) -> None:
        self.state("running", "preflight")
        self.validate_paths()
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        import ultralytics  # noqa: PLC0415
        from ultralytics import YOLO  # noqa: PLC0415

        package_path = Path(ultralytics.__file__).resolve()
        if self.root not in package_path.parents:
            raise ChainFailure(f"Imported ultralytics outside project root: {package_path}")
        for variant in VARIANTS:
            YOLO(str(self.model_config(variant))).model
        test_command = [str(self.python), "-m", "pytest", "-q", "tests/test_tdr_yolov13.py", "tests/test_urpc2019_one_based_labels.py"]
        test_log = self.chain_root / "preflight_tests.log"
        with test_log.open("wb") as stream:
            completed = subprocess.run(test_command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise ChainFailure(f"preflight tests failed; see {test_log}")
        self.snapshot_protocol()

    def resume_weight(self, seed: int) -> Path:
        matches = sorted(self.resume_d0_root.glob(f"*_d0_seed{seed}/weights/best.pt"))
        if len(matches) != 1:
            raise ChainFailure(f"Expected exactly one completed D0 best.pt for seed {seed}, found {matches}")
        return matches[0]

    def terminate_active(self) -> None:
        for process in self.active.values():
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
        deadline = time.monotonic() + 30
        for process in self.active.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
        self.active.clear()

    def train_variant(self, variant: str) -> dict[int, Path]:
        phase = f"{variant}.train"
        self.state("running", phase)
        train_root = self.chain_root / "training" / variant
        log_root = self.chain_root / "logs" / variant
        train_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update({"WANDB_DISABLED": "true", "PIN_MEMORY": "false", "CUDA_VISIBLE_DEVICES": "0"})
        for seed in SEEDS:
            name = f"seed{seed}"
            command = [
                str(self.python),
                str(self.root / "tools" / "train_tdr_worker.py"),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--project",
                str(train_root),
                "--name",
                name,
            ]
            stream = (log_root / f"train_seed{seed}.log").open("wb")
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            stream.close()
            self.active[f"{variant}_seed{seed}"] = process
        self.state("running", phase)
        pending = dict(self.active)
        while pending:
            for name, process in tuple(pending.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.pop(name)
                if return_code != 0:
                    self.terminate_active()
                    raise ChainFailure(f"training failure for {variant}: {name} exited with {return_code}")
            if pending:
                time.sleep(2)
        self.active.clear()
        weights = {seed: train_root / f"seed{seed}" / "weights" / "best.pt" for seed in SEEDS}
        missing = [str(path) for path in weights.values() if not path.is_file()]
        if missing:
            raise ChainFailure(f"training completed without required checkpoints: {missing}")
        return weights

    def evaluate_variant(self, variant: str, weights: dict[int, Path]) -> list[dict[str, Any]]:
        self.state("running", f"{variant}.test")
        validation_root = self.chain_root / "validation" / variant
        log_root = self.chain_root / "logs" / variant
        validation_root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        environment = os.environ.copy()
        environment.update({"WANDB_DISABLED": "true", "PIN_MEMORY": "false", "CUDA_VISIBLE_DEVICES": "0"})
        for seed in SEEDS:
            command = [
                str(self.python),
                str(self.root / "test.py"),
                "--weights",
                str(weights[seed]),
                "--data",
                str(self.data_yaml),
                "--name",
                f"seed{seed}",
                "--project",
                str(validation_root),
                "--device",
                "0",
                "--batch",
                "16",
                "--workers",
                "2",
                "--imgsz",
                "640",
                "--no-plots",
            ]
            log_path = log_root / f"test_seed{seed}.log"
            with log_path.open("wb") as stream:
                completed = subprocess.run(command, cwd=self.root, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                raise ChainFailure(f"validation failure for {variant} seed {seed}; see {log_path}")
            summary_path = validation_root / f"seed{seed}" / "summary_metrics.json"
            scale_path = validation_root / f"seed{seed}" / "scale_ap_metrics.json"
            if not summary_path.is_file() or not scale_path.is_file():
                raise ChainFailure(f"validation output incomplete for {variant} seed {seed}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary.get("metrics", {})
            scale = summary.get("scale_metrics_percent", {})
            record: dict[str, Any] = {"variant": variant, "seed": seed, "weights": str(weights[seed])}
            for display, key in METRICS.items():
                if key not in metrics:
                    raise ChainFailure(f"{summary_path} missing {key}")
                record[display] = float(metrics[key]) * 100.0
            for name in SCALE_METRICS:
                if name not in scale:
                    raise ChainFailure(f"{summary_path} missing {name}")
                record[name] = float(scale[name])
            records.append(record)
        return records

    def write_summary(self, records: list[dict[str, Any]]) -> Path:
        output = self.chain_root / "summary"
        output.mkdir(parents=True, exist_ok=True)
        fields = ["variant", "seed", "P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL", "weights"]
        with (output / "seed_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        variants: dict[str, Any] = {}
        for variant in self.completed:
            stage_records = [record for record in records if record["variant"] == variant]
            variants[variant] = {
                "n": len(stage_records),
                "runs": stage_records,
                "metrics_percent": {name: mean_std([float(record[name]) for record in stage_records]) for name in fields[2:11]},
            }
        summary_path = output / "summary_metrics.json"
        atomic_json(
            summary_path,
            {
                "run_id": self.args.run_id,
                "dataset": str(self.data_yaml),
                "completed_variants": list(self.completed),
                "variants": variants,
                "seed_metrics_csv": str(output / "seed_metrics.csv"),
                "updated_at": utc_now(),
            },
        )
        return summary_path

    def run(self) -> None:
        self.preflight()
        records: list[dict[str, Any]] = []
        for variant in VARIANTS:
            if variant == "d0":
                weights = {seed: self.resume_weight(seed) for seed in SEEDS}
                atomic_json(
                    self.chain_root / "training" / "d0" / "resume_manifest.json",
                    {"source_run": str(self.resume_d0_root), "weights": {str(seed): str(path) for seed, path in weights.items()}},
                )
            else:
                weights = self.train_variant(variant)
            records.extend(self.evaluate_variant(variant, weights))
            self.completed.append(variant)
            summary = self.write_summary(records)
            self.state("running", f"{variant}.complete", summary_metrics=str(summary))
        summary = self.write_summary(records)
        self.state("completed", "complete", summary_metrics=str(summary), completed_at=utc_now())


def main() -> None:
    args = parse_args()
    runner = TDRMatrixRunner(args)

    def cancel(signum: int, _frame: Any) -> None:
        runner.cancelled = True
        runner.terminate_active()
        runner.state("cancelled", "cancelled", failure_reason=f"received signal {signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    try:
        runner.run()
    except BaseException as error:
        if not runner.cancelled:
            runner.terminate_active()
            runner.state("failed", "failed", failure_reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
