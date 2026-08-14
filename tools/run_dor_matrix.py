#!/usr/bin/env python3
"""Fail-fast DOR 2^3 matrix: DCPR, OCARFuse, and RQDDetect, three seeds per active cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


VARIANTS = ("d0", "dg", "oa", "r", "dgoa", "dgr", "oar", "dor")
SEEDS = (0, 1, 2)
STRUCTURES = {
    "d0": {"DCPR": False, "OCARFuse": False, "RQDDetect": False},
    "dg": {"DCPR": True, "OCARFuse": False, "RQDDetect": False},
    "oa": {"DCPR": False, "OCARFuse": True, "RQDDetect": False},
    "r": {"DCPR": False, "OCARFuse": False, "RQDDetect": True},
    "dgoa": {"DCPR": True, "OCARFuse": True, "RQDDetect": False},
    "dgr": {"DCPR": True, "OCARFuse": False, "RQDDetect": True},
    "oar": {"DCPR": False, "OCARFuse": True, "RQDDetect": True},
    "dor": {"DCPR": True, "OCARFuse": True, "RQDDetect": True},
}
METRICS = {"P": "metrics/precision(B)", "R": "metrics/recall(B)", "mAP50": "metrics/mAP50(B)", "mAP75": "metrics/mAP75(B)", "mAP50-95": "metrics/mAP50-95(B)"}
SCALE_METRICS = ("APS", "APM", "APL")
DCPR_CONFIG = {"reduction": 4, "prompt_kernel": 5, "num_bases": 4, "router_temperature": 1.0, "max_scale": 0.05, "detach_prompt": True, "finite_fallback": True, "eps": 1.0e-6}
OCAR_CONFIG = {"reduction": 4, "support_kernel": 5, "max_residual_ratio": 0.06, "detach_gate": True, "detach_bound": True, "finite_fallback": True, "eps": 1.0e-6}
RQD_CONFIG = {"quality_mix": 0.50, "stat_strengths": [1.0, 0.5, 0.25], "detach_stats": True, "rescue_mix": 0.30, "rescue_level_strengths": [1.0, 0.6, 0.3], "objectness_prior_logit": -9.0, "eps": 1.0e-6}


class ChainFailure(RuntimeError):
    """A required DOR preflight, worker, test, or summary artifact failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def variant_yaml(flags: dict[str, bool]) -> dict[str, Any]:
    """Build D0 topology exactly, adding only requested DCPR/OCAR/RQD factors."""
    dg, oa, rqd = flags["DCPR"], flags["OCARFuse"], flags["RQDDetect"]
    backbone: list[list[Any]] = [
        [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2, 1, 2]],
        [-1, 2, "DSC3k2", [256, False, 0.25]], [-1, 1, "Conv", [256, 3, 2, 1, 4]],
        [-1, 2, "DSC3k2", [512, False, 0.25]], [-1, 1, "DSConv", [512, 3, 2]],
        [-1, 4, "A2C2f", [512, True, 4]], [-1, 1, "DSConv", [1024, 3, 2]],
        [-1, 4, "A2C2f", [1024, True, 1]],
    ]
    # Build the head by remembered layer indices rather than hand-maintained offsets.
    # That keeps every one-factor and interaction cell topologically identical to D0
    # except for the requested factor insertions.
    p5 = 8
    if dg:
        backbone.append([[2, 8], 1, "DCPR", [DCPR_CONFIG]])
        p5 = 9
    head: list[list[Any]] = []

    def add(source: Any, repeats: int, module: str, args: list[Any]) -> int:
        head.append([source, repeats, module, args])
        return len(backbone) + len(head) - 1

    hyper = add([4, 6, p5], 2, "HyperACE", [512, 8, True, True, 0.5, 1, "both"])
    hyper_up = add(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    hyper_down = add(hyper, 1, "DownsampleConv", [])
    p4_lateral = add([6, hyper], 1, "FullPAD_Tunnel", [])
    p3_lateral = add([4, hyper_up], 1, "FullPAD_Tunnel", [])
    p5_lateral = add([p5, hyper_down], 1, "FullPAD_Tunnel", [])
    deep_up = add(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    deep_cat = add([deep_up, p4_lateral], 1, "Concat", [1])
    p4_base = add(-1, 2, "DSC3k2", [512, True])
    if oa:
        p4_fused = add([deep_up, p4_lateral, p4_base], 1, "OCARFuse", [OCAR_CONFIG])
    else:
        p4_fused = p4_base
    p4_tunnel = add([p4_fused, hyper], 1, "FullPAD_Tunnel", [])

    p3_up = add(p4_fused, 1, "nn.Upsample", [None, 2, "nearest"])
    p3_cat = add([p3_up, p3_lateral], 1, "Concat", [1])
    p3_feature = add(-1, 2, "DSC3k2", [256, True])
    hyper_p3 = add(hyper_up, 1, "Conv", [256, 1, 1])
    p3_detect = add([p3_feature, hyper_p3], 1, "FullPAD_Tunnel", [])
    p4_down = add(-1, 1, "Conv", [256, 3, 2])
    p4_cat = add([p4_down, p4_tunnel], 1, "Concat", [1])
    p4_feature = add(-1, 2, "DSC3k2", [512, True])
    p4_detect = add([p4_feature, hyper], 1, "FullPAD_Tunnel", [])
    p5_down = add(-1, 1, "Conv", [512, 3, 2])
    p5_cat = add([p5_down, p5_lateral], 1, "Concat", [1])
    p5_feature = add(-1, 2, "DSC3k2", [1024, True])
    p5_detect = add([p5_feature, hyper_down], 1, "FullPAD_Tunnel", [])
    head.append([[p3_detect, p4_detect, p5_detect], 1, "RQDDetect" if rqd else "Detect", ["nc", RQD_CONFIG] if rqd else ["nc"]])
    payload: dict[str, Any] = {"nc": 4, "scale": "n", "scales": {"n": [0.50, 0.25, 1024], "s": [0.50, 0.50, 1024], "l": [1.0, 1.0, 512], "x": [1.0, 1.5, 512]}, "backbone": backbone, "head": head}
    if rqd:
        payload.update({"quality_gain": 0.25, "objectness_gain": 0.10, "objectness_gamma": 2.0, "objectness_neg_weight": 0.25})
    else:
        payload.update({"quality_gain": 0.0, "objectness_gain": 0.0})
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scheme-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--upstream-status", choices=("completed", "failed", "cancelled"), required=True)
    parser.add_argument("--upstream-reason", default="")
    parser.add_argument("--resume-d0-root", type=Path, required=True)
    parser.add_argument("--pod-controls-root", type=Path, required=True)
    return parser.parse_args()


class DORMatrixRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args, self.root, self.scheme_root = args, args.project_root.resolve(), args.scheme_root.resolve()
        self.train_root, self.test_root = self.scheme_root / "train", self.scheme_root / "test"
        self.data_yaml = Path(os.environ.get("URPC2019_ROOT", "/home/room305/ZZF/URPC2019")) / "data.yaml"
        self.python, self.active, self.completed, self.cancelled = Path(sys.executable).resolve(), {}, [], False

    @property
    def state_path(self) -> Path:
        return self.scheme_root / "state.json"

    def state(self, status: str, phase: str, **extra: Any) -> None:
        payload = {"run_id": self.args.run_id, "status": status, "phase": phase, "updated_at": utc_now(), "project_root": str(self.root), "scheme_root": str(self.scheme_root), "training_root": str(self.train_root), "test_root": str(self.test_root), "upstream_state": str(self.args.upstream_state), "upstream_status": self.args.upstream_status, "upstream_failure_reason": self.args.upstream_reason, "resume_d0_root": str(self.args.resume_d0_root.resolve()), "pod_controls_root": str(self.args.pod_controls_root.resolve()), "completed_variants": list(self.completed), "worker_pids": {name: process.pid for name, process in self.active.items() if process.poll() is None}}
        payload.update(extra)
        atomic_json(self.state_path, payload)

    def config_path(self, variant: str) -> Path:
        return self.train_root / "model_yaml" / f"yolov13n-dor-{variant}.yaml"

    def prepare_protocol(self) -> None:
        self.train_root.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for variant in VARIANTS:
            path = self.config_path(variant)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(variant_yaml(STRUCTURES[variant]), sort_keys=False, allow_unicode=True), encoding="utf-8")
            manifest[variant] = {"snapshot": str(path), "sha256": sha256(path), "structure": STRUCTURES[variant]}
        atomic_json(self.train_root / "training_parameters.json", {"run_id": self.args.run_id, "variants": list(VARIANTS), "variant_structures": STRUCTURES, "seeds": list(SEEDS), "data": str(self.data_yaml), "initialization": "YOLO(...).load(yolov13n.pt)", "epochs": 300, "patience": 40, "device": 0, "single_gpu": True, "workers": 2, "amp": False, "deterministic": True, "plots": False, "imgsz": 640, "batch": 16, "continuity_controls": ["POD-D", "POD-OD"], "created_at": utc_now()})
        atomic_json(self.train_root / "model_yaml_manifest.json", manifest)

    def validate_paths(self) -> None:
        required = (self.root / "tools" / "train_dor_worker.py", self.root / "tools" / "dor_preflight.py", self.root / "test.py", self.root / "yolov13n.pt", self.data_yaml, self.args.resume_d0_root, self.args.pod_controls_root)
        if not self.root.is_absolute() or not self.scheme_root.is_absolute() or self.scheme_root.parent != (self.root / "runs").resolve():
            raise ChainFailure("project_root and scheme_root must be absolute; scheme_root must be immediately under runs/.")
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
        for variant in VARIANTS:
            if not self.config_path(variant).is_file():
                raise FileNotFoundError(self.config_path(variant))
        for control in ("d", "od"):
            for seed in SEEDS:
                path = self.args.pod_controls_root / control / f"seed{seed}" / "weights" / "best.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)

    def preflight(self) -> None:
        self.prepare_protocol(); self.validate_paths(); self.state("running", "preflight")
        log_path = self.train_root / "preflight.log"
        command = [str(self.python), str(self.root / "tools" / "dor_preflight.py"), "--project-root", str(self.root), "--config-dir", str(self.config_path("d0").parent), "--data", str(self.data_yaml), "--output-dir", str(self.test_root / "preflight")]
        with log_path.open("wb") as stream:
            result = subprocess.run(command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise ChainFailure(f"DOR preflight failed; see {log_path}")

    def weights_from(self, root: Path, variant: str) -> dict[int, Path]:
        weights = {seed: root / variant / f"seed{seed}" / "weights" / "best.pt" for seed in SEEDS}
        if missing := [str(path) for path in weights.values() if not path.is_file()]:
            raise ChainFailure(f"missing locked weights: {missing}")
        return weights

    def terminate_active(self) -> None:
        for process in self.active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM) if os.name == "posix" else process.terminate()
        for process in self.active.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        self.active.clear()

    def train_variant(self, variant: str) -> dict[int, Path]:
        train_dir, log_dir, pid_dir = self.train_root / variant, self.train_root / variant / "logs", self.scheme_root / "pids"
        log_dir.mkdir(parents=True, exist_ok=True); pid_dir.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "WANDB_DISABLED": "true", "PIN_MEMORY": "false", "CUDA_VISIBLE_DEVICES": "0"}
        for seed in SEEDS:
            stream = (log_dir / f"train_seed{seed}.log").open("wb")
            command = [str(self.python), str(self.root / "tools" / "train_dor_worker.py"), "--model-yaml", str(self.config_path(variant)), "--seed", str(seed), "--project", str(train_dir), "--name", f"seed{seed}"]
            process = subprocess.Popen(command, cwd=self.root, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            stream.close(); name = f"{variant}_seed{seed}"; self.active[name] = process
            (pid_dir / f"{name}.pid").write_text(f"{process.pid}\n", encoding="utf-8")
        self.state("running", f"{variant}.train")
        while self.active:
            for name, process in tuple(self.active.items()):
                code = process.poll()
                if code is not None:
                    self.active.pop(name)
                    if code:
                        self.terminate_active(); raise ChainFailure(f"training failure: {name} exited {code}")
            if self.active:
                time.sleep(2)
        weights = {seed: train_dir / f"seed{seed}" / "weights" / "best.pt" for seed in SEEDS}
        if missing := [str(path) for path in weights.values() if not path.is_file()]:
            raise ChainFailure(f"training did not create checkpoints: {missing}")
        return weights

    def evaluate(self, variant: str, weights: dict[int, Path], structure: dict[str, bool], source: str = "trained") -> list[dict[str, Any]]:
        self.state("running", f"{variant}.test")
        validation_dir, log_dir = self.test_root / variant, self.test_root / variant / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "WANDB_DISABLED": "true", "PIN_MEMORY": "false", "CUDA_VISIBLE_DEVICES": "0"}
        records = []
        for seed in SEEDS:
            log_path = log_dir / f"test_seed{seed}.log"
            command = [str(self.python), str(self.root / "test.py"), "--weights", str(weights[seed]), "--data", str(self.data_yaml), "--name", f"seed{seed}", "--project", str(validation_dir), "--device", "0", "--batch", "16", "--workers", "2", "--imgsz", "640", "--no-plots"]
            with log_path.open("wb") as stream:
                result = subprocess.run(command, cwd=self.root, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise ChainFailure(f"test failure for {variant} seed{seed}; see {log_path}")
            output = validation_dir / f"seed{seed}"
            summary_path, scale_path, class_scale_path = output / "summary_metrics.json", output / "scale_ap_metrics.json", output / "class_scale_ap.json"
            if not all(path.is_file() for path in (summary_path, scale_path, class_scale_path)):
                raise ChainFailure(f"incomplete test summary for {variant} seed{seed}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            class_scale = json.loads(class_scale_path.read_text(encoding="utf-8")).get("classes", {})
            if not class_scale or any(not all(scale in metrics and metrics[scale] is not None for scale in SCALE_METRICS) for metrics in class_scale.values()):
                raise ChainFailure(f"class-scale AP is incomplete for {variant} seed{seed}")
            metrics, scale = summary["metrics"], summary["scale_metrics_percent"]
            record = {"variant": variant, "seed": seed, "source": source, "weights": str(weights[seed]), "summary_metrics": str(summary_path), "class_scale_ap": str(class_scale_path), "structure": structure}
            record.update({name: float(metrics[key]) * 100.0 for name, key in METRICS.items()}); record.update({name: float(scale[name]) for name in SCALE_METRICS})
            records.append(record)
        return records

    def write_summary(self, records: list[dict[str, Any]], controls: list[dict[str, Any]]) -> Path:
        output = self.test_root / "summary"; output.mkdir(parents=True, exist_ok=True)
        fields = ["variant", "seed", "source", *METRICS, *SCALE_METRICS, "weights", "summary_metrics", "class_scale_ap"]
        with (output / "seed_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(records)
        variants = {variant: {"n": len(rows), "structure": STRUCTURES[variant], "runs": rows, "metrics_percent": {metric: mean_std([float(row[metric]) for row in rows]) for metric in (*METRICS, *SCALE_METRICS)}} for variant in self.completed if (rows := [row for row in records if row["variant"] == variant])}
        path = output / "summary_metrics.json"
        atomic_json(path, {"scheme": "DOR-YOLOv13", "run_id": self.args.run_id, "dataset": str(self.data_yaml), "completed_variants": list(self.completed), "variant_structures": STRUCTURES, "variants": variants, "continuity_controls": controls, "seed_metrics_csv": str(output / "seed_metrics.csv"), "updated_at": utc_now()})
        return path

    def run(self) -> None:
        self.preflight(); records: list[dict[str, Any]] = []
        controls = []
        for name, folder in (("pod_d", "d"), ("pod_od", "od")):
            values = self.evaluate(name, self.weights_from(self.args.pod_controls_root, folder), {"DCPR": False, "OCARFuse": False, "RQDDetect": True}, source="locked_pod_control")
            controls.extend(values)
        for variant in VARIANTS:
            if variant == "d0":
                weights = self.weights_from(self.args.resume_d0_root, "d0")
                atomic_json(self.train_root / "d0" / "resume_manifest.json", {"source": str(self.args.resume_d0_root), "weights": {str(seed): str(path) for seed, path in weights.items()}})
            else:
                weights = self.train_variant(variant)
            records.extend(self.evaluate(variant, weights, STRUCTURES[variant])); self.completed.append(variant)
            self.state("running", f"{variant}.complete", summary_metrics=str(self.write_summary(records, controls)))
        self.state("completed", "complete", summary_metrics=str(self.write_summary(records, controls)), completed_at=utc_now())


def main() -> None:
    runner = DORMatrixRunner(parse_args())
    def cancel(signum: int, _frame: Any) -> None:
        runner.cancelled = True; runner.terminate_active(); runner.state("cancelled", "cancelled", failure_reason=f"received signal {signum}"); raise SystemExit(128 + signum)
    signal.signal(signal.SIGINT, cancel); signal.signal(signal.SIGTERM, cancel)
    try:
        runner.run()
    except BaseException as error:
        if not runner.cancelled:
            runner.terminate_active(); runner.state("failed", "failed", failure_reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
