"""Run the locked URPC2019 L0/POD-OD reproduction and O/D/OD rerun matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
TRAIN_WORKER = ROOT / "tools" / "train_pod_repro_worker.py"
TEST_WORKER = ROOT / "tools" / "test_pod_repro_worker.py"
DATA_YAML = Path(os.environ.get("URPC2019_ROOT", "/home/room305/ZZF/URPC2019")) / "data.yaml"
CONFIGS = {
    "l0": ROOT / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-lcer-dcra-l0.yaml",
    "o": ROOT / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n-pod-o.yaml",
    "d": ROOT / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n-pod-d.yaml",
    "od": ROOT / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n-pod-od.yaml",
}
WAVES = (
    ("stage1_l0", "l0", "paired baseline"),
    ("stage1_od", "od", "paired POD-OD"),
    ("stage2_o", "o", "independent O rerun"),
    ("stage2_d", "d", "independent D rerun"),
    ("stage2_od", "od", "independent OD rerun"),
)
METRICS = (
    ("P", "metrics/precision(B)", 100.0),
    ("R", "metrics/recall(B)", 100.0),
    ("mAP50", "metrics/mAP50(B)", 100.0),
    ("mAP75", "metrics/mAP75(B)", 100.0),
    ("mAP50_95", "metrics/mAP50-95(B)", 100.0),
    ("APS", "APS", 1.0),
    ("APM", "APM", 1.0),
    ("APL", "APL", 1.0),
)
ACTIVE: list[subprocess.Popen] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def ensure_within_root(path: Path, run_root: Path) -> None:
    resolved = path.resolve(strict=False)
    if resolved != run_root and run_root not in resolved.parents:
        raise RuntimeError(f"Path escapes run root: {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def process_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{ROOT}{os.pathsep}{environment.get('PYTHONPATH', '')}",
            "WANDB_DISABLED": "true",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    return environment


def terminate_active() -> None:
    for process in ACTIVE:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for process in ACTIVE:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    ACTIVE.clear()


def read_seed_summary(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {}
    for label, key, scale in METRICS:
        source = payload["metrics"] if key.startswith("metrics/") else payload.get("scale_metrics_percent", {})
        value = float(source[key]) * scale
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite {label} in {path}")
        values[label] = value
    return values


def collect_stage(run_root: Path, stage: str) -> dict:
    test_root = run_root / "test" / stage
    rows = []
    for seed in range(3):
        summary_path = test_root / f"seed{seed}" / "summary_metrics.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        row = {"seed": seed, "summary": str(summary_path), **read_seed_summary(summary_path)}
        rows.append(row)
    summary = {}
    for label, _, _ in METRICS:
        values = [row[label] for row in rows]
        summary[label] = {"mean": mean(values), "sd": stdev(values), "min": min(values), "max": max(values)}
    return {"rows": rows, "summary": summary}


def write_aggregate(run_root: Path, completed_stages: list[str]) -> None:
    aggregate = {"run_root": str(run_root), "updated_at": utc_now(), "stages": {}}
    csv_rows = []
    for stage in completed_stages:
        aggregate["stages"][stage] = collect_stage(run_root, stage)
        for row in aggregate["stages"][stage]["rows"]:
            csv_rows.append({"stage": stage, **row})
    if {"stage1_l0", "stage1_od"}.issubset(aggregate["stages"]):
        paired = {}
        l0_rows = aggregate["stages"]["stage1_l0"]["rows"]
        od_rows = aggregate["stages"]["stage1_od"]["rows"]
        for label, _, _ in METRICS:
            deltas = [od_rows[index][label] - l0_rows[index][label] for index in range(3)]
            paired[label] = {"deltas_pp": deltas, "mean_pp": mean(deltas), "sd_pp": stdev(deltas), "wins": sum(value > 0 for value in deltas)}
        aggregate["paired_l0_vs_od"] = paired
    if {"stage1_l0", "stage2_o", "stage2_d", "stage2_od"}.issubset(aggregate["stages"]):
        nonadditive = {}
        l0 = aggregate["stages"]["stage1_l0"]["summary"]
        o = aggregate["stages"]["stage2_o"]["summary"]
        d = aggregate["stages"]["stage2_d"]["summary"]
        od = aggregate["stages"]["stage2_od"]["summary"]
        for label, _, _ in METRICS:
            o_effect = o[label]["mean"] - l0[label]["mean"]
            d_effect = d[label]["mean"] - l0[label]["mean"]
            od_effect = od[label]["mean"] - l0[label]["mean"]
            nonadditive[label] = {
                "o_effect_vs_l0_pp": o_effect,
                "d_effect_vs_l0_pp": d_effect,
                "od_effect_vs_l0_pp": od_effect,
                "interaction_od_minus_o_minus_d_pp": od_effect - o_effect - d_effect,
                "note": "Descriptive cross-wave interaction; replicate this full factorial design with a same-wave L0 for confirmatory inference.",
            }
        aggregate["nonadditivity_descriptive"] = nonadditive
    atomic_json(run_root / "test" / "AllSeed_Summary.json", aggregate)
    fields = ["stage", "seed", "summary", *[label for label, _, _ in METRICS]]
    with (run_root / "test" / "AllSeed_Summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)


def run_training_wave(run_root: Path, stage: str, variant: str, update_state) -> None:
    stage_root = run_root / "train" / stage
    log_root = stage_root / "logs"
    stage_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    config_snapshot = run_root / "train" / "configs" / f"{stage}.yaml"
    shutil.copy2(CONFIGS[variant], config_snapshot)
    processes: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for seed in range(3):
            log_path = log_root / f"seed{seed}.log"
            handle = log_path.open("x", encoding="utf-8")
            command = [
                str(PYTHON), str(TRAIN_WORKER), "--variant", variant, "--seed", str(seed),
                "--project", str(stage_root), "--name", f"seed{seed}",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=process_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ACTIVE.append(process)
            processes.append((seed, process, handle))
        update_state({"current_stage": stage, "phase": "training", "worker_pids": {str(seed): process.pid for seed, process, _ in processes}})
        failures = []
        while any(process.poll() is None for _, process, _ in processes):
            for seed, process, _ in processes:
                code = process.poll()
                if code not in (None, 0):
                    failures.append((seed, code))
            if failures:
                terminate_active()
                raise RuntimeError(f"Training worker failure in {stage}: {failures}")
            time.sleep(10)
        failures = [(seed, process.returncode) for seed, process, _ in processes if process.returncode != 0]
        if failures:
            raise RuntimeError(f"Training worker failure in {stage}: {failures}")
        for seed, _, _ in processes:
            checkpoint = stage_root / f"seed{seed}" / "weights" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    finally:
        for _, process, handle in processes:
            if process in ACTIVE:
                ACTIVE.remove(process)
            handle.close()


def run_test_wave(run_root: Path, stage: str, update_state) -> None:
    train_root = run_root / "train" / stage
    test_root = run_root / "test" / stage
    log_root = test_root / "logs"
    test_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    update_state({"current_stage": stage, "phase": "testing", "worker_pids": {}})
    for seed in range(3):
        checkpoint = train_root / f"seed{seed}" / "weights" / "best.pt"
        log_path = log_root / f"seed{seed}.log"
        command = [str(PYTHON), str(TEST_WORKER), "--weights", str(checkpoint), "--project", str(test_root), "--name", f"seed{seed}"]
        with log_path.open("x", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=process_environment(), stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Test worker failure in {stage}, seed={seed}; see {log_path}")
        summary_path = test_root / f"seed{seed}" / "summary_metrics.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    if ROOT not in run_root.parents:
        raise RuntimeError(f"Run root must be inside project: {run_root}")
    if run_root.exists():
        allowed = {"train", "launcher.pid"}
        existing = {path.name for path in run_root.iterdir()}
        if existing - allowed or not (run_root / "train").is_dir():
            raise FileExistsError(f"Run root already exists: {run_root}")
    for path in (PYTHON, TRAIN_WORKER, TEST_WORKER, DATA_YAML, ROOT / "yolov13n.pt", ROOT / "test.py", *CONFIGS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "configs").mkdir(parents=True)
    (run_root / "test").mkdir()
    state_path = run_root / "state.json"
    state = {
        "status": "running",
        "run_root": str(run_root),
        "started_at": utc_now(),
        "dataset": str(DATA_YAML),
        "protocol": {"epochs": 160, "device": 0, "workers": 0, "amp": False, "deterministic": True, "plots": False, "imgsz": 640, "batch": 16, "seeds": [0, 1, 2]},
        "waves": [{"stage": stage, "variant": variant, "purpose": purpose} for stage, variant, purpose in WAVES],
        "completed_stages": [],
        "current_stage": None,
        "phase": None,
        "worker_pids": {},
    }

    def update_state(changes: dict) -> None:
        state.update(changes)
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)

    def handle_signal(signum, _frame) -> None:
        terminate_active()
        update_state({"status": "cancelled", "cancelled_at": utc_now(), "failure_reason": f"signal {signum}"})
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    update_state({})
    try:
        for stage, variant, _ in WAVES:
            run_training_wave(run_root, stage, variant, update_state)
            run_test_wave(run_root, stage, update_state)
            state["completed_stages"].append(stage)
            write_aggregate(run_root, state["completed_stages"])
            update_state({"phase": "completed_stage", "worker_pids": {}})
        update_state({"status": "completed", "completed_at": utc_now(), "current_stage": None, "phase": None, "worker_pids": {}})
    except SystemExit:
        raise
    except Exception as error:
        terminate_active()
        update_state({"status": "failed", "failed_at": utc_now(), "failure_reason": repr(error), "worker_pids": {}})
        raise


if __name__ == "__main__":
    main()
