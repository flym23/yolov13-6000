#!/usr/bin/env python3
"""Wait for an exact source-project training group, then exec the DCRA chain."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--run-script", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-roots", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--settle-polls", type=int, default=2)
    args = parser.parse_args()
    if args.expected_roots < 0:
        parser.error("--expected-roots must be non-negative")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.settle_polls <= 0:
        parser.error("--settle-polls must be positive")
    return args


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": utc_now()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_process(pid: int) -> dict[str, object] | None:
    process_dir = Path("/proc") / str(pid)
    try:
        cwd = (process_dir / "cwd").resolve(strict=True)
        cmdline_raw = (process_dir / "cmdline").read_bytes()
        stat_raw = (process_dir / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    fields = stat_raw[stat_raw.rfind(")") + 2 :].split()
    if len(fields) < 20:
        return None
    command = [part.decode("utf-8", "replace") for part in cmdline_raw.split(b"\0") if part]
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "start_ticks": int(fields[19]),
        "cwd": str(cwd),
        "command": command,
    }


def is_training_command(command: list[str]) -> bool:
    """Recognize train.py and project-specific train_*_worker.py entry points."""
    for argument in command:
        name = Path(argument).name
        if name == "train.py" or (name.startswith("train_") and name.endswith(".py")):
            return True
    return False


def source_train_processes(source_root: Path) -> dict[int, dict[str, object]]:
    matches: dict[int, dict[str, object]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process = read_process(int(entry.name))
        if process is None or process["cwd"] != str(source_root):
            continue
        command = process["command"]
        if is_training_command(command):
            matches[int(process["pid"])] = process
    return matches


def root_processes(processes: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        (process for process in processes.values() if int(process["ppid"]) not in processes),
        key=lambda process: int(process["pid"]),
    )


def execute_wait(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve(strict=True)
    target_root = args.target_root.resolve(strict=True)
    run_script = args.run_script.resolve(strict=True)
    if not os.access(run_script, os.R_OK):
        raise PermissionError(run_script)

    lock_path = args.state.parent / "wait.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another DCRA waiter is already active: {lock_path}") from exc

    initial = source_train_processes(source_root)
    roots = root_processes(initial)
    if len(roots) != args.expected_roots:
        raise RuntimeError(
            f"Expected {args.expected_roots} source training roots, found {len(roots)}: "
            f"{[process['pid'] for process in roots]}"
        )

    started_at = utc_now()
    started_monotonic = time.monotonic()
    stable_empty_polls = 0
    base_state = {
        "run_id": args.run_id,
        "waiter_pid": os.getpid(),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "run_script": str(run_script),
        "started_at": started_at,
        "expected_roots": args.expected_roots,
        "initial_roots": roots,
    }
    while stable_empty_polls < args.settle_polls:
        current = source_train_processes(source_root)
        stable_empty_polls = stable_empty_polls + 1 if not current else 0
        write_state(
            args.state,
            {
                **base_state,
                "status": "waiting",
                "remaining_processes": sorted(current.values(), key=lambda process: int(process["pid"])),
                "remaining_count": len(current),
                "stable_empty_polls": stable_empty_polls,
                "elapsed_seconds": round(time.monotonic() - started_monotonic, 1),
            },
        )
        if stable_empty_polls < args.settle_polls:
            time.sleep(args.poll_seconds)

    write_state(
        args.state,
        {
            **base_state,
            "status": "launching",
            "remaining_processes": [],
            "remaining_count": 0,
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 1),
        },
    )
    os.chdir(target_root)
    environment = os.environ.copy()
    environment["DCRA_RUN_ID"] = args.run_id
    os.execve("/bin/bash", ["/bin/bash", str(run_script)], environment)


def main() -> None:
    args = parse_args()
    try:
        execute_wait(args)
    except Exception as exc:
        write_state(
            args.state,
            {
                "run_id": args.run_id,
                "status": "failed",
                "waiter_pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
