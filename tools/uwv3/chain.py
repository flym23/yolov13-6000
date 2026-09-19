"""One controlled three-seed batch, with atomic state and terminal-state chaining."""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import (DATA, ROOT, SEEDS, TRAIN_ARGS, VARIANTS, WEIGHTS, model_yaml, now,
                       read_json, required_files, run_root, sha256, write_json)

TERMINAL = {'completed', 'failed', 'cancelled'}


class Cancelled(Exception):
    pass


def cancelled(signum, frame):
    raise Cancelled(f'Received signal {signum}')


def state_update(root, **fields):
    state = read_json(root / 'state.json') if (root / 'state.json').is_file() else {}
    state.update(fields, updated_at=now())
    write_json(root / 'state.json', state)
    print(f'{now()} {root.name}: {fields}', flush=True)
    return state


def initialize(args):
    root = run_root(args.variant, args.run_id)
    for sub in ('train', 'test'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if not (root / 'state.json').is_file():
        state_update(root, status='waiting', variant=args.variant, run_id=args.run_id,
                     created_at=now(), upstream_state=str(args.upstream) if args.upstream else None,
                     upstream_status=None, upstream_failure_reason=None, failure_reason=None, worker_pids=[])
    state = read_json(root / 'state.json')
    if state['variant'] != args.variant or state['run_id'] != args.run_id:
        raise RuntimeError('Immutable experiment identity mismatch')
    return root


def wait_upstream(args, root):
    if not args.upstream or not args.upstream.is_absolute():
        raise ValueError('An absolute upstream state path is required')
    if args.upstream == root / 'state.json':
        raise ValueError('Cannot wait for this experiment itself')
    state = read_json(root / 'state.json')
    if state['status'] != 'waiting':
        raise RuntimeError(f'Refusing duplicate wait: {state["status"]}')
    state_update(root, status='waiting', upstream_state=str(args.upstream), waiting_since=now())
    previous = None
    while True:
        try:
            upstream = read_json(args.upstream)
            status = upstream.get('status')
        except (OSError, ValueError):
            upstream, status = {}, 'missing_or_invalid'
        if status != previous:
            print(f'{now()} upstream={args.upstream} status={status}', flush=True)
            previous = status
        if status in TERMINAL:
            state_update(root, upstream_status=status,
                         upstream_failure_reason=upstream.get('failure_reason'), triggered_at=now())
            # The same launcher PID execs the single formal run script.
            os.execv('/usr/bin/bash', ['bash', str(ROOT / 'scripts/run_yolo11uw_v3.sh'),
                                     args.variant, args.run_id])
        time.sleep(30)


def stop_children(children):
    # Each worker has a private session: terminate its DataLoader children as well.
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for process in children:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_workers(args, root, mode, seeds):
    children, handles = [], []
    try:
        for seed in seeds:
            log = (root / 'test' / 'benchmark_worker.log' if mode=='benchmark'
                   else root / mode / f'seed{seed}_worker.log')
            handle = log.open('a', encoding='utf-8', buffering=1)
            handles.append(handle)
            command = [sys.executable, '-u', str(ROOT / 'tools/uwv3/worker.py'), mode,
                       '--variant', args.variant, '--seed', str(seed), '--run-root', str(root)]
            if args.resume:
                command.append('--resume')
            env = dict(os.environ, PYTHONHASHSEED=str(seed), OMP_NUM_THREADS='2',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_DISABLED='true', PIN_MEMORY='false')
            process = subprocess.Popen(command, cwd='/tmp', env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            (root / f'{mode}_seed{seed}.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
        state_update(root, phase=mode, worker_pids=[p.pid for p in children])
        while True:
            codes = [p.poll() for p in children]
            for process, code in zip(children, codes):
                if code is not None and code != 0:
                    raise RuntimeError(f'{mode} worker {process.pid} exited {code}; inspect {root / mode}')
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
    except BaseException:
        stop_children(children)
        raise
    finally:
        for handle in handles:
            handle.close()
        state_update(root, worker_pids=[])


def aggregate(root):
    results = []
    for seed in SEEDS:
        folder = root / 'test' / f'seed{seed}'
        if not (folder / 'scale_ap_metrics.json').is_file():
            raise FileNotFoundError(folder / 'scale_ap_metrics.json')
        results.append(read_json(folder / 'summary_metrics.json'))
    summary = {}
    for metric in results[0]['metrics']:
        values = [row['metrics'][metric] for row in results]
        valid = [v for v in values if v is not None]
        summary[metric] = dict(values=values, mean=statistics.mean(valid) if valid else None,
            std_sample=statistics.stdev(valid) if len(valid) > 1 else None,
            min=min(valid) if valid else None, max=max(valid) if valid else None,
            n=len(valid))
    map_values = [row['metrics']['mAP50_95'] for row in results]
    payload = dict(seeds=results, statistics=summary, units='AP/P/R are fractions, not percentages',
                   best_seed=int(max(range(3), key=map_values.__getitem__)),
                   worst_seed=int(min(range(3), key=map_values.__getitem__)))
    write_json(root / 'test/AllSeed_summary.json', payload)
    with (root / 'test/AllSeed_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'seed0', 'seed1', 'seed2', 'mean', 'std_sample', 'min', 'max', 'n'])
        for metric, stats in summary.items():
            writer.writerow([metric, *stats['values'], stats['mean'], stats['std_sample'], stats['min'], stats['max'], stats['n']])
    return payload


def compare_all(args, root):
    rows = []
    baseline_file = run_root('C0', args.run_id) / 'test/AllSeed_summary.json'
    baseline = read_json(baseline_file)
    write_json(root / 'test/C0_baseline.json', baseline)
    for variant in VARIANTS:
        candidate = run_root(variant, args.run_id)
        source = candidate / 'test/AllSeed_summary.json'
        status_file = candidate / 'state.json'
        row = dict(variant=variant, root=str(candidate),
                   status=read_json(status_file)['status'] if status_file.is_file() else 'missing')
        if source.is_file():
            payload = read_json(source)
            row['statistics'] = payload['statistics']
            if baseline:
                deltas = [payload['seeds'][s]['metrics']['mAP50_95'] -
                          baseline['seeds'][s]['metrics']['mAP50_95'] for s in SEEDS]
                row['paired_mAP50_95_delta'] = dict(values=deltas, mean=statistics.mean(deltas),
                    std_sample=statistics.stdev(deltas), improved_seed_count=sum(d > 0 for d in deltas))
        rows.append(row)
    rows[-1]['status'] = 'completed'
    write_json(root / 'test/Ablation_comparison.json', dict(variants=rows,
        baseline_source=str(baseline_file), baseline_sha256=sha256(baseline_file),
        baseline_policy='Fresh C0 on URPC2020half with the same training contract.',
        caveat='All experiments: epochs=300/patience=30 on URPC2020half. '
               'Three seeds are descriptive evidence; val and test share images/test, not independent testing.'))


def run(args, root):
    if args.resume:
        raise ValueError('UW v3 resume is disabled; use a new immutable run_id for a fresh attempt')
    # A nonblocking lock rejects duplicate launches, including concurrent resume requests.
    with (root / 'manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(root / 'state.json')
        if state['status'] == 'completed' or (state['status'] != 'waiting' and not args.resume):
            raise RuntimeError(f'Refusing duplicate experiment: {state["status"]}')
        (root / 'launcher.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
        try:
            files = required_files(args.variant)
            contract = {str(p.relative_to(ROOT)): sha256(p) for p in files}
            metadata_file = root / 'metadata.json'
            if metadata_file.is_file() and read_json(metadata_file)['sha256'] != contract:
                raise RuntimeError('Code/config/weights changed since initial launch; use a new run_id')
            write_json(metadata_file, dict(variant=args.variant, run_id=args.run_id, sha256=contract,
                train_args=TRAIN_ARGS, seeds=SEEDS, python=sys.executable, root=str(ROOT), root_dir=str(root),
                data=str(DATA), weights=str(WEIGHTS)))
            for source in (DATA, model_yaml(args.variant)):
                (root / 'train' / source.name).write_bytes(source.read_bytes())
            write_json(root / 'train/train_parameters.json', TRAIN_ARGS)
            state_update(root, status='running', started_at=now(), launcher_pid=os.getpid(), failure_reason=None)
            run_workers(args, root, 'train', SEEDS)
            # Sequential evaluation avoids three competing latency benchmarks.
            for seed in SEEDS:
                run_workers(args, root, 'test', [seed])
            state_update(root, phase='aggregate')
            aggregate(root)
            if args.variant == 'C3':
                compare_all(args, root)
                run_workers(args, root, 'benchmark', [0])
            state_update(root, status='completed', phase='done', completed_at=now(), worker_pids=[])
        except BaseException as exc:
            state_update(root, status='cancelled' if isinstance(exc, Cancelled) else 'failed',
                         failure_reason=str(exc), finished_at=now(), worker_pids=[])
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['wait', 'run'])
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    root = initialize(args)
    if args.mode == 'wait':
        with (root / 'waiter.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read_json(root / 'state.json')['status'] != 'waiting':
                raise RuntimeError('Refusing duplicate waiter for a started experiment')
            (root / 'launcher.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
            try:
                wait_upstream(args, root)
            except BaseException as exc:
                state_update(root, status='cancelled' if isinstance(exc, Cancelled) else 'failed', failure_reason=str(exc))
                raise
    else:
        run(args, root)


if __name__ == '__main__':
    main()
