"""Locked YOLO11-UW experiment contract shared by the launcher and workers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data.yaml'
WEIGHTS = ROOT / 'yolov11n.pt'
VARIANTS = ('B1', 'B3', 'B2', 'B4', 'B5')
BASELINE = ROOT / 'runs/yolo11uw_A0_urpc2020_20260916_r1'
SEEDS = (0, 1, 2)
TRAIN_ARGS = dict(epochs=180, patience=0, device=0, workers=2, amp=False,
                  deterministic=True, plots=False, imgsz=640, batch=16,
                  optimizer='auto', cos_lr=True, close_mosaic=10, cache=False, resume=False)


def runtime():
    # Set before importing torch or ultralytics, including from a non-project cwd.
    sys.path.insert(0, str(ROOT))
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['PIN_MEMORY'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import ultralytics
    package = Path(ultralytics.__file__).resolve()
    if ROOT not in package.parents:
        raise RuntimeError(f'Unexpected ultralytics import: {package}')
    print(f'PROJECT_IMPORT={package}', flush=True)
    return ultralytics


def model_yaml(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    return ROOT / 'ultralytics/cfg/models/11' / f'yolo11n_uw_v2_{variant}.yaml'


def run_root(variant, run_id):
    if variant not in VARIANTS or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid variant or immutable run_id')
    return ROOT / 'runs' / f'yolo11uw_v2_{variant}_{run_id}'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def required_files(variant):
    return [DATA, WEIGHTS, model_yaml(variant), Path(__file__).with_name('worker.py'),
            Path(__file__).with_name('common.py'), Path(__file__).with_name('chain.py'),
            ROOT / 'ultralytics/nn/tasks.py', ROOT / 'ultralytics/nn/modules/uw_v2.py',
            ROOT / 'ultralytics/nn/modules/__init__.py', ROOT / 'ultralytics/utils/loss.py',
            ROOT / 'ultralytics/utils/wiou_v3.py',
            ROOT / 'ultralytics/nn/modules/block.py', ROOT / 'ultralytics/nn/modules/conv.py',
            ROOT / 'ultralytics/nn/modules/head.py', ROOT / 'ultralytics/utils/tal.py',
            ROOT / 'ultralytics/cfg/default.yaml',
            ROOT / 'scripts/run_yolo11uw_v2.sh', ROOT / 'scripts/wait_yolo11uw_v2_after_upstream.sh']
