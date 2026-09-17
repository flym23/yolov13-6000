"""Locked YOLO11-UW experiment contract shared by the launcher and workers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data.yaml'
WEIGHTS = ROOT / 'yolov11n.pt'
VARIANTS = ('A0', 'A1', 'A2', 'A3', 'A4')
SEEDS = (0, 1, 2)
TRAIN_ARGS = dict(epochs=300, patience=30, device=0, workers=2, amp=False,
                  deterministic=True, plots=False, imgsz=640, batch=16,
                  optimizer='auto', cos_lr=True, close_mosaic=10, cache=False)


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
    return ROOT / 'ultralytics/cfg/models/11' / f'yolo11n_uw_{variant}.yaml'


def run_root(variant, run_id):
    if variant not in VARIANTS or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid variant or immutable run_id')
    return ROOT / 'runs' / f'yolo11uw_{variant}_{run_id}'


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
    return [DATA, WEIGHTS, model_yaml(variant), ROOT / 'tools/uw_worker.py',
            ROOT / 'tools/uw_common.py', ROOT / 'tools/uw_chain.py',
            ROOT / 'ultralytics/nn/tasks.py', ROOT / 'ultralytics/nn/modules/eca.py',
            ROOT / 'ultralytics/nn/modules/__init__.py', ROOT / 'ultralytics/utils/loss.py',
            ROOT / 'ultralytics/utils/inner_iou.py']
