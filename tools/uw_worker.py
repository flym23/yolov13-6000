"""One seeded training worker, or one sequential evaluation worker."""
from __future__ import annotations

import argparse
import gc
import math
import os
import time
from pathlib import Path

from uw_common import DATA, ROOT, TRAIN_ARGS, WEIGHTS, model_yaml, read_json, runtime, sha256, write_json


def assert_finite(trainer):
    import torch
    if not torch.isfinite(trainer.loss.detach()).all():
        raise FloatingPointError(f'Non-finite training loss at epoch {trainer.epoch}')


def train(args):
    runtime()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(args.seed, deterministic=True)
    out = args.run_root / 'train' / f'seed{args.seed}'
    completion = out / 'training_complete.json'
    if args.resume and completion.is_file():
        if not (out / 'weights/best.pt').is_file():
            raise FileNotFoundError(out / 'weights/best.pt')
        return
    last = out / 'weights/last.pt'
    if args.resume and last.is_file():
        model = YOLO(str(last))
        model.add_callback('on_train_batch_end', assert_finite)
        model.train(resume=True)
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f'Existing incomplete output without resumable last.pt: {out}')
        model = YOLO(str(model_yaml(args.variant))).load(str(WEIGHTS))
        model.add_callback('on_train_batch_end', assert_finite)
        model.train(data=str(DATA), seed=args.seed, project=str(out.parent), name=out.name,
                    exist_ok=False, **TRAIN_ARGS)
    if not (out / 'weights/best.pt').is_file():
        raise FileNotFoundError(out / 'weights/best.pt')
    write_json(completion, dict(seed=args.seed, variant=args.variant,
                               best_sha256=sha256(out / 'weights/best.pt')))


def coco_scale_metrics(model, prediction_path, out):
    """Real COCO area-range evaluation, including ignore rules and maxDets=100."""
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import yaml
    from ultralytics.data.utils import img2label_paths
    data = yaml.safe_load(DATA.read_text(encoding='utf-8'))
    image_root = Path(data['path']) / data['test']
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = sorted(p for p in image_root.rglob('*') if p.suffix.lower() in suffixes)
    gt = dict(info={}, images=[], annotations=[], categories=[
        dict(id=int(i) + 1, name=n) for i, n in model.names.items()])
    seen_ids = set()
    for p in images:
        image_id = int(p.stem) if p.stem.isnumeric() else p.stem
        if image_id in seen_ids:
            raise ValueError(f'Duplicate COCO image id: {image_id}')
        seen_ids.add(image_id)
        with Image.open(p) as im:
            width, height = im.size
        gt['images'].append(dict(id=image_id, width=width, height=height, file_name=p.name))
        label = Path(img2label_paths([str(p)])[0])
        if not label.is_file():
            raise FileNotFoundError(label)
        for row in label.read_text(encoding='utf-8').splitlines():
            if not row.strip():
                continue
            cls, cx, cy, w, h = map(float, row.split())
            if int(cls) != cls or not 0 <= cls < len(model.names) or not all(
                    math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, w, h)):
                raise ValueError(f'Invalid YOLO annotation: {label}: {row}')
            bw, bh = w * width, h * height
            gt['annotations'].append(dict(id=len(gt['annotations']) + 1, image_id=image_id,
                category_id=int(cls) + 1, bbox=[cx * width - bw / 2, cy * height - bh / 2, bw, bh],
                area=bw * bh, iscrowd=0))
    write_json(out / 'coco_ground_truth.json', gt)
    truth = COCO(str(out / 'coco_ground_truth.json'))
    predictions = read_json(prediction_path) if prediction_path.is_file() else []
    if predictions:
        detected = truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = dict(images=gt['images'], categories=gt['categories'], annotations=[])
        detected.createIndex()
    evaluator = COCOeval(truth, detected, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ['AP', 'AP50', 'AP75', 'APS', 'APM', 'APL']
    metrics = {k: (float(v) if v >= 0 else None) for k, v in zip(keys, evaluator.stats[:6])}
    write_json(out / 'scale_ap_metrics.json', dict(method='pycocotools COCOeval; original image area; maxDets=100',
        units='fraction', area_ranges={'small': [0, 1024], 'medium': [1024, 9216], 'large': [9216, None]},
        metrics=metrics))
    return metrics


def evaluate(args):
    runtime()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, init_seeds
    init_seeds(args.seed, deterministic=True)
    weights = args.run_root / 'train' / f'seed{args.seed}/weights/best.pt'
    out = args.run_root / 'test' / f'seed{args.seed}'
    if args.resume and (out / 'summary_metrics.json').is_file() and (out / 'scale_ap_metrics.json').is_file():
        saved = read_json(out / 'summary_metrics.json')
        if saved['weights_sha256'] == sha256(weights):
            return
    model = YOLO(str(weights))
    params = sum(p.numel() for p in model.model.parameters())
    gflops = get_flops(model.model, imgsz=640)
    result = model.val(data=str(DATA), split='test', device=0, batch=16, workers=2, imgsz=640,
                       amp=False, half=False, plots=False, save_json=True, conf=0.001, iou=0.7,
                       max_det=300, project=str(out.parent), name=out.name, exist_ok=True)
    metrics = dict(P=float(result.box.mp), R=float(result.box.mr), mAP50=float(result.box.map50),
                   mAP50_95=float(result.box.map), params=params, GFLOPs=gflops)
    metrics.update({k: v for k, v in coco_scale_metrics(model, out / 'predictions.json', out).items()
                    if k in ('APS', 'APM', 'APL')})
    # Batch-1 FP32 model latency, sequential across our seeds; external GPU load remains possible.
    net = model.model.to('cuda:0').float().eval()
    x = torch.zeros(1, 3, 640, 640, device='cuda:0')
    with torch.inference_mode():
        for _ in range(20):
            net(x)
        torch.cuda.synchronize()
        timings = []
        for _ in range(100):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            net(x)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    metrics['latency_ms_batch1_fp32'] = sum(timings) / len(timings)
    if not all(v is None or math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError(metrics)
    write_json(out / 'summary_metrics.json', dict(variant=args.variant, seed=args.seed, metrics=metrics,
        weights=str(weights), weights_sha256=sha256(weights), latency_samples_ms=timings,
        latency_note='20 warmups, 100 CUDA-event forwards, no NMS; shared GPU, not isolated deployment latency',
        split='test', split_caveat='Existing dataset uses images/test for both validation and testing',
        speed_ms_per_image=result.speed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--variant', choices=['A0', 'A1', 'A2', 'A3', 'A4'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.run_root.is_absolute() or args.run_root.parent != ROOT / 'runs':
        raise ValueError('run-root must be an absolute child of project/runs')
    (train if args.mode == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
