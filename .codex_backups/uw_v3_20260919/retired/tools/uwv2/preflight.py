"""CPU model/loss/transfer checks from any cwd; no training or exports."""
from __future__ import annotations
import ast
import gc
import importlib.util
import subprocess
from pathlib import Path
from common import ROOT, VARIANTS, BASELINE, WEIGHTS, model_yaml, runtime, write_json


def main():
    runtime()
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    from ultralytics.utils.torch_utils import get_flops
    from ultralytics.nn.modules import C3k2_UWFEM, StableBiConcat2
    torch.set_num_threads(2)
    # Parse the whole application tree, not only the new modules.
    paths = [p for base in ('ultralytics', 'tools', 'tests') for p in (ROOT/base).rglob('*.py')
             if '__pycache__' not in p.parts]
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for filename in ('run_yolo11uw_v2.sh', 'wait_yolo11uw_v2_after_upstream.sh'):
        subprocess.run(['bash', '-n', str(ROOT/'scripts'/filename)], check=True)
    spec = importlib.util.spec_from_file_location('uw_v2_tests', ROOT/'tests/test_uw_v2.py')
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    names = sorted(n for n in vars(tests) if n.startswith('test_'))
    for name in names:
        getattr(tests, name)()
        print(f'PASS {name}', flush=True)
    source = YOLO(str(WEIGHTS)).model.float().state_dict()
    results = {}
    reference_cfg = yaml.safe_load((BASELINE/'train/yolo11n_uw_A0.yaml').read_text())
    for variant in ('B0', *VARIANTS):
        path = ROOT/f'ultralytics/cfg/models/11/yolo11n_uw_v2_{variant}.yaml'
        cfg = yaml.safe_load(path.read_text())
        if variant == 'B0':
            assert cfg['backbone'] == reference_cfg['backbone']
            assert cfg['head'] == reference_cfg['head']
        yolo = YOLO(str(path)).load(str(WEIGHTS))
        net = yolo.model
        assert net.stride.tolist() == [8, 16, 32]
        fem = sum(isinstance(m, C3k2_UWFEM) for m in net.modules())
        fusion = sum(isinstance(m, StableBiConcat2) for m in net.modules())
        assert fem == (2 if variant in ('B1','B4','B5') else 0)
        assert fusion == (4 if variant in ('B2','B4','B5') else 0)
        loaded = net.state_dict()
        transferred = [k for k,v in source.items() if k in loaded and v.shape == loaded[k].shape]
        assert all(torch.equal(source[k], loaded[k]) for k in transferred)
        checks = {}
        for layer in (4,6):
            keys = [k for k in transferred if k.startswith(f'model.{layer}.')
                    and any(f'.{part}.' in k for part in ('cv1','cv2','m'))]
            assert keys
            checks[str(layer)] = len(keys)
        net.eval()
        with torch.no_grad():
            prediction = net(torch.zeros(1,3,640,640))
            assert torch.isfinite(prediction[0]).all()
        params = sum(p.numel() for p in net.parameters())
        flops = get_flops(net, imgsz=640)
        net.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        net.train()
        # Real detector loss and backward at the requested image size.
        batch = dict(img=torch.rand(1,3,640,640), batch_idx=torch.tensor([0.]),
                     cls=torch.tensor([[1.]]), bboxes=torch.tensor([[.5,.5,.2,.2]]))
        loss, parts = net.loss(batch)
        loss.sum().backward()
        assert torch.isfinite(loss).all()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        criterion = net.criterion.bbox_loss
        assert criterion.use_wiou_v3 == (variant in ('B3','B5'))
        assert not criterion.use_inner_ciou
        if criterion.use_wiou_v3:
            mean = criterion.wiou_v3.iou_mean.clone()
            with torch.no_grad():
                net.loss(batch)
            assert torch.equal(mean, criterion.wiou_v3.iou_mean)
            assert mean.item() != 1.0
        results[variant] = dict(params=params, GFLOPs=flops, stride=[8,16,32],
            transferred=len(transferred), total_state_items=len(loaded),
            original_P3_P4_tensors_verified=checks, FEM_blocks=fem, weighted_fusions=fusion,
            use_wiou_v3=criterion.use_wiou_v3, loss_parts=parts.detach().tolist())
        print('MODEL_PASS',variant,results[variant],flush=True)
        del yolo, net, loaded, loss, parts, criterion, prediction
        gc.collect()
    write_json(ROOT/'uw_v2_preflight.json', dict(status='passed', python_files_parsed=len(paths),
               unit_tests=names, models=results, exports_performed=False))

if __name__ == '__main__':
    main()
