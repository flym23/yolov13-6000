"""Audit data, integration, disabled-path parity and an optional one-epoch smoke."""
import argparse
import ast
import gc
import importlib.util
import sys
import types
from pathlib import Path
from common import DATA, ROOT, TRAIN_ARGS, VARIANTS, WEIGHTS, model_yaml, runtime, write_json

BACKUP=ROOT/'.codex_backups/uw_v3_20260919'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    runtime()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    from ultralytics.utils.torch_utils import get_flops, init_seeds
    from ultralytics.utils.loss import v8DetectionLoss
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'tools'))
    from analyze_urpc_scale import audit
    audits={split:audit(DATA,split,640) for split in ('train','test')}
    for split,value in audits.items():
        assert value['missing_label_files']==0
        assert value['image_count']==(2301 if split=='train' else 986)
    write_json(ROOT/'uw_v3_scale_audit.json',audits)
    print('SCALE_AUDIT',audits['train']['ratios'],flush=True)
    spec=importlib.util.spec_from_file_location('v3_tests',ROOT/'tests/test_uw_v3_training.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    names=sorted(n for n in vars(tests) if n.startswith('test_'))
    for name in names:
        getattr(tests,name)();print('PASS',name,flush=True)
    # Load the original criterion under its package, without the now-retired optional WIoU import.
    old=types.ModuleType('ultralytics.utils._v3_old_loss')
    old.__package__='ultralytics.utils'
    source=(BACKUP/'ultralytics/utils/loss.py').read_text(encoding='utf-8')
    source=source.replace('from .wiou_v3 import WiseIoUv3Loss, weighted_wiou_mean','')
    exec(compile(source,'original_loss.py','exec'),old.__dict__)
    newsource=(ROOT/'ultralytics/utils/loss.py').read_text(encoding='utf-8')
    def cls(text,name): return next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==name)
    assert ast.dump(cls(source,'DFLoss'))==ast.dump(cls(newsource,'DFLoss'))
    def call(text): return next(n for n in cls(text,'v8DetectionLoss').body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    assert ast.dump(call(source))==ast.dump(call(newsource))
    results={};reference=None;reference_state=None
    for variant in VARIANTS:
        init_seeds(0,deterministic=True)
        yolo=YOLO(str(model_yaml(variant))).load(str(WEIGHTS))
        net=yolo.model
        net.args=IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        assert net.stride.tolist()==[8,16,32]
        signature={k:tuple(v.shape) for k,v in net.state_dict().items()}
        if reference_state is None: reference_state=signature
        assert signature==reference_state
        net.eval()
        with torch.no_grad(): assert torch.isfinite(net(torch.zeros(1,3,640,640))[0]).all()
        params=sum(p.numel() for p in net.parameters());flops=get_flops(net,640)
        if reference is None: reference=(params,flops)
        assert (params,flops)==reference
        net.train()
        batch=dict(img=torch.rand(1,3,640,640),batch_idx=torch.tensor([0.,0.]),
                   cls=torch.tensor([[1.],[2.]]),bboxes=torch.tensor([[.5,.5,.005,.005],[.2,.2,.2,.2]]))
        preds=net(batch['img'])
        loss=net.init_criterion()
        assert loss.assigner.stal_enabled==(variant in ('C1','C3'))
        assert loss.bbox_loss.small_nwd_enabled==(variant in ('C2','C3'))
        value,parts=loss(preds,batch)
        if variant=='C0':
            oldvalue,oldparts=old.v8DetectionLoss(net)(preds,batch)
            assert torch.equal(value,oldvalue) and torch.equal(parts,oldparts)
        value.sum().backward()
        assert torch.isfinite(value).all()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        results[variant]=dict(params=params,GFLOPs=flops,stride=[8,16,32],loss=parts.detach().tolist(),
                              original_inference_graph=True)
        print('MODEL_PASS',variant,results[variant],flush=True)
        del net,yolo,loss,value,parts,preds
        gc.collect()
    write_json(ROOT/'uw_v3_preflight.json',dict(status='passed',unit_tests=names,models=results,
        baseline_loss_bitwise_parity=True,DFL_unchanged=True,classification_call_unchanged=True,
        kd_status='deferred: no native distill_model support; main experiments first per guide section 16',
        exports_performed=False))
    if args.smoke:
        init_seeds(0,deterministic=True)
        model=YOLO(str(model_yaml('C3'))).load(str(WEIGHTS))
        options=dict(TRAIN_ARGS,epochs=1,patience=0,close_mosaic=0)
        from worker import assert_finite
        model.add_callback('on_train_batch_end',assert_finite)
        model.train(data=str(DATA),seed=0,project=str(ROOT/'runs/uw_v3_preflight_20260919/train'),
                    name='smoke_C3',**options)
        write_json(ROOT/'uw_v3_smoke.json',dict(status='passed',epochs=1,variant='C3'))

if __name__=='__main__': main()
