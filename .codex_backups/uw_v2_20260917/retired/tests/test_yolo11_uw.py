"""UW regressions: attention, aligned weighted loss, DFL preservation and model graphs."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from uw_common import WEIGHTS, model_yaml, runtime, write_json
runtime()

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import ECA
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.inner_iou import inner_ciou_xyxy
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.torch_utils import get_flops, init_seeds

torch.set_num_threads(2)


class TestECA(unittest.TestCase):
    def test_shape_gradient_and_invalid_kernel(self):
        x = torch.randn(2, 32, 80, 80, requires_grad=True)
        m = ECA(32)
        y = m(x)
        self.assertEqual(x.shape, y.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(m.conv.weight.grad.abs().sum().item(), 0)
        for k in (0, -1, 2):
            with self.assertRaises(ValueError):
                ECA(32, k)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_autocast(self):
        x = torch.randn(2, 32, 80, 80, device='cuda', requires_grad=True)
        m = ECA(32).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            y = m(x)
        y.mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class TestInnerCIoU(unittest.TestCase):
    def test_identical_degenerate_and_random(self):
        torch.manual_seed(4)
        a = torch.rand(1000, 4)
        a[:, 2:] += a[:, :2] + 0.01
        b = torch.rand(1000, 4)
        b[:, 2:] += b[:, :2] + 0.01
        self.assertTrue(torch.allclose(inner_ciou_xyxy(a, a), torch.ones(1000), atol=0.003))
        expected = bbox_iou(a, b, xywh=False, CIoU=True).squeeze(-1)
        self.assertTrue(torch.allclose(inner_ciou_xyxy(a, b, ratio=1), expected, atol=1e-5))
        for ratio in (0.5, 0.7, 0.9):
            pred = torch.cat((a, torch.zeros(1, 4))).requires_grad_()
            score = inner_ciou_xyxy(pred, torch.cat((b, torch.zeros(1, 4))), ratio)
            score.sum().backward()
            self.assertTrue(torch.isfinite(score).all())
            self.assertTrue(torch.isfinite(pred.grad).all())
        for ratio in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                inner_ciou_xyxy(a, b, ratio)

    def test_weighted_loss_and_unchanged_dfl(self):
        pred = torch.tensor([[[0., 0., 3., 3.], [1., 1., 5., 5.]]], requires_grad=True)
        target = torch.tensor([[[0.2, 0.1, 3.3, 2.7], [1.4, 1.2, 4., 4.8]]])
        dist = torch.randn(1, 2, 64, requires_grad=True)
        anchors = torch.tensor([[1., 1.], [2., 2.]])
        scores = torch.tensor([[[0.2], [0.8]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        args = (dist, pred, anchors, target, scores, scores.sum(), mask)
        base_iou, base_dfl = BboxLoss()( *args )
        inner_loss, inner_dfl = BboxLoss(use_inner_ciou=True)( *args )
        expected = ((1 - inner_ciou_xyxy(pred[mask], target[mask])) * torch.tensor([0.2, 0.8])).sum()
        self.assertTrue(torch.allclose(inner_loss, expected))
        self.assertTrue(torch.equal(base_dfl, inner_dfl))
        (inner_loss + inner_dfl).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(torch.isfinite(dist.grad).all())
        self.assertNotEqual(float(base_iou), float(inner_loss))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_half_precision_large_boxes(self):
        a = torch.tensor([[10., 10., 630., 630.]], device='cuda', dtype=torch.float16, requires_grad=True)
        b = a.detach() + 1
        with torch.autocast('cuda', dtype=torch.float16):
            value = inner_ciou_xyxy(a, b)
        value.sum().backward()
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.isfinite(a.grad).all())


class TestModels(unittest.TestCase):
    def test_all_ablations_and_full_backward(self):
        report = {}
        for variant, expected_strides, eca_count, inner in (
                ('A0', [8, 16, 32], 0, False), ('A1', [4, 8, 16, 32], 0, False),
                ('A2', [8, 16, 32], 1, False), ('A3', [8, 16, 32], 0, True),
                ('A4', [4, 8, 16, 32], 2, True)):
            with self.subTest(variant=variant):
                init_seeds(0, deterministic=True)
                yolo = YOLO(str(model_yaml(variant))).load(str(WEIGHTS))
                net = yolo.model
                self.assertEqual(net.stride.tolist(), expected_strides)
                self.assertEqual(sum(isinstance(m, ECA) for m in net.modules()), eca_count)
                net.args = SimpleNamespace(**DEFAULT_CFG_DICT)
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                net.to(device).train()
                size = 640 if variant == 'A4' else 128
                batch = dict(img=torch.rand(2, 3, size, size, device=device),
                    batch_idx=torch.tensor([0., 1.], device=device),
                    cls=torch.tensor([[0.], [1.]], device=device),
                    bboxes=torch.tensor([[.5, .5, .1, .1], [.4, .4, .2, .2]], device=device))
                loss, items = net.loss(batch)
                self.assertEqual(net.criterion.bbox_loss.use_inner_ciou, inner)
                self.assertEqual(type(net.criterion.assigner).__name__, 'TaskAlignedAssigner')
                self.assertTrue(torch.isfinite(loss).all())
                loss.sum().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
                for module in net.modules():
                    if isinstance(module, ECA):
                        self.assertIsNotNone(module.conv.weight.grad)
                        self.assertGreater(module.conv.weight.grad.abs().sum().item(), 0)
                net.eval()
                with torch.no_grad():
                    output = net(torch.zeros(1, 3, 640, 640, device=device))
                self.assertEqual(output[0].shape[-1], 34000 if variant in ('A1', 'A4') else 8400)
                report[variant] = dict(stride=expected_strides, eca_count=eca_count, inner_ciou=inner,
                    params=sum(p.numel() for p in net.parameters()), GFLOPs=get_flops(net, imgsz=640),
                    loss=[float(v) for v in items], forward_640=True, backward=True)
                net.info()
                del yolo, net, output, loss, items, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        write_json(ROOT / 'uw_preflight_results.json', report)


class TestEvaluation(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_worker_evaluation_artifacts(self):
        import uw_worker
        from PIL import Image
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            images, labels = folder / 'images/test', folder / 'labels/test'
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for i in range(2):
                Image.new('RGB', (640, 480), (50, 70, 90)).save(images / f'{i}.jpg')
                (labels / f'{i}.txt').write_text('0 0.5 0.5 0.05 0.05\n', encoding='utf-8')
            config = folder / 'data.yaml'
            config.write_text(yaml.safe_dump(dict(path=str(folder), train='images/test', val='images/test',
                test='images/test', names={i: str(i) for i in range(4)})), encoding='utf-8')
            root = folder / 'run'
            weights = root / 'train/seed0/weights/best.pt'
            weights.parent.mkdir(parents=True)
            YOLO(str(model_yaml('A0'))).save(str(weights))
            args = SimpleNamespace(run_root=root, variant='A0', seed=0, resume=False)
            with patch.object(uw_worker, 'DATA', config):
                uw_worker.evaluate(args)
            from uw_common import read_json
            summary = read_json(root / 'test/seed0/summary_metrics.json')
            self.assertEqual(summary['seed'], 0)
            self.assertGreater(summary['metrics']['params'], 0)
            self.assertGreater(summary['metrics']['GFLOPs'], 0)
            self.assertGreater(summary['metrics']['latency_ms_batch1_fp32'], 0)
            scale = read_json(root / 'test/seed0/scale_ap_metrics.json')
            self.assertIsNone(scale['metrics']['APM'])
            self.assertIsNone(scale['metrics']['APL'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
