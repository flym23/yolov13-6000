import torch
import pytest

from ultralytics.nn.modules import (
    C3k2_UWFEM,
    FEMLite,
    StableBiConcat2,
)
from ultralytics.utils.wiou_v3 import (
    WiseIoUv3Loss,
    weighted_wiou_mean,
)


def test_wiou_state_and_edge_cases():
    loss = WiseIoUv3Loss()
    pred = torch.tensor([[0., 0., 1., 1.], [1., 1., 1., 1.]], requires_grad=True)
    target = torch.tensor([[2., 2., 4., 4.], [1., 1., 1., 1.]])
    out = loss(pred, target)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(pred.grad).all()
    clone = WiseIoUv3Loss()
    clone.load_state_dict(loss.state_dict())
    assert torch.equal(clone.iou_mean, loss.iou_mean)
    clone.eval()
    before = clone.iou_mean.clone()
    clone(pred.detach(), target)
    assert torch.equal(before, clone.iou_mean)
    assert clone(pred[:0], target[:0]).numel() == 0


def test_bbox_dfl_unchanged_and_loss_exclusive():
    from ultralytics.utils.loss import BboxLoss
    with pytest.raises(ValueError, match='mutually exclusive'):
        BboxLoss(use_inner_ciou=True, use_wiou_v3=True)
    pred_dist = torch.randn(1, 2, 64, requires_grad=True)
    boxes = torch.tensor([[[0.,0.,2.,2.],[1.,1.,4.,4.]]], requires_grad=True)
    target = torch.tensor([[[0.,0.,3.,3.],[1.,1.,3.,3.]]])
    anchors = torch.tensor([[1.,1.],[2.,2.]])
    scores = torch.tensor([[[.5,0.,0.,0.],[0.,1.,0.,0.]]])
    fg = torch.ones(1,2,dtype=torch.bool)
    arguments = (pred_dist, boxes, anchors, target, scores, scores.sum(), fg)
    base = BboxLoss()(*arguments)
    wise = BboxLoss(use_wiou_v3=True)(*arguments)
    assert torch.equal(base[1], wise[1])
    sum(wise).backward()
    assert torch.isfinite(pred_dist.grad).all() and torch.isfinite(boxes.grad).all()


def test_c3k2_split_keeps_refinement():
    m = C3k2_UWFEM(64,128,n=1,e=.25).eval()
    x = torch.randn(1,64,20,20)
    with torch.no_grad():
        assert torch.allclose(m(x),m.forward_split(x),atol=1e-6)


def test_femlite_shape_backward():
    x = torch.randn(
        2, 128, 80, 80,
        requires_grad=True,
    )

    m = FEMLite(
        128,
        128,
        expansion=0.25,
        gamma_init=0.05,
    )

    y = m(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_stable_biconcat_exact_baseline_init():
    a = torch.randn(
        2, 64, 40, 40,
        requires_grad=True,
    )

    b = torch.randn(
        2, 96, 40, 40,
        requires_grad=True,
    )

    m = StableBiConcat2(
        dimension=1
    )

    y = m([a, b])

    ref = torch.cat(
        [a, b],
        dim=1,
    )

    assert torch.equal(
        y,
        ref,
    )

    y.square().mean().backward()

    assert m.logits.grad is not None
    assert torch.isfinite(
        m.logits.grad
    ).all()


def test_wiou_identical_box_zero():
    m = WiseIoUv3Loss()

    pred = torch.tensor(
        [[1.0, 2.0, 10.0, 12.0]],
        requires_grad=True,
    )

    target = pred.detach().clone()

    loss = m(
        pred,
        target,
    )

    assert torch.allclose(
        loss,
        torch.zeros_like(loss),
        atol=1e-7,
    )


def test_wiou_backward_and_weighting():
    m = WiseIoUv3Loss()

    pred = torch.tensor(
        [
            [10.0, 10.0, 30.0, 30.0],
            [12.0, 12.0, 29.0, 31.0],
        ],
        requires_grad=True,
    )

    target = torch.tensor(
        [
            [11.0, 11.0, 31.0, 31.0],
            [10.0, 10.0, 30.0, 30.0],
        ]
    )

    per_box = m(
        pred,
        target,
    )

    weight = torch.tensor(
        [
            [1.0],
            [0.5],
        ]
    )

    loss = weighted_wiou_mean(
        per_box,
        weight,
        normalizer=torch.tensor(1.5),
    )

    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(
        pred.grad
    ).all()


def test_c3k2_uwfem_forward():
    # The parser should construct the scaled channel counts.
    m = C3k2_UWFEM(
        c1=64,
        c2=128,
        n=1,
        c3k=False,
        e=0.25,
    )

    x = torch.randn(
        1, 64, 80, 80
    )

    y = m(x)

    assert y.shape == (
        1, 128, 80, 80
    )


def test_trace_inference_modules():
    fem = FEMLite(
        64,
        64,
    ).eval()

    bicat = StableBiConcat2(
        1
    ).eval()

    traced_fem = torch.jit.trace(
        fem,
        torch.randn(
            1, 64, 80, 80
        ),
        strict=True,
    )

    traced_bicat = torch.jit.trace(
        bicat,
        (
            [
                torch.randn(
                    1, 32, 40, 40
                ),
                torch.randn(
                    1, 64, 40, 40
                ),
            ],
        ),
        strict=False,
    )

    y1 = traced_fem(
        torch.randn(
            1, 64, 80, 80
        )
    )

    y2 = traced_bicat(
        [
            torch.randn(
                1, 32, 40, 40
            ),
            torch.randn(
                1, 64, 40, 40
            ),
        ]
    )

    assert y1.shape == (
        1, 64, 80, 80
    )

    assert y2.shape == (
        1, 96, 40, 40
    )
