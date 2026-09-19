
import torch
from ultralytics.utils.uw_v3_training import (
    stal_candidate_mask,
    normalized_wasserstein_similarity_xyxy,
    SmallObjectHybridIoUNWDLoss,
    weighted_positive_loss,
    ScoreWeightedFeatureDistillationLoss,
)


def test_stal_exact_for_normal_gt():
    anchors = torch.tensor([[4.,4.],[12.,4.],[4.,12.],[12.,12.],[20.,20.]])
    gt = torch.tensor([[[0.,0.,24.,24.]]])
    mask = torch.ones(1,1,1)
    a = stal_candidate_mask(anchors, gt, mask, enabled=False)
    b = stal_candidate_mask(anchors, gt, mask, enabled=True)
    assert torch.equal(a, b)


def test_stal_expands_tiny_candidate_only_and_keeps_gt():
    anchors = torch.tensor([[4.,4.],[12.,4.],[4.,12.],[12.,12.],[20.,20.]])
    gt = torch.tensor([[[6.,6.,10.,10.]]])
    original = gt.clone()
    mask = torch.ones(1,1,1)

    base = stal_candidate_mask(anchors, gt, mask, enabled=False)
    stal = stal_candidate_mask(anchors, gt, mask, strides=(8,16,32), enabled=True)

    assert base.sum().item() == 0
    assert stal.sum().item() == 4
    assert torch.equal(gt, original)


def test_stal_invalid_padding_never_positive():
    anchors = torch.tensor([[4.,4.],[12.,12.]])
    gt = torch.tensor([[[0.,0.,4.,4.],[0.,0.,4.,4.]]])
    mask = torch.tensor([[[1.],[0.]]])
    out = stal_candidate_mask(anchors, gt, mask, enabled=True)
    assert out[0,1].sum().item() == 0


def test_nwd_identical_and_grad():
    pred = torch.tensor(
        [[10.,10.,20.,20.],[0.,0.,6.,6.]],
        requires_grad=True,
    )
    target = pred.detach().clone()
    sim = normalized_wasserstein_similarity_xyxy(pred, target)
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-4)

    target2 = torch.tensor([[11.,10.,21.,20.],[1.,1.,7.,7.]])
    loss = (1.0 - normalized_wasserstein_similarity_xyxy(pred, target2)).mean()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_small_hybrid_leaves_large_exactly_baseline():
    score = torch.tensor([0.5, 0.5], requires_grad=True)
    pred = torch.tensor([[0.,0.,16.,16.],[0.,0.,64.,64.]], requires_grad=True)
    target = torch.tensor([[1.,1.,17.,17.],[2.,2.,66.,66.]])

    m = SmallObjectHybridIoUNWDLoss(nwd_gain=0.2, small_area_px2=32*32)
    out = m(score, pred, target)

    baseline = 1.0 - score
    assert out.shape == score.shape
    # second box target area = 64^2 and must remain bitwise baseline
    assert torch.equal(out[1:2], baseline[1:2])
    # first small box should contain NWD and differ
    assert not torch.equal(out[0:1], baseline[0:1])

    out.mean().backward()
    assert score.grad is not None
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_nwd_gain_zero_exact_baseline():
    score = torch.randn(10).sigmoid()
    pred = torch.rand(10,4)
    target = torch.rand(10,4)
    # make valid xyxy
    pred = torch.cat((torch.minimum(pred[:,:2], pred[:,2:]),
                      torch.maximum(pred[:,:2], pred[:,2:]) + 1), 1)
    target = torch.cat((torch.minimum(target[:,:2], target[:,2:]),
                        torch.maximum(target[:,:2], target[:,2:]) + 1), 1)

    m = SmallObjectHybridIoUNWDLoss(nwd_gain=0.0)
    out = m(score, pred, target)
    assert torch.equal(out, 1.0-score)


def test_weighted_positive_loss_shapes():
    per = torch.tensor([0.2, 0.4], requires_grad=True)
    w = torch.tensor([[1.0],[0.5]])
    out = weighted_positive_loss(per, w, normalizer=torch.tensor(1.5))
    ref = (0.2*1.0 + 0.4*0.5)/1.5
    assert abs(out.item()-ref) < 1e-6
    out.backward()
    assert per.grad is not None


def test_kd_core_backward_teacher_detached():
    kd = ScoreWeightedFeatureDistillationLoss(
        student_channels=(16,32,64),
        teacher_channels=(32,64,128),
        weight=6.0,
    )
    student = [
        torch.randn(2,16,20,20, requires_grad=True),
        torch.randn(2,32,10,10, requires_grad=True),
        torch.randn(2,64,5,5, requires_grad=True),
    ]
    teacher = [
        torch.randn(2,32,20,20, requires_grad=True),
        torch.randn(2,64,10,10, requires_grad=True),
        torch.randn(2,128,5,5, requires_grad=True),
    ]
    score = [
        torch.rand(2,1,20,20),
        torch.rand(2,1,10,10),
        torch.rand(2,1,5,5),
    ]
    loss = kd(student, teacher, score)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(x.grad is not None for x in student)
    assert all(x.grad is None for x in teacher)
    assert all(p.grad is not None for p in kd.parameters())


def run_all():
    test_stal_exact_for_normal_gt()
    test_stal_expands_tiny_candidate_only_and_keeps_gt()
    test_stal_invalid_padding_never_positive()
    test_nwd_identical_and_grad()
    test_small_hybrid_leaves_large_exactly_baseline()
    test_nwd_gain_zero_exact_baseline()
    test_weighted_positive_loss_shapes()
    test_kd_core_backward_teacher_detached()
    print("ROUND1 PASS")


def test_real_assigner_gt_and_baseline():
    from ultralytics.utils.tal import TaskAlignedAssigner
    anchors = torch.tensor([[x,y] for y in (4.,12.,20.,28.) for x in (4.,12.,20.,28.)])
    gt=torch.tensor([[[6.,6.,10.,10.]]]); saved=gt.clone()
    scores=torch.full((1,16,4),.8); boxes=gt.expand(1,16,4).clone()
    labels=torch.zeros(1,1,1); mask=torch.ones(1,1,1)
    original=TaskAlignedAssigner(topk=10,num_classes=4,alpha=.5)
    expanded=TaskAlignedAssigner(topk=10,num_classes=4,alpha=.5,stal_enabled=True)
    a=original(scores,boxes,anchors,labels,gt,mask)
    b=expanded(scores,boxes,anchors,labels,gt,mask)
    assert a[3].sum()==0 and b[3].sum()==4
    assert torch.equal(gt,saved)
    assert torch.equal(b[1][b[3]],saved[0].expand(4,4))
    ordinary=torch.tensor([[[0.,0.,32.,32.]]])
    a=original(scores,boxes,anchors,labels,ordinary,mask)
    b=expanded(scores,boxes,anchors,labels,ordinary,mask)
    assert all(torch.equal(x,y) for x,y in zip(a,b))


def test_integrated_smallnwd_pixel_units_and_dfl():
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.metrics import bbox_iou
    strides=torch.tensor([[8.],[16.],[32.]])
    target_px=torch.tensor([[[1.,1.,17.,17.],[0.,0.,32.,32.],[0.,0.,64.,64.]]])
    pred_px=(target_px+1).requires_grad_()
    pred=pred_px/strides; target=target_px/strides
    dist=torch.randn(1,3,64,requires_grad=True); anchors=torch.ones(3,2)
    scores=torch.ones(1,3,4)/4; fg=torch.ones(1,3,dtype=torch.bool)
    inputs=(dist,pred,anchors,target,scores,scores.sum(),fg)
    base=BboxLoss()(*inputs,stride_tensor=strides)
    new=BboxLoss(small_nwd_enabled=True)(*inputs,stride_tensor=strides)
    score=bbox_iou(pred[fg],target[fg],xywh=False,CIoU=True)
    expected=SmallObjectHybridIoUNWDLoss()(score,pred_px[fg],target_px[fg]).sum()/3
    torch.testing.assert_close(new[0],expected)
    assert torch.equal(new[1],base[1])
    assert torch.equal(SmallObjectHybridIoUNWDLoss()(score,pred_px[fg],target_px[fg])[1:],(1-score)[1:])
    sum(new).backward()
    assert torch.isfinite(pred_px.grad).all() and torch.isfinite(dist.grad).all()


if __name__ == "__main__":
    run_all()
