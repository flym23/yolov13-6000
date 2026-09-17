"""Aligned Inner-CIoU, following the supplied UW guide (inner-IoU alpha)."""
import math
import torch


def _make_inner_box_xyxy(boxes, ratio=0.7):
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f'ratio must be finite and positive, got {ratio}')
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    half_wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0) * (ratio * 0.5)
    return torch.cat((center - half_wh, center + half_wh), dim=-1)


def box_iou_xyxy(box1, box2, eps=1e-7):
    """Elementwise IoU for aligned (..., 4) boxes, without an NxN expansion."""
    inter = (torch.minimum(box1[..., 2:], box2[..., 2:]) -
             torch.maximum(box1[..., :2], box2[..., :2])).clamp_min(0).prod(-1)
    area1 = (box1[..., 2:] - box1[..., :2]).clamp_min(0).prod(-1)
    area2 = (box2[..., 2:] - box2[..., :2]).clamp_min(0).prod(-1)
    return inter / (area1 + area2 - inter + eps)


def inner_ciou_xyxy(box1, box2, ratio=0.7, eps=1e-7):
    """Use inner IoU and the original boxes' center/aspect penalties.

    Returns (...,) to be explicitly reshaped to (N, 1) in BboxLoss.
    Half precision inputs are promoted for stable area and distance arithmetic.
    """
    if box1.shape != box2.shape or box1.shape[-1] != 4:
        raise ValueError('Inner-CIoU requires matching (..., 4) boxes')
    if box1.dtype in (torch.float16, torch.bfloat16):
        box1 = box1.float()
    if box2.dtype in (torch.float16, torch.bfloat16):
        box2 = box2.float()
    iou = box_iou_xyxy(_make_inner_box_xyxy(box1, ratio), _make_inner_box_xyxy(box2, ratio), eps)
    wh1 = (box1[..., 2:] - box1[..., :2]).clamp_min(eps)
    wh2 = (box2[..., 2:] - box2[..., :2]).clamp_min(eps)
    center1 = (box1[..., :2] + box1[..., 2:]) * 0.5
    center2 = (box2[..., :2] + box2[..., 2:]) * 0.5
    distance = (center1 - center2).square().sum(-1)
    enclosing = torch.maximum(box1[..., 2:], box2[..., 2:]) - torch.minimum(box1[..., :2], box2[..., :2])
    diagonal = enclosing.square().sum(-1) + eps
    v = (4.0 / math.pi ** 2) * (torch.atan(wh2[..., 0] / wh2[..., 1]) -
                                    torch.atan(wh1[..., 0] / wh1[..., 1])).square()
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    return iou - distance / diagonal - alpha * v
