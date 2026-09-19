
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] boxes from xyxy to cxcywh without modifying the input."""
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes last dimension must be 4, got {boxes.shape}")
    xy1 = boxes[..., 0:2]
    xy2 = boxes[..., 2:4]
    wh = (xy2 - xy1).clamp(min=0)
    cxy = (xy1 + xy2) * 0.5
    return torch.cat((cxy, wh), dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] boxes from cxcywh to xyxy without modifying the input."""
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes last dimension must be 4, got {boxes.shape}")
    cxy = boxes[..., 0:2]
    half_wh = boxes[..., 2:4] * 0.5
    return torch.cat((cxy - half_wh, cxy + half_wh), dim=-1)


def stal_candidate_mask(
    anchor_points: torch.Tensor,
    gt_bboxes: torch.Tensor,
    mask_gt: torch.Tensor,
    strides: Sequence[float] = (8.0, 16.0, 32.0),
    enabled: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Small-Target-Aware Label Assignment candidate mask.

    This backports the current Ultralytics STAL candidate-expansion behavior:
    each GT width/height dimension smaller than the minimum detection stride
    is temporarily expanded to the second stride value for candidate selection.

    IMPORTANT:
        - Only the candidate-selection box is expanded.
        - The original gt_bboxes tensor is never modified.
        - Regression must still use the original ground-truth box.

    Args:
        anchor_points:
            Tensor[A, 2] of anchor centers in *pixel coordinates*.
        gt_bboxes:
            Tensor[B, M, 4] in xyxy *pixel coordinates*.
        mask_gt:
            Tensor[B, M, 1], nonzero for valid GT slots.
        strides:
            Detection strides, typically (8, 16, 32) for YOLO11.
        enabled:
            If False, behaves like ordinary center-inside-GT selection.
        eps:
            Strict boundary margin.

    Returns:
        Bool Tensor[B, M, A].
    """
    if anchor_points.ndim != 2 or anchor_points.shape[-1] != 2:
        raise ValueError(f"anchor_points must be [A,2], got {anchor_points.shape}")
    if gt_bboxes.ndim != 3 or gt_bboxes.shape[-1] != 4:
        raise ValueError(f"gt_bboxes must be [B,M,4], got {gt_bboxes.shape}")
    if mask_gt.ndim != 3 or mask_gt.shape[-1] != 1:
        raise ValueError(f"mask_gt must be [B,M,1], got {mask_gt.shape}")
    if gt_bboxes.shape[:2] != mask_gt.shape[:2]:
        raise ValueError("gt_bboxes and mask_gt batch/GT dimensions do not match")
    if len(strides) == 0:
        raise ValueError("strides must not be empty")

    boxes = gt_bboxes

    if enabled:
        stride0 = float(strides[0])
        stride_val = float(strides[1] if len(strides) > 1 else strides[0])
        if stride0 <= 0 or stride_val <= 0:
            raise ValueError(f"strides must be positive, got {strides}")

        xywh = xyxy_to_xywh(gt_bboxes)
        wh = xywh[..., 2:4]
        valid = mask_gt.to(dtype=torch.bool).expand_as(wh)
        small_dim = (wh < stride0) & valid

        replacement = wh.new_tensor(stride_val)
        expanded_wh = torch.where(small_dim, replacement, wh)
        boxes = xywh_to_xyxy(torch.cat((xywh[..., :2], expanded_wh), dim=-1))

    lt = boxes[..., :2].unsqueeze(2)   # [B,M,1,2]
    rb = boxes[..., 2:4].unsqueeze(2)  # [B,M,1,2]
    ap = anchor_points.view(1, 1, -1, 2)

    inside = ((ap - lt > eps) & (rb - ap > eps)).all(dim=-1)
    return inside & mask_gt.to(dtype=torch.bool)


def target_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Area for [...,4] xyxy boxes."""
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes last dimension must be 4, got {boxes.shape}")
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def normalized_wasserstein_similarity_xyxy(
    pred_boxes_px: torch.Tensor,
    target_boxes_px: torch.Tensor,
    constant: float = 12.8,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Normalized Gaussian Wasserstein similarity for aligned xyxy boxes.

    W2^2 = (dcx)^2 + (dcy)^2 + (dw)^2/4 + (dh)^2/4
    NWD  = exp(-sqrt(W2^2 + eps) / constant)

    Args:
        pred_boxes_px, target_boxes_px:
            Aligned boxes [...,4] in PIXEL coordinates.
        constant:
            NWD normalization constant. 12.8 is the common tiny-object default.
    """
    if pred_boxes_px.shape != target_boxes_px.shape:
        raise ValueError(
            f"pred/target shapes differ: {pred_boxes_px.shape} vs {target_boxes_px.shape}"
        )
    if pred_boxes_px.shape[-1] != 4:
        raise ValueError("boxes must end in dimension 4")
    if constant <= 0:
        raise ValueError(f"constant must be >0, got {constant}")

    p = xyxy_to_xywh(pred_boxes_px)
    t = xyxy_to_xywh(target_boxes_px)
    delta = p - t

    wasserstein_sq = (
        delta[..., 0].square()
        + delta[..., 1].square()
        + 0.25 * delta[..., 2].square()
        + 0.25 * delta[..., 3].square()
    )
    distance = torch.sqrt(wasserstein_sq.clamp(min=0) + eps)
    return torch.exp(-distance / float(constant))


class SmallObjectHybridIoUNWDLoss(nn.Module):
    """
    Scale-selective IoU/NWD hybrid for positive boxes.

    Medium/large objects remain EXACTLY on the baseline IoU loss.
    Only targets with pixel area < small_area_px2 receive:
        (1 - nwd_gain) * baseline_iou_loss + nwd_gain * (1 - NWD)

    This module intentionally does NOT compute CIoU itself. Pass the baseline
    per-positive CIoU score produced by Ultralytics `bbox_iou(..., CIoU=True)`.
    That preserves the repository's exact CIoU semantics.

    Input shapes accepted:
        ciou_score: [N] or [N,1]
        boxes:      [N,4]
    Output:
        per-box loss with the same shape as ciou_score.
    """

    def __init__(
        self,
        nwd_gain: float = 0.20,
        small_area_px2: float = 32.0 * 32.0,
        nwd_constant: float = 12.8,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if not (0.0 <= nwd_gain <= 1.0):
            raise ValueError(f"nwd_gain must be in [0,1], got {nwd_gain}")
        if small_area_px2 <= 0:
            raise ValueError("small_area_px2 must be positive")
        if nwd_constant <= 0:
            raise ValueError("nwd_constant must be positive")
        self.nwd_gain = float(nwd_gain)
        self.small_area_px2 = float(small_area_px2)
        self.nwd_constant = float(nwd_constant)
        self.eps = float(eps)

    def forward(
        self,
        ciou_score: torch.Tensor,
        pred_boxes_px: torch.Tensor,
        target_boxes_px: torch.Tensor,
    ) -> torch.Tensor:
        if pred_boxes_px.ndim != 2 or pred_boxes_px.shape[-1] != 4:
            raise ValueError(f"pred_boxes_px must be [N,4], got {pred_boxes_px.shape}")
        if target_boxes_px.shape != pred_boxes_px.shape:
            raise ValueError("pred_boxes_px and target_boxes_px must have equal shape")

        original_shape = ciou_score.shape
        score = ciou_score.reshape(-1)
        if score.numel() != pred_boxes_px.shape[0]:
            raise ValueError(
                f"ciou_score has {score.numel()} values but boxes have {pred_boxes_px.shape[0]}"
            )

        ciou_loss = 1.0 - score
        if self.nwd_gain == 0.0 or score.numel() == 0:
            return ciou_loss.reshape(original_shape)

        nwd = normalized_wasserstein_similarity_xyxy(
            pred_boxes_px,
            target_boxes_px,
            constant=self.nwd_constant,
            eps=self.eps,
        )
        nwd_loss = 1.0 - nwd

        small_mask = target_area_xyxy(target_boxes_px) < self.small_area_px2
        hybrid_small = (
            (1.0 - self.nwd_gain) * ciou_loss
            + self.nwd_gain * nwd_loss
        )
        out = torch.where(small_mask, hybrid_small, ciou_loss)
        return out.reshape(original_shape)


def weighted_positive_loss(
    per_box_loss: torch.Tensor,
    target_score_weight: torch.Tensor,
    normalizer: torch.Tensor | float,
) -> torch.Tensor:
    """Ultralytics-compatible weighted reduction for foreground box loss."""
    loss = per_box_loss
    weight = target_score_weight

    if loss.ndim == weight.ndim - 1:
        loss = loss.unsqueeze(-1)
    if weight.ndim == loss.ndim - 1:
        weight = weight.unsqueeze(-1)
    if loss.shape != weight.shape:
        try:
            loss = loss.expand_as(weight)
        except RuntimeError as exc:
            raise ValueError(f"cannot broadcast loss {loss.shape} to weight {weight.shape}") from exc

    if isinstance(normalizer, torch.Tensor):
        denom = normalizer.clamp(min=torch.finfo(loss.dtype).eps)
    else:
        denom = max(float(normalizer), torch.finfo(loss.dtype).eps)

    return (loss * weight).sum() / denom


class FeatureHook:
    """Forward hook storing the latest tensor output from a selected layer."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def __call__(self, module: nn.Module, inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"distillation hook expects Tensor output, got {type(output).__name__}"
            )
        self.value = output


class ScoreWeightedFeatureDistillationLoss(nn.Module):
    """
    Standalone score-weighted feature distillation core.

    This mirrors the current Ultralytics distillation design at the loss level:
    student neck features are projected to teacher channels with two 1x1 convs,
    then compared by teacher-score-weighted MSE.

    This module is TRAINING ONLY and must not be kept in the exported student.

    Args:
        student_channels:
            channels of P3/P4/P5 student neck features.
        teacher_channels:
            channels of corresponding teacher neck features.
        weight:
            global distillation loss multiplier.
    """

    def __init__(
        self,
        student_channels: Sequence[int],
        teacher_channels: Sequence[int],
        weight: float = 6.0,
    ) -> None:
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels/teacher_channels length mismatch")
        if len(student_channels) == 0:
            raise ValueError("at least one feature level is required")
        if weight < 0:
            raise ValueError("weight must be >=0")

        self.weight = float(weight)
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(sc), int(tc), kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(tc), int(tc), kernel_size=1, stride=1, padding=0),
                )
                for sc, tc in zip(student_channels, teacher_channels)
            ]
        )

    @staticmethod
    def _one_level(
        projected_student: torch.Tensor,
        teacher: torch.Tensor,
        teacher_score: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        if projected_student.shape != teacher.shape:
            raise ValueError(
                f"projected student/teacher mismatch: {projected_student.shape} vs {teacher.shape}"
            )
        if teacher_score.ndim != 4 or teacher_score.shape[1] != 1:
            raise ValueError(f"teacher_score must be [N,1,H,W], got {teacher_score.shape}")
        if teacher_score.shape[0] != teacher.shape[0] or teacher_score.shape[-2:] != teacher.shape[-2:]:
            raise ValueError("teacher_score spatial/batch shape mismatch")

        # Teacher must be a fixed target.
        target = teacher.detach()
        score = teacher_score.detach().clamp(min=0)

        mse = (projected_student - target).square()
        weighted_sum = (mse * score).sum()
        denom = score.sum() * teacher.shape[1] + eps
        return weighted_sum / denom

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
        teacher_scores: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        n = len(self.projectors)
        if not (len(student_features) == len(teacher_features) == len(teacher_scores) == n):
            raise ValueError("feature/score level count mismatch")

        loss = student_features[0].new_zeros(())
        for projector, sf, tf, ts in zip(
            self.projectors, student_features, teacher_features, teacher_scores
        ):
            projected = projector(sf)
            loss = loss + self._one_level(projected, tf, ts)

        return loss * self.weight
