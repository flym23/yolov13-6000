# Ultralytics 棣冩�?AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import (
    RotatedTaskAlignedAssigner,
    SmallObjectAwareTaskAlignedAssigner,
    TaskAlignedAssigner,
    dist2bbox,
    dist2rbox,
    make_anchors,
)

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


def _clear_ucra_aux_cache():
    """Clear UCRA auxiliary predictions produced during the current forward pass."""
    try:
        from ultralytics.nn.modules.block import _UCRABaseRefine

        cache = getattr(_UCRABaseRefine, "_ucra_aux_forward_cache", None)
        if cache is not None:
            cache.clear()
    except Exception:
        pass


def wasserstein_loss_xyxy(pred, target, eps=1e-7, constant=12.8):
    """Return element-wise normalized Wasserstein loss for aligned ``xyxy`` box pairs."""
    if pred.shape != target.shape or pred.shape[-1] != 4:
        raise ValueError(f"NWD expects matching (..., 4) tensors, got {pred.shape} and {target.shape}.")
    if constant <= 0:
        raise ValueError(f"NWD constant must be positive, got {constant}.")

    pred_center = (pred[..., :2] + pred[..., 2:]) * 0.5
    target_center = (target[..., :2] + target[..., 2:]) * 0.5
    pred_wh = (pred[..., 2:] - pred[..., :2]).clamp_min(eps)
    target_wh = (target[..., 2:] - target[..., :2]).clamp_min(eps)
    center_dist = (pred_center - target_center).pow(2).sum(-1)
    size_dist = (pred_wh - target_wh).pow(2).sum(-1) * 0.25
    distance = torch.sqrt(center_dist + size_dist + eps)
    return (1.0 - torch.exp(-distance / constant)).unsqueeze(-1)



class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        stride_tensor=None,
        imgsz=None,
    ):
        """CIoU loss."""
        weight = target_scores.sum(-1)[fg_mask].reshape(-1, 1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou).reshape(-1, 1) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.sum() * 0.0

        return loss_iou, loss_dfl


class QualityAwareBboxLoss(BboxLoss):
    """CIoU/DFL with NWD applied only to positive targets below a pixel-size threshold."""

    def __init__(self, reg_max=16, nwd_gain=0.0, nwd_small=32.0, nwd_constant=12.8):
        super().__init__(reg_max=reg_max)
        self.nwd_gain = float(nwd_gain)
        self.nwd_small = float(nwd_small)
        self.nwd_constant = float(nwd_constant)
        if self.nwd_gain < 0 or self.nwd_small <= 0 or self.nwd_constant <= 0:
            raise ValueError(
                f"Invalid small-gated NWD settings: gain={self.nwd_gain}, small={self.nwd_small}, "
                f"constant={self.nwd_constant}."
            )

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        stride_tensor=None,
        imgsz=None,
    ):
        loss_iou, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            stride_tensor=stride_tensor,
            imgsz=imgsz,
        )
        if self.nwd_gain == 0:
            return loss_iou, loss_dfl
        if stride_tensor is None:
            raise ValueError("Small-gated NWD requires stride_tensor for grid-to-pixel conversion.")

        pred_pos = pred_bboxes[fg_mask]
        target_pos = target_bboxes[fg_mask]
        stride_map = stride_tensor.reshape(1, -1, 1).expand(fg_mask.shape[0], -1, -1)
        stride_pos = stride_map[fg_mask].to(device=pred_pos.device, dtype=pred_pos.dtype).reshape(-1, 1)
        pred_pos_px = pred_pos * stride_pos
        target_pos_px = target_pos * stride_pos
        target_wh_px = (target_pos_px[..., 2:] - target_pos_px[..., :2]).clamp_min(0)
        target_scale = target_wh_px.prod(-1, keepdim=True).clamp_min(1e-7).sqrt()
        small_mask = (target_scale <= self.nwd_small).to(dtype=pred_pos.dtype)
        if small_mask.any():
            weight = target_scores.sum(-1)[fg_mask].reshape(-1, 1)
            nwd_loss = wasserstein_loss_xyxy(pred_pos_px, target_pos_px, constant=self.nwd_constant)
            loss_iou = loss_iou + self.nwd_gain * (nwd_loss * small_mask * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def varifocal_quality_loss(pred_logits, target_score, positive_mask, alpha=0.75, gamma=2.0):
    """Element-wise Varifocal-style loss for a class-agnostic localization-quality logit."""
    pred_prob = pred_logits.sigmoid()
    positive_mask = positive_mask.to(dtype=pred_logits.dtype)
    weight = alpha * pred_prob.pow(gamma) * (1.0 - positive_mask) + target_score * positive_mask
    return F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="none") * weight


class v8DetectionLoss:
    def __init__(self, model, tal_topk=None):
        device = next(model.parameters()).device
        h = getattr(model, "args", {})

        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.has_quality = hasattr(m, "cvq")
        self.no = m.nc + m.reg_max * 4 + int(self.has_quality)
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        yaml_cfg = model.yaml if hasattr(model, "yaml") and isinstance(model.yaml, dict) else {}
        self.stal_enabled = bool(yaml_cfg.get("stal_enabled", _cfg_get(h, "stal_enabled", False)))
        self.small_obj_px = float(yaml_cfg.get("small_obj_px", _cfg_get(h, "small_obj_px", 0.0)))
        self.small_topk = int(yaml_cfg.get("small_topk", _cfg_get(h, "small_topk", 20)))
        self.center_radius = float(yaml_cfg.get("center_radius", _cfg_get(h, "center_radius", 2.5)))
        assigner_args = {"topk": tal_topk or 10, "num_classes": self.nc, "alpha": 0.5, "beta": 6.0}
        if self.stal_enabled:
            self.assigner = SmallObjectAwareTaskAlignedAssigner(
                **assigner_args,
                small_obj_px=self.small_obj_px,
                small_topk=self.small_topk,
                center_radius=self.center_radius,
            )
        else:
            self.assigner = TaskAlignedAssigner(**assigner_args)

        self.nwd_gain = float(yaml_cfg.get("nwd_gain", _cfg_get(h, "nwd_gain", 0.0)))
        self.nwd_small = float(yaml_cfg.get("nwd_small", _cfg_get(h, "nwd_small", 32.0)))
        self.nwd_constant = float(yaml_cfg.get("nwd_constant", _cfg_get(h, "nwd_constant", 12.8)))
        self.quality_gain = float(yaml_cfg.get("quality_gain", _cfg_get(h, "quality_gain", 0.0)))
        if self.quality_gain < 0:
            raise ValueError(f"quality_gain must be non-negative, got {self.quality_gain}.")
        if self.quality_gain > 0 and not self.has_quality:
            raise ValueError("quality_gain > 0 requires a QDetect head.")
        self.bbox_loss = (
            QualityAwareBboxLoss(
                m.reg_max,
                nwd_gain=self.nwd_gain,
                nwd_small=self.nwd_small,
                nwd_constant=self.nwd_constant,
            )
            if self.nwd_gain > 0
            else BboxLoss(m.reg_max)
        ).to(device)
        self.ucra_aux_gain = 0.0
        if hasattr(model, "yaml") and isinstance(model.yaml, dict):
            self.ucra_aux_gain = float(model.yaml.get("ucra_aux", _cfg_get(h, "ucra_aux", 0.0)))
        else:
            self.ucra_aux_gain = float(_cfg_get(h, "ucra_aux", 0.0))
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def ucra_aux_loss(self, batch, batch_size):
        if self.ucra_aux_gain <= 0:
            _clear_ucra_aux_cache()
            return torch.zeros((), device=self.device)
        from ultralytics.nn.modules.block import _UCRABaseRefine

        cache = getattr(_UCRABaseRefine, "_ucra_aux_forward_cache", None)
        if not cache:
            return torch.zeros((), device=self.device)
        preds = [item["pred"] for item in cache if item["pred"].shape[0] == batch_size]
        _clear_ucra_aux_cache()
        if not preds:
            return torch.zeros((), device=self.device)

        batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
        bboxes = batch["bboxes"].to(self.device)
        cls = batch["cls"].view(-1).long().to(self.device)
        level_losses = []
        for pred in preds:
            pred = pred.float()
            _, _, h, w = pred.shape
            obj_mask = torch.zeros((batch_size, 1, h, w), device=self.device, dtype=pred.dtype)
            weight_map = torch.ones_like(obj_mask)
            if bboxes.numel():
                for bi in range(batch_size):
                    inds = (batch_idx == bi).nonzero(as_tuple=False).flatten()
                    if inds.numel() == 0:
                        continue
                    for idx in inds:
                        x, y, bw, bh = bboxes[idx]
                        cx = x.clamp(0, 1) * (w - 1)
                        cy = y.clamp(0, 1) * (h - 1)
                        sx = (bw.clamp(min=1.0 / max(w, 1)) * w / 2.0).clamp(min=1.0)
                        sy = (bh.clamp(min=1.0 / max(h, 1)) * h / 2.0).clamp(min=1.0)
                        x0 = int(torch.clamp((cx - 3.0 * sx).floor(), 0, w - 1).item())
                        x1 = int(torch.clamp((cx + 3.0 * sx).ceil(), 0, w - 1).item())
                        y0 = int(torch.clamp((cy - 3.0 * sy).floor(), 0, h - 1).item())
                        y1 = int(torch.clamp((cy + 3.0 * sy).ceil(), 0, h - 1).item())
                        yy = torch.arange(y0, y1 + 1, device=self.device, dtype=pred.dtype).view(-1, 1)
                        xx = torch.arange(x0, x1 + 1, device=self.device, dtype=pred.dtype).view(1, -1)
                        gaussian = torch.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
                        obj_mask[bi, 0, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(
                            obj_mask[bi, 0, y0 : y1 + 1, x0 : x1 + 1], gaussian
                        )
                        class_count = max((cls == cls[idx]).sum().item(), 1)
                        class_weight = (len(cls) / class_count) ** 0.5 if len(cls) else 1.0
                        weight_map[bi, 0, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(
                            weight_map[bi, 0, y0 : y1 + 1, x0 : x1 + 1],
                            torch.full_like(gaussian, min(class_weight, 3.0)),
                        )
            bce = F.binary_cross_entropy_with_logits(pred, obj_mask, reduction="none")
            bce_loss = (bce * weight_map).mean()
            pred_prob = pred.sigmoid()
            intersection = (pred_prob * obj_mask).sum(dim=(1, 2, 3))
            dice = 1.0 - (2.0 * intersection + 1.0) / (
                pred_prob.sum(dim=(1, 2, 3)) + obj_mask.sum(dim=(1, 2, 3)) + 1.0
            )
            level_losses.append(bce_loss + dice.mean())
        return torch.stack(level_losses).mean() * self.ucra_aux_gain

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_cat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        if self.has_quality:
            pred_distri, pred_scores, pred_quality = pred_cat.split((self.reg_max * 4, self.nc, 1), 1)
            pred_quality = pred_quality.permute(0, 2, 1).contiguous()
        else:
            pred_distri, pred_scores = pred_cat.split((self.reg_max * 4, self.nc), 1)
            pred_quality = None

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        target_bboxes_grid = target_bboxes / stride_tensor
        if pred_quality is not None and self.quality_gain > 0:
            quality_target = torch.zeros_like(pred_quality)
            if fg_mask.any():
                with torch.no_grad():
                    quality_target[fg_mask] = bbox_iou(
                        pred_bboxes.detach()[fg_mask],
                        target_bboxes_grid[fg_mask],
                        xywh=False,
                        CIoU=False,
                    ).clamp_(0.0, 1.0).to(dtype=quality_target.dtype)
            quality_loss = varifocal_quality_loss(pred_quality, quality_target, fg_mask.unsqueeze(-1))
            loss[1] = loss[1] + self.quality_gain * quality_loss.sum() / target_scores_sum
        if self.ucra_aux_gain > 0:
            aux_loss = self.ucra_aux_loss(batch, batch_size)
        else:
            _clear_ucra_aux_cache()
            aux_loss = torch.zeros((), device=self.device)

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes_grid,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor=stride_tensor,
                imgsz=imgsz,
            )

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[1] += aux_loss
        loss[2] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR segment dataset incorrectly formatted.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask,
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "box", 7.5)
        loss[2] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[3] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def calculate_segmentation_loss(
        fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, overlap
    ):
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        for bi in range(proto.shape[0]):
            if overlaps := (batch_idx == bi).nonzero(as_tuple=False):
                if 0 in overlaps.shape:
                    continue
            else:
                continue
            b_mask = masks[bi].unsqueeze(0)
            b_fg_mask = fg_mask[bi].unsqueeze(1)
            b_target_gt_idx = target_gt_idx[bi].unsqueeze(1)
            b_mxyxy = mxyxy[bi]
            b_marea = marea[bi]
            loss += single_mask_loss(
                b_mask, proto[bi], b_fg_mask, b_target_gt_idx, b_mxyxy, b_marea, pred_masks[bi], overlap=overlap
            )
        return loss / mask_h / mask_w / batch_idx.shape[0]

    @staticmethod
    def single_mask_loss(gt_mask, pred_proto, fg_mask, target_gt_idx, mxyxy, marea, pred_mask, overlap):
        if fg_mask.any():
            loss = F.binary_cross_entropy_with_logits(
                pred_mask[fg_mask], gt_mask[target_gt_idx[fg_mask]], reduction="none"
            )
            if overlap:
                loss = (loss.mean(-1) / marea[target_gt_idx[fg_mask]]).mean()
            else:
                loss = loss.mean()
            return loss
        return 0.0


class v8ClassificationLoss:
    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR OBB dataset incorrectly formatted.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[2] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initializes v8PoseLoss with model, assigner, and keypoint losses."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls, and kpts multiplied by batch size."""
        loss = torch.zeros(5, device=self.device)  # box, kpts, kpts_obj, cls, dfl
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        batch_size = feats[0].shape[0]

        pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor=stride_tensor,
                imgsz=imgsz,
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "pose", 12.0)
        loss[2] *= _cfg_get(self.hyp, "kobj", 1.0)
        loss[3] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[4] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """Calculate the keypoints loss for the model."""
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss


class E2EDetectLoss:
    def __init__(self, model):
        self.one2many = v8DetectionLoss(model)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


# single_mask_loss helper (must be module-level for v8SegmentationLoss to reference)
def single_mask_loss(gt_mask, pred_proto, fg_mask, target_gt_idx, mxyxy, marea, pred_mask, overlap):
    "Compute instance segmentation loss for a single image."
    if fg_mask.any():
        loss = F.binary_cross_entropy_with_logits(
            pred_mask[fg_mask], gt_mask[target_gt_idx[fg_mask]], reduction="none"
        )
        if overlap:
            loss = (loss.mean(-1) / marea[target_gt_idx[fg_mask]]).mean()
        else:
            loss = loss.mean()
        return loss
    return 0.0
