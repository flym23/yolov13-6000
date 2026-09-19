from __future__ import annotations

import torch
import torch.nn as nn


class WiseIoUv3Loss(nn.Module):
    """
    Wise-IoU v3 style loss for ALIGNED bounding boxes in XYXY format.

    For each positive pair:

        L_iou = 1 - IoU

        R_wiou =
            exp(
                center_distance^2
                /
                enclosing_diagonal^2.detach()
            )

        beta =
            L_iou.detach()
            /
            running_mean(L_iou)

        focus =
            beta
            /
            (
                delta
                *
                alpha ** (beta - delta)
            )

        L_wiou_v3 =
            focus
            *
            R_wiou
            *
            L_iou

    Defaults follow the official Wise-IoU v2 implementation:
        momentum = 1e-2
        alpha = 1.7
        delta = 2.7

    This is a TRAINING-ONLY loss. It must not alter the inference graph.
    """

    def __init__(
        self,
        momentum: float = 1e-2,
        alpha: float = 1.7,
        delta: float = 2.7,
        eps: float = 1e-7,
    ) -> None:

        super().__init__()

        if not (0.0 < momentum <= 1.0):
            raise ValueError(
                "momentum must be in (0, 1], "
                f"got {momentum}"
            )

        if (
            alpha <= 0.0
            or delta <= 0.0
            or eps <= 0.0
        ):
            raise ValueError(
                "alpha, delta and eps must be positive"
            )

        self.momentum = float(momentum)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.eps = float(eps)

        self.register_buffer(
            "iou_mean",
            torch.tensor(
                1.0,
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _validate_boxes(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:

        if pred.shape != target.shape:
            raise ValueError(
                "pred/target shape mismatch: "
                f"{pred.shape} vs {target.shape}"
            )

        if pred.shape[-1] != 4:
            raise ValueError(
                "last dimension must be 4, "
                f"got {pred.shape[-1]}"
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_boxes(
            pred,
            target,
        )

        px1, py1, px2, py2 = pred.unbind(
            dim=-1
        )

        tx1, ty1, tx2, ty2 = target.unbind(
            dim=-1
        )

        pw = (px2 - px1).clamp(
            min=self.eps
        )

        ph = (py2 - py1).clamp(
            min=self.eps
        )

        tw = (tx2 - tx1).clamp(
            min=self.eps
        )

        th = (ty2 - ty1).clamp(
            min=self.eps
        )

        ix1 = torch.maximum(
            px1,
            tx1,
        )

        iy1 = torch.maximum(
            py1,
            ty1,
        )

        ix2 = torch.minimum(
            px2,
            tx2,
        )

        iy2 = torch.minimum(
            py2,
            ty2,
        )

        iw = (ix2 - ix1).clamp(
            min=0.0
        )

        ih = (iy2 - iy1).clamp(
            min=0.0
        )

        inter = iw * ih

        union = (
            pw * ph
            +
            tw * th
            -
            inter
        )

        iou = inter / (
            union + self.eps
        )

        iou_loss = 1.0 - iou

        pcx = (
            px1 + px2
        ) * 0.5

        pcy = (
            py1 + py2
        ) * 0.5

        tcx = (
            tx1 + tx2
        ) * 0.5

        tcy = (
            ty1 + ty2
        ) * 0.5

        center_distance2 = (
            (pcx - tcx).square()
            +
            (pcy - tcy).square()
        )

        ex1 = torch.minimum(
            px1,
            tx1,
        )

        ey1 = torch.minimum(
            py1,
            ty1,
        )

        ex2 = torch.maximum(
            px2,
            tx2,
        )

        ey2 = torch.maximum(
            py2,
            ty2,
        )

        enclosing_diagonal2 = (
            (ex2 - ex1).square()
            +
            (ey2 - ey1).square()
        ).clamp(
            min=self.eps
        )

        distance_attention = torch.exp(
            center_distance2
            /
            enclosing_diagonal2.detach()
        )

        base_loss = (
            distance_attention
            *
            iou_loss
        )

        if (
            self.training
            and iou_loss.numel() > 0
        ):
            with torch.no_grad():

                batch_mean = (
                    iou_loss
                    .detach()
                    .float()
                    .mean()
                )

                self.iou_mean.mul_(
                    1.0 - self.momentum
                )

                self.iou_mean.add_(
                    self.momentum
                    *
                    batch_mean
                )

        beta = (
            iou_loss
            .detach()
            .float()
            /
            self.iou_mean.clamp(
                min=self.eps
            )
        )

        alpha_tensor = torch.as_tensor(
            self.alpha,
            device=beta.device,
            dtype=beta.dtype,
        )

        divisor = (
            self.delta
            *
            torch.pow(
                alpha_tensor,
                beta - self.delta,
            )
        )

        focus = (
            beta
            /
            divisor.clamp(
                min=self.eps
            )
        )

        return (
            focus.to(
                dtype=base_loss.dtype
            )
            *
            base_loss
        )


def weighted_wiou_mean(
    per_box_loss: torch.Tensor,
    weight: torch.Tensor,
    normalizer: torch.Tensor | float,
) -> torch.Tensor:
    """
    Apply Ultralytics target-score weighting to per-positive WIoU values.
    """

    if (
        per_box_loss.ndim
        ==
        weight.ndim - 1
    ):
        per_box_loss = (
            per_box_loss.unsqueeze(-1)
        )

    if per_box_loss.shape != weight.shape:

        try:
            per_box_loss = (
                per_box_loss.expand_as(
                    weight
                )
            )

        except RuntimeError as exc:

            raise ValueError(
                "WIoU loss shape "
                f"{per_box_loss.shape} "
                "cannot match weight "
                f"{weight.shape}"
            ) from exc

    return (
        per_box_loss
        *
        weight
    ).sum() / normalizer
