# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import contextlib
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv, RepConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "Detect",
    "QDetect",
    "UDQDetect",
    "RLDHead",
    "SDDCDetect",
    "BRDDetect",
    "EBDRDetect",
    "Segment",
    "Pose",
    "Classify",
    "OBB",
    "RTDETRDecoder",
    "v10Detect",
)


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class _BRDLogitAdapter(nn.Module):
    """Zero-start, zero-common-mode boundary residual for DFL box logits."""

    def __init__(
        self,
        channels,
        out_channels,
        reduction=4,
        detail_kernel=3,
        support_kernel=5,
        max_logit_delta=0.75,
        level_strength=1.0,
        zero_mean_bins=True,
        detach_gate=True,
        finite_fallback=True,
        eps=1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        self.reduction = int(reduction)
        self.detail_kernel = int(detail_kernel)
        self.support_kernel = int(support_kernel)
        self.max_logit_delta = float(max_logit_delta)
        self.level_strength = float(level_strength)
        self.zero_mean_bins = bool(zero_mean_bins)
        self.detach_gate = bool(detach_gate)
        self.finite_fallback = bool(finite_fallback)
        self.eps = float(eps)
        if self.channels <= 0 or self.out_channels <= 0:
            raise ValueError("channels and out_channels must be positive.")
        if self.out_channels % 4:
            raise ValueError(f"out_channels must equal 4*reg_max, got {self.out_channels}.")
        self.reg_max = self.out_channels // 4
        if self.reduction <= 0:
            raise ValueError(f"reduction must be positive, got {self.reduction}.")
        for name, value in (("detail_kernel", self.detail_kernel), ("support_kernel", self.support_kernel)):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value}.")
        if self.max_logit_delta <= 0.0:
            raise ValueError(f"max_logit_delta must be positive, got {self.max_logit_delta}.")
        if not 0.0 <= self.level_strength <= 1.0:
            raise ValueError(f"level_strength must satisfy 0 <= strength <= 1, got {self.level_strength}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

        hidden = max(16, min(96, self.channels // self.reduction))
        self.depthwise = nn.Conv2d(
            self.channels, self.channels, self.detail_kernel, 1, self.detail_kernel // 2,
            groups=self.channels, bias=False,
        )
        self.norm = nn.BatchNorm2d(self.channels)
        self.reduce = nn.Conv2d(self.channels, hidden, 1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.out = nn.Conv2d(hidden, self.out_channels, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _replicate_avg_pool2d(x, kernel_size):
        if kernel_size == 1:
            return x
        pad = kernel_size // 2
        return F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=kernel_size, stride=1, padding=0
        )

    def _detail_gate(self, x):
        x_fp32 = x.float()
        detail = x_fp32 - self._replicate_avg_pool2d(x_fp32, self.detail_kernel)
        magnitude = detail.abs().mean(dim=1, keepdim=True)
        reference = self._replicate_avg_pool2d(magnitude, self.support_kernel)
        ratio = magnitude / reference.add(self.eps)
        salience = (ratio / (1.0 + ratio)).clamp(0.0, 1.0)
        support = self._replicate_avg_pool2d(salience, self.support_kernel)
        gate = (salience * support).clamp(0.0, 1.0).sqrt()
        if self.finite_fallback:
            gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
            detail = torch.nan_to_num(detail, nan=0.0, posinf=0.0, neginf=0.0)
        elif not torch.isfinite(gate).all() or not torch.isfinite(detail).all():
            raise RuntimeError("BRD box-logit gate/detail contains non-finite values.")
        return detail.to(dtype=x.dtype), (gate.detach() if self.detach_gate else gate)

    def _distribution_delta(self, raw):
        delta = torch.tanh(raw.float()).reshape(raw.shape[0], 4, self.reg_max, raw.shape[2], raw.shape[3])
        if self.zero_mean_bins:
            # DFL is invariant to a common shift across bins, so spend capacity on distribution shape only.
            delta = delta - delta.mean(dim=2, keepdim=True)
            delta = delta / delta.abs().amax(dim=2, keepdim=True).clamp_min(1.0)
        return delta.reshape_as(raw)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"BRD adapter expected [B,{self.channels},H,W], got {tuple(x.shape)}.")
        if self.level_strength == 0.0:
            return x.new_zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]))
        detail, gate = self._detail_gate(x)
        hidden = self.act(self.reduce(self.act(self.norm(self.depthwise(detail)))))
        raw = self.out(hidden)
        delta = self.level_strength * self.max_logit_delta * self._distribution_delta(raw) * gate
        if self.finite_fallback:
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        elif not torch.isfinite(delta).all():
            raise RuntimeError("BRD box-logit residual contains non-finite values.")
        return delta.to(dtype=x.dtype)


class BRDDetect(Detect):
    """Detect head with bounded P3/P4 DFL-distribution refinement and unchanged classification towers."""

    _DEFAULT_CONFIG = {
        "reduction": 4,
        "detail_kernel": 3,
        "support_kernel": 5,
        "max_logit_delta": 0.75,
        "level_strengths": [1.0, 0.5, 0.0],
        "zero_mean_bins": True,
        "detach_gate": True,
        "finite_fallback": True,
        "eps": 1e-6,
    }

    def __init__(self, nc=80, config=None, ch=()):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"BRDDetect config must be a dict or None, got {type(config).__name__}.")
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown BRDDetect config keys: {unknown}.")
        cfg = dict(self._DEFAULT_CONFIG)
        cfg.update(config)
        reduction = int(cfg["reduction"])
        detail_kernel = int(cfg["detail_kernel"])
        support_kernel = int(cfg["support_kernel"])
        max_logit_delta = float(cfg["max_logit_delta"])
        eps = float(cfg["eps"])
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        for name, value in (("detail_kernel", detail_kernel), ("support_kernel", support_kernel)):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value}.")
        if max_logit_delta <= 0.0:
            raise ValueError(f"max_logit_delta must be positive, got {max_logit_delta}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("BRDDetect supports the standard one-to-many path only.")
        strengths = [float(value) for value in cfg["level_strengths"]]
        if len(strengths) != self.nl:
            raise ValueError(f"level_strengths must contain {self.nl} values, got {len(strengths)}.")
        if any(value < 0.0 or value > 1.0 for value in strengths):
            raise ValueError(f"Each level strength must be in [0,1], got {strengths}.")

        self.level_strengths = tuple(strengths)
        self.box_refine = nn.ModuleList(
            nn.Identity()
            if strength == 0.0
            else _BRDLogitAdapter(
                channels=channels,
                out_channels=4 * self.reg_max,
                reduction=reduction,
                detail_kernel=detail_kernel,
                support_kernel=support_kernel,
                max_logit_delta=max_logit_delta,
                level_strength=strength,
                zero_mean_bins=bool(cfg["zero_mean_bins"]),
                detach_gate=bool(cfg["detach_gate"]),
                finite_fallback=bool(cfg["finite_fallback"]),
                eps=eps,
            )
            for channels, strength in zip(ch, strengths)
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            raise TypeError(f"BRDDetect expects a {self.nl}-level feature list.")
        outputs = []
        for index, feature in enumerate(x):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise TypeError(f"BRDDetect level {index} must be a 4D tensor.")
            box_logits = self.cv2[index](feature)
            adapter = self.box_refine[index]
            if not isinstance(adapter, nn.Identity):
                box_logits = box_logits + adapter(feature).to(dtype=box_logits.dtype)
            outputs.append(torch.cat((box_logits, self.cv3[index](feature)), dim=1))
        if self.training:
            return outputs
        y = self._inference(outputs)
        return y if self.export else (y, outputs)


class _SDDCTaskAdapter(nn.Module):
    """Zero-start detail or context adapter used ahead of one SDDC detection tower."""

    _VALID_MODES = frozenset(("detail", "context"))

    def __init__(self, channels, mode, gain=1.0, max_residual=0.10, eps=1e-6):
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).lower()
        self.gain = float(gain)
        self.max_residual = float(max_residual)
        self.eps = float(eps)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}.")
        if self.mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}, got {mode!r}.")
        if not 0.0 <= self.gain <= 1.0:
            raise ValueError(f"gain must satisfy 0 <= gain <= 1, got {self.gain}.")
        if not 0.0 < self.max_residual <= 1.0:
            raise ValueError(f"max_residual must satisfy 0 < value <= 1, got {self.max_residual}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

        self.depthwise = nn.Conv2d(self.channels, self.channels, 3, 1, 1, groups=self.channels, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pointwise_groups = math.gcd(self.channels, 4)
        self.pointwise = nn.Conv2d(
            self.channels, self.channels, 1, 1, 0, groups=self.pointwise_groups, bias=True
        )
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    @staticmethod
    def _replicate_avg_pool2d(x, kernel_size):
        pad = kernel_size // 2
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size, stride=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"_SDDCTaskAdapter expects [B,C,H,W] with C={self.channels}, got {tuple(x.shape)}."
            )
        source = x - self._replicate_avg_pool2d(x, 3) if self.mode == "detail" else self._replicate_avg_pool2d(x, 5)
        correction = self.pointwise(self.act(self.depthwise(source)))
        correction_fp32 = correction.float()
        feature_rms = x.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        correction_rms = correction_fp32.square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        rms_scale = torch.minimum(
            torch.ones_like(correction_rms),
            (self.max_residual * feature_rms) / correction_rms.clamp_min(self.eps),
        ).detach()
        return x + self.gain * (correction_fp32 * rms_scale).to(dtype=x.dtype)


class SDDCDetect(Detect):
    """Scale-decoupled detail/context head with exact Detect initialization and interfaces."""

    def __init__(self, nc=80, ch=(), max_residual=0.10):
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("SDDCDetect supports the standard one-to-many detection path only.")
        if len(ch) < 1:
            raise ValueError("SDDCDetect requires at least one pyramid feature.")
        if self.nl == 1:
            detail_gains = context_gains = [1.0]
        else:
            detail_gains = [1.0 - 0.5 * index / (self.nl - 1) for index in range(self.nl)]
            context_gains = [0.5 + 0.5 * index / (self.nl - 1) for index in range(self.nl)]
        self.box_adapters = nn.ModuleList(
            _SDDCTaskAdapter(c, "detail", detail_gains[index], max_residual) for index, c in enumerate(ch)
        )
        self.cls_adapters = nn.ModuleList(
            _SDDCTaskAdapter(c, "context", context_gains[index], max_residual) for index, c in enumerate(ch)
        )

    def forward(self, x):
        if not isinstance(x, list) or len(x) != self.nl:
            raise TypeError(f"SDDCDetect expects a list of {self.nl} feature maps.")
        for index in range(self.nl):
            box_feature = self.box_adapters[index](x[index])
            cls_feature = self.cls_adapters[index](x[index])
            x[index] = torch.cat((self.cv2[index](box_feature), self.cv3[index](cls_feature)), dim=1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class QDetect(Detect):
    """Detect head with an explicit localization-quality branch for score calibration."""

    def __init__(self, nc=80, ch=()):
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("QDetect currently supports the standard one-to-many detection path only.")
        cq = max(16, ch[0] // 4)
        self.cvq = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, cq, 1)),
                nn.Sequential(DWConv(cq, cq, 3), Conv(cq, cq, 1)),
                nn.Conv2d(cq, 1, 1),
            )
            for x in ch
        )
        self.no = self.nc + self.reg_max * 4 + 1

    def forward(self, x):
        """Return raw quality logits in training and quality-calibrated class scores in inference."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cvq[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        """Decode boxes and calibrate each class score by the predicted localization quality."""
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (anchors.transpose(0, 1) for anchors in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls, quality = x_cat.split((self.reg_max * 4, self.nc, 1), 1)
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            scores = cls.sigmoid() * quality.sigmoid().clamp(0.0, 1.0).sqrt()
            return dbox.transpose(1, 2), scores.permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        scores = cls.sigmoid() * quality.sigmoid().clamp(0.0, 1.0).sqrt()
        return torch.cat((dbox, scores), 1)

    def bias_init(self):
        """Initialize box/class biases as Detect does and quality logits to a conservative prior."""
        super().bias_init()
        for quality_head in self.cvq:
            quality_head[-1].bias.data.fill_(-2.0)


class _DistributionGuidedQuality(nn.Module):
    """Predict localization quality from features and detached DFL statistics."""

    def __init__(self, channels, reg_max=16, hidden=None, stat_strength=1.0, detach_stats=True, eps=1e-6):
        super().__init__()
        self.channels = int(channels)
        self.reg_max = int(reg_max)
        self.stat_strength = float(stat_strength)
        self.detach_stats = bool(detach_stats)
        self.eps = float(eps)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}.")
        if self.reg_max <= 1:
            raise ValueError(f"reg_max must be >1, got {self.reg_max}.")
        if self.stat_strength < 0:
            raise ValueError(f"stat_strength must be non-negative, got {self.stat_strength}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")
        hidden = int(hidden or max(16, min(96, self.channels // 4)))
        stat_hidden = max(8, hidden // 2)
        self.feature_path = nn.Sequential(
            DWConv(self.channels, self.channels, 3),
            Conv(self.channels, hidden, 1),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        self.stat_path = nn.Sequential(
            nn.Conv2d(12, stat_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(stat_hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.feature_path[-1].weight)
        nn.init.constant_(self.feature_path[-1].bias, -2.0)
        nn.init.zeros_(self.stat_path[-1].weight)
        nn.init.zeros_(self.stat_path[-1].bias)
        bins = torch.arange(self.reg_max, dtype=torch.float32)
        self.register_buffer("bins", bins.view(1, 1, self.reg_max, 1, 1), persistent=False)

    def _distribution_stats(self, box_logits):
        if not isinstance(box_logits, torch.Tensor) or box_logits.ndim != 4:
            raise TypeError("box_logits must be a 4D tensor.")
        batch, channels, height, width = box_logits.shape
        expected = 4 * self.reg_max
        if channels != expected:
            raise ValueError(f"DFL logits have {channels} channels, expected {expected}.")
        with torch.autocast(device_type=box_logits.device.type, enabled=False):
            logits = box_logits.float().view(batch, 4, self.reg_max, height, width)
            probability = logits.softmax(dim=2)
            entropy = -(probability * probability.clamp_min(self.eps).log()).sum(dim=2) / math.log(float(self.reg_max))
            peak = probability.amax(dim=2)
            mean = (probability * self.bins).sum(dim=2)
            variance = (probability * (self.bins - mean.unsqueeze(2)).square()).sum(dim=2)
            variance = (variance / (((self.reg_max - 1) ** 2) / 4.0)).clamp(0.0, 1.0)
            stats = torch.cat(
                ((1.0 - entropy).clamp(0.0, 1.0), peak.clamp(0.0, 1.0), (1.0 - variance).clamp(0.0, 1.0)), dim=1
            )
            stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return stats.detach() if self.detach_stats else stats

    def forward(self, feature, box_logits):
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise TypeError("feature must be a 4D tensor.")
        if feature.shape[1] != self.channels:
            raise ValueError(f"quality feature has {feature.shape[1]} channels, expected {self.channels}.")
        if feature.shape[0] != box_logits.shape[0] or feature.shape[-2:] != box_logits.shape[-2:]:
            raise ValueError(f"feature/logit shape mismatch: feature={tuple(feature.shape)}, box={tuple(box_logits.shape)}.")
        stats = self._distribution_stats(box_logits).to(device=feature.device, dtype=feature.dtype)
        return self.feature_path(feature) + self.stat_strength * self.stat_path(stats)


class UDQDetect(Detect):
    """Detect head that uses detached DFL uncertainty only for score calibration."""

    _DEFAULT_CONFIG = {
        "quality_mix": 0.50,
        "stat_strengths": [1.0, 0.5, 0.25],
        "detach_stats": True,
        "eps": 1e-6,
    }

    def __init__(self, nc=80, config=None, ch=()):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"UDQDetect config must be dict or None, got {type(config).__name__}.")
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown UDQDetect config keys: {unknown}.")
        cfg = dict(self._DEFAULT_CONFIG)
        cfg.update(config)
        self.quality_mix = float(cfg["quality_mix"])
        self.detach_stats = bool(cfg["detach_stats"])
        self.eps = float(cfg["eps"])
        if not 0.0 <= self.quality_mix <= 1.0:
            raise ValueError(f"quality_mix must be in [0,1], got {self.quality_mix}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("UDQDetect supports the standard one-to-many detection path only.")
        strengths = [float(value) for value in cfg["stat_strengths"]]
        if len(strengths) != self.nl:
            raise ValueError(f"stat_strengths must contain {self.nl} values, got {len(strengths)}.")
        if any(value < 0 for value in strengths):
            raise ValueError(f"stat_strengths must be non-negative, got {strengths}.")
        self.stat_strengths = tuple(strengths)
        with torch.random.fork_rng(devices=[], enabled=True):
            local_seed = (int(torch.initial_seed()) + 7919 * self.nc + 104729 * sum(int(value) for value in ch)) % (2**63 - 1)
            torch.manual_seed(local_seed)
            self.cvq = nn.ModuleList(
                _DistributionGuidedQuality(
                    channels=channels,
                    reg_max=self.reg_max,
                    stat_strength=strength,
                    detach_stats=self.detach_stats,
                    eps=self.eps,
                )
                for channels, strength in zip(ch, self.stat_strengths)
            )
        self.no = self.nc + self.reg_max * 4 + 1

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            raise TypeError(f"UDQDetect expects a {self.nl}-level feature list.")
        outputs = []
        for index, feature in enumerate(x):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise TypeError(f"UDQDetect level {index} must be a 4D tensor.")
            box_logits = self.cv2[index](feature)
            class_logits = self.cv3[index](feature)
            quality_logits = self.cvq[index](feature, box_logits)
            outputs.append(torch.cat((box_logits, class_logits, quality_logits), dim=1))
        if self.training:
            return outputs
        predictions = self._inference(outputs)
        return predictions if self.export else (predictions, outputs)

    def _calibrate_scores(self, class_logits, quality_logits):
        class_scores = class_logits.sigmoid()
        quality = quality_logits.sigmoid().clamp(0.0, 1.0)
        return class_scores * ((1.0 - self.quality_mix) + self.quality_mix * quality)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (anchors.transpose(0, 1) for anchors in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls, quality = x_cat.split((self.reg_max * 4, self.nc, 1), dim=1)
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False)
            return dbox.transpose(1, 2), self._calibrate_scores(cls, quality).permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, self._calibrate_scores(cls, quality)), dim=1)

    def bias_init(self):
        super().bias_init()
        for predictor in self.cvq:
            predictor.feature_path[-1].bias.data.fill_(-2.0)
            predictor.stat_path[-1].bias.data.zero_()


class _RLDBoxBlock(Conv):
    """Conv-compatible box-tower block with a zero-start localization residual."""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        if c1 != c2 or s != 1:
            raise ValueError(f"_RLDBoxBlock expects same-channel stride-1 input, got {c1}->{c2}, stride={s}.")
        self.rep = RepConv(c2, c2, 3, 1, 1, act=False, bn=True)
        self.proj = nn.Conv2d(c2, c2, 1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        out = super().forward(x)
        return out + 0.05 * self.alpha.tanh() * self.proj(self.rep(out))

    def forward_fuse(self, x):
        out = super().forward_fuse(x)
        return out + 0.05 * self.alpha.tanh() * self.proj(self.rep(out))


class RLDHead(Detect):
    """Detection head with re-parameterized localization detail blocks in the bbox tower only."""

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), _RLDBoxBlock(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

class _UGDRLogitAdapter(nn.Module):
    """Zero-start uncertainty-guided DFL-logit residual for one pyramid level."""

    def __init__(
        self,
        channels,
        reg_max,
        reduction=4,
        detail_kernel=3,
        support_kernel=5,
        max_logit_delta=0.75,
        level_strength=1.0,
        uncertainty_power=1.0,
        detail_power=1.0,
        detach_uncertainty=True,
        detach_detail_gate=True,
        finite_fallback=True,
        eps=1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.reg_max = int(reg_max)
        self.out_channels = 4 * self.reg_max
        self.reduction = int(reduction)
        self.detail_kernel = int(detail_kernel)
        self.support_kernel = int(support_kernel)
        self.max_logit_delta = float(max_logit_delta)
        self.level_strength = float(level_strength)
        self.uncertainty_power = float(uncertainty_power)
        self.detail_power = float(detail_power)
        self.detach_uncertainty = bool(detach_uncertainty)
        self.detach_detail_gate = bool(detach_detail_gate)
        self.finite_fallback = bool(finite_fallback)
        self.eps = float(eps)

        if self.channels <= 0 or self.reg_max <= 1:
            raise ValueError(
                f"channels must be positive and reg_max > 1, got {self.channels}, {self.reg_max}."
            )
        if self.reduction <= 0:
            raise ValueError(f"reduction must be positive, got {self.reduction}.")
        for name, value in (
            ("detail_kernel", self.detail_kernel),
            ("support_kernel", self.support_kernel),
        ):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value}.")
        if self.max_logit_delta <= 0.0:
            raise ValueError(
                f"max_logit_delta must be positive, got {self.max_logit_delta}."
            )
        if not 0.0 <= self.level_strength <= 1.0:
            raise ValueError(
                f"level_strength must satisfy 0 <= strength <= 1, got {self.level_strength}."
            )
        if self.uncertainty_power <= 0.0 or self.detail_power <= 0.0:
            raise ValueError(
                "uncertainty_power and detail_power must be positive, got "
                f"{self.uncertainty_power}, {self.detail_power}."
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

        hidden = max(16, min(96, self.channels // self.reduction))
        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            self.detail_kernel,
            1,
            self.detail_kernel // 2,
            groups=self.channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(self.channels)
        self.reduce = nn.Conv2d(self.channels, hidden, 1, 1, 0, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.out = nn.Conv2d(hidden, self.out_channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _replicate_avg_pool2d(x, kernel_size):
        if kernel_size == 1:
            return x
        padding = kernel_size // 2
        return F.avg_pool2d(
            F.pad(x, (padding, padding, padding, padding), mode="replicate"),
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def _distribution_uncertainty(self, box_logits):
        if box_logits.ndim != 4 or box_logits.shape[1] != self.out_channels:
            raise ValueError(
                "box_logits must have shape [B,4*reg_max,H,W], got "
                f"{tuple(box_logits.shape)}."
            )
        shaped = box_logits.float().reshape(
            box_logits.shape[0], 4, self.reg_max, box_logits.shape[2], box_logits.shape[3]
        )
        probability = torch.softmax(shaped, dim=2)
        entropy = -(
            probability * probability.clamp_min(self.eps).log()
        ).sum(dim=2, keepdim=True)
        uncertainty = (entropy / math.log(float(self.reg_max))).clamp(0.0, 1.0)
        uncertainty = uncertainty.pow(self.uncertainty_power)
        if self.finite_fallback:
            uncertainty = torch.where(
                torch.isfinite(uncertainty), uncertainty, torch.zeros_like(uncertainty)
            )
        elif not torch.isfinite(uncertainty).all():
            raise RuntimeError("UGDR localization uncertainty contains non-finite values.")
        return uncertainty.detach() if self.detach_uncertainty else uncertainty

    def _detail_gate(self, feature):
        feature_fp32 = feature.float()
        detail = feature_fp32 - self._replicate_avg_pool2d(
            feature_fp32, self.detail_kernel
        )
        magnitude = detail.abs().mean(dim=1, keepdim=True)
        local_mean = self._replicate_avg_pool2d(magnitude, self.support_kernel)
        ratio = magnitude / local_mean.add(self.eps)
        salience = (ratio / (1.0 + ratio)).clamp(0.0, 1.0)
        local_support = self._replicate_avg_pool2d(salience, self.support_kernel)
        gate = (salience * local_support).clamp(0.0, 1.0).sqrt().pow(
            self.detail_power
        )
        if self.finite_fallback:
            gate = torch.where(torch.isfinite(gate), gate, torch.zeros_like(gate))
            detail = torch.where(torch.isfinite(detail), detail, torch.zeros_like(detail))
        elif not torch.isfinite(gate).all() or not torch.isfinite(detail).all():
            raise RuntimeError("UGDR detail gate contains non-finite values.")
        if self.detach_detail_gate:
            gate = gate.detach()
        return detail.to(dtype=feature.dtype), gate

    def _shape_effective_delta(self, raw):
        delta = torch.tanh(raw.float()).reshape(
            raw.shape[0], 4, self.reg_max, raw.shape[2], raw.shape[3]
        )
        # DFL softmax ignores a common shift across all bins. Remove this null
        # direction so all branch capacity changes the boundary distribution.
        delta = delta - delta.mean(dim=2, keepdim=True)
        # Mean subtraction can enlarge the range; renormalize only when needed.
        max_abs = delta.abs().amax(dim=2, keepdim=True).clamp_min(1.0)
        return delta / max_abs

    def forward(self, feature, box_logits):
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"UGDR adapter expected [B,{self.channels},H,W], got {tuple(feature.shape)}."
            )
        if box_logits.ndim != 4 or box_logits.shape[1] != self.out_channels:
            raise ValueError(
                "UGDR adapter received invalid box-logit shape: "
                f"{tuple(box_logits.shape)}."
            )
        if feature.shape[0] != box_logits.shape[0] or feature.shape[-2:] != box_logits.shape[-2:]:
            raise ValueError(
                "feature and box_logits must share batch and spatial dimensions, got "
                f"{tuple(feature.shape)} and {tuple(box_logits.shape)}."
            )
        if self.level_strength == 0.0:
            return box_logits.new_zeros(box_logits.shape)

        uncertainty = self._distribution_uncertainty(box_logits)
        detail, detail_gate = self._detail_gate(feature)
        hidden = self.act(self.reduce(self.act(self.norm(self.depthwise(detail)))))
        raw = self.out(hidden)
        shaped_delta = self._shape_effective_delta(raw)
        gate = uncertainty * detail_gate.float().unsqueeze(1)
        delta = (
            self.level_strength
            * self.max_logit_delta
            * shaped_delta
            * gate
        )
        delta = delta.reshape_as(box_logits)

        if self.finite_fallback:
            delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        elif not torch.isfinite(delta).all():
            raise RuntimeError("UGDR box-logit residual contains non-finite values.")
        return delta.to(dtype=box_logits.dtype)


def _ebdr_autocast_off(x: torch.Tensor):
    if x.device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=x.device.type, enabled=False)
    return contextlib.nullcontext()


class _EBDRLogitAdapter(nn.Module):
    """Shift-invariant evidential and directional-boundary DFL logit adapter."""

    def __init__(
        self,
        channels,
        out_channels,
        reduction=4,
        detail_kernel=3,
        support_kernel=5,
        evidence_temperature=1.0,
        max_logit_delta=0.35,
        level_strength=1.0,
        use_evidential=True,
        use_entropy=True,
        use_directional_support=True,
        zero_mean_bins=True,
        detach_uncertainty=True,
        detach_support=True,
        finite_fallback=True,
        zero_init=True,
        eps=1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        if self.out_channels % 4 != 0:
            raise ValueError("out_channels must equal 4 * reg_max.")
        self.reg_max = self.out_channels // 4
        self.reduction = int(reduction)
        self.detail_kernel = int(detail_kernel)
        self.support_kernel = int(support_kernel)
        self.evidence_temperature = float(evidence_temperature)
        self.max_logit_delta = float(max_logit_delta)
        self.level_strength = float(level_strength)
        self.use_evidential = bool(use_evidential)
        self.use_entropy = bool(use_entropy)
        self.use_directional_support = bool(use_directional_support)
        self.zero_mean_bins = bool(zero_mean_bins)
        self.detach_uncertainty = bool(detach_uncertainty)
        self.detach_support = bool(detach_support)
        self.finite_fallback = bool(finite_fallback)
        self.zero_init = bool(zero_init)
        self.eps = float(eps)
        if self.channels <= 0 or self.reduction <= 0:
            raise ValueError("channels and reduction must be positive.")
        for name, value in (("detail_kernel", self.detail_kernel), ("support_kernel", self.support_kernel)):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer.")
        if self.evidence_temperature <= 0 or self.max_logit_delta <= 0:
            raise ValueError("evidence_temperature and max_logit_delta must be positive.")
        if not 0 <= self.level_strength <= 1:
            raise ValueError("level_strength must be in [0,1].")
        if not (self.use_evidential or self.use_entropy):
            raise ValueError("At least one uncertainty estimator must be enabled.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        hidden = max(16, min(96, self.channels // self.reduction))
        with torch.random.fork_rng(devices=[], enabled=True):
            local_seed = (
                int(torch.initial_seed())
                + 524287 * self.channels
                + 4099 * self.out_channels
                + 131 * self.detail_kernel
                + int(round(1000 * self.level_strength))
            ) % (2**63 - 1)
            torch.manual_seed(local_seed)
            self.depthwise = nn.Conv2d(
                self.channels, self.channels, self.detail_kernel, 1, self.detail_kernel // 2, groups=self.channels, bias=False
            )
            self.norm = nn.BatchNorm2d(self.channels)
            self.reduce = nn.Conv2d(self.channels, hidden, 1, bias=False)
            self.act = nn.SiLU(inplace=True)
            self.out = nn.Conv2d(hidden, self.out_channels, 1, bias=True)
        if self.zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    @staticmethod
    def _replicate_avg_pool2d(x, kernel_size):
        if kernel_size == 1:
            return x
        p = kernel_size // 2
        return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), kernel_size, stride=1)

    def _uncertainty(self, box_logits):
        b, _, h, w = box_logits.shape
        z = box_logits.float().reshape(b, 4, self.reg_max, h, w)
        centered = (z - z.mean(dim=2, keepdim=True)) / self.evidence_temperature
        signals = []
        if self.use_evidential:
            evidence = (F.softplus(centered) - math.log(2.0)).clamp_min(0.0)
            alpha = evidence + 1.0
            u_evi = self.reg_max / alpha.sum(dim=2)
            signals.append(u_evi.clamp(0.0, 1.0))
        if self.use_entropy:
            p = torch.softmax(centered, dim=2)
            entropy = -(p * p.clamp_min(self.eps).log()).sum(dim=2) / math.log(float(self.reg_max))
            signals.append(entropy.clamp(0.0, 1.0))
        uncertainty = torch.ones_like(signals[0])
        for item in signals:
            uncertainty = uncertainty * item.clamp_min(self.eps)
        uncertainty = uncertainty.pow(1.0 / len(signals)).clamp(0.0, 1.0)
        if self.finite_fallback:
            uncertainty = torch.nan_to_num(uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
        elif not torch.isfinite(uncertainty).all():
            raise RuntimeError("EBDR uncertainty contains non-finite values.")
        return uncertainty.detach() if self.detach_uncertainty else uncertainty

    def _directional_support(self, feature):
        if not self.use_directional_support:
            return feature.new_ones((feature.shape[0], 4, feature.shape[2], feature.shape[3]), dtype=torch.float32)
        energy = feature.float().square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        p = F.pad(energy, (1, 1, 1, 1), mode="replicate")
        gx = 0.5 * (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]).abs()
        gy = 0.5 * (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]).abs()
        gd1 = 0.3535533905932738 * (p[:, :, 2:, 2:] - p[:, :, :-2, :-2]).abs()
        gd2 = 0.3535533905932738 * (p[:, :, 2:, :-2] - p[:, :, :-2, 2:]).abs()
        diagonal = 0.5 * (gd1 + gd2)
        total = gx + gy + diagonal
        ref = self._replicate_avg_pool2d(total, self.support_kernel)
        h_score = (gx + 0.5 * diagonal) / (gx + 0.5 * diagonal + ref + self.eps)
        v_score = (gy + 0.5 * diagonal) / (gy + 0.5 * diagonal + ref + self.eps)
        support = torch.cat((h_score, v_score, h_score, v_score), dim=1).clamp(0.0, 1.0)
        if self.finite_fallback:
            support = torch.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)
        elif not torch.isfinite(support).all():
            raise RuntimeError("EBDR support contains non-finite values.")
        return support.detach() if self.detach_support else support

    def _shape_delta(self, raw):
        b, _, h, w = raw.shape
        delta = torch.tanh(raw.float()).reshape(b, 4, self.reg_max, h, w)
        if self.zero_mean_bins:
            delta = delta - delta.mean(dim=2, keepdim=True)
            delta = delta / delta.abs().amax(dim=2, keepdim=True).clamp_min(1.0)
        return delta

    def forward(self, feature, box_logits):
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(f"Expected feature [B,{self.channels},H,W], got {tuple(feature.shape)}.")
        if box_logits.ndim != 4 or box_logits.shape[1] != self.out_channels:
            raise ValueError(f"Expected box logits [B,{self.out_channels},H,W], got {tuple(box_logits.shape)}.")
        if feature.shape[0] != box_logits.shape[0] or feature.shape[-2:] != box_logits.shape[-2:]:
            raise ValueError("Feature and box-logit shapes are incompatible.")
        if self.level_strength == 0:
            return torch.zeros_like(box_logits)
        uncertainty = self._uncertainty(box_logits)
        support = self._directional_support(feature)
        q = (uncertainty * support).clamp(0.0, 1.0).unsqueeze(2)
        if self.out.weight.dtype == torch.float32:
            with _ebdr_autocast_off(feature):
                hidden = self.act(self.reduce(self.act(self.norm(self.depthwise(feature.float())))))
                raw = self.out(hidden)
        else:
            fd = feature.to(self.depthwise.weight.dtype)
            hidden = self.act(self.reduce(self.act(self.norm(self.depthwise(fd)))))
            raw = self.out(hidden).float()
        delta = self._shape_delta(raw) * q
        delta = delta.reshape_as(box_logits) * self.level_strength * self.max_logit_delta
        if self.finite_fallback:
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        elif not torch.isfinite(delta).all():
            raise RuntimeError("EBDR residual contains non-finite values.")
        return delta.to(box_logits.dtype)


class EBDRDetect(Detect):
    """Detect head with evidential boundary-distribution refinement on selected pyramid levels."""

    _DEFAULT_CONFIG = {
        "reduction": 4,
        "detail_kernel": 3,
        "support_kernel": 5,
        "evidence_temperature": 1.0,
        "max_logit_delta": 0.35,
        "level_strengths": [1.0, 0.5, 0.0],
        "use_evidential": True,
        "use_entropy": True,
        "use_directional_support": True,
        "zero_mean_bins": True,
        "detach_uncertainty": True,
        "detach_support": True,
        "finite_fallback": True,
        "zero_init": True,
        "eps": 1e-6,
    }

    def __init__(self, nc=80, config=None, ch=()):
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise TypeError(f"EBDRDetect config must be dict or None, got {type(config).__name__}.")
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown EBDRDetect config keys: {unknown}.")
        cfg = dict(self._DEFAULT_CONFIG)
        cfg.update(config)
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("EBDRDetect supports the standard one-to-many detection path only.")
        strengths = [float(x) for x in cfg["level_strengths"]]
        if len(strengths) != self.nl:
            raise ValueError(f"level_strengths must contain {self.nl} values, got {len(strengths)}.")
        if any(x < 0.0 or x > 1.0 for x in strengths):
            raise ValueError(f"Each level strength must be in [0,1], got {strengths}.")
        self.level_strengths = tuple(strengths)
        adapter_kwargs = {
            "reduction": int(cfg["reduction"]),
            "detail_kernel": int(cfg["detail_kernel"]),
            "support_kernel": int(cfg["support_kernel"]),
            "evidence_temperature": float(cfg["evidence_temperature"]),
            "max_logit_delta": float(cfg["max_logit_delta"]),
            "use_evidential": bool(cfg["use_evidential"]),
            "use_entropy": bool(cfg["use_entropy"]),
            "use_directional_support": bool(cfg["use_directional_support"]),
            "zero_mean_bins": bool(cfg["zero_mean_bins"]),
            "detach_uncertainty": bool(cfg["detach_uncertainty"]),
            "detach_support": bool(cfg["detach_support"]),
            "finite_fallback": bool(cfg["finite_fallback"]),
            "zero_init": bool(cfg["zero_init"]),
            "eps": float(cfg["eps"]),
        }
        self.box_refine = nn.ModuleList(
            nn.Identity()
            if strength == 0.0
            else _EBDRLogitAdapter(
                channels=channels,
                out_channels=4 * self.reg_max,
                level_strength=strength,
                **adapter_kwargs,
            )
            for channels, strength in zip(ch, strengths)
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            raise TypeError(f"EBDRDetect expects a {self.nl}-level feature list.")
        outputs = []
        for i, feature in enumerate(x):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise TypeError(f"EBDRDetect level {i} must be a 4D tensor.")
            box_logits = self.cv2[i](feature)
            adapter = self.box_refine[i]
            if not isinstance(adapter, nn.Identity):
                box_logits = box_logits + adapter(feature, box_logits)
            cls_logits = self.cv3[i](feature)
            outputs.append(torch.cat((box_logits, cls_logits), dim=1))
        if self.training:
            return outputs
        y = self._inference(outputs)
        return y if self.export else (y, outputs)


