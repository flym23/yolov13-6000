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


class UGDRDetect(Detect):
    """
    Uncertainty-Guided Distribution Refinement detection head.

    The original classification towers are untouched. The original DFL box
    logits receive a zero-start residual only where the current boundary
    distribution is uncertain and the input feature has locally supported
    detail. The residual has zero mean over each edge's DFL bins and a strict
    per-logit magnitude bound. Loss, assigner, DFL decoder and NMS interfaces
    remain identical to Detect.

    Required existing symbols in head.py:
        math, torch, torch.nn as nn, torch.nn.functional as F, Detect
    """

    _DEFAULT_CONFIG = {
        "reduction": 4,
        "detail_kernel": 3,
        "support_kernel": 5,
        "max_logit_delta": 0.75,
        "level_strengths": [1.0, 0.5, 0.0],
        "uncertainty_power": 1.0,
        "detail_power": 1.0,
        "detach_uncertainty": True,
        "detach_detail_gate": True,
        "finite_fallback": True,
        "eps": 1e-6,
    }

    def __init__(self, nc=80, config=None, ch=()):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(
                f"UGDRDetect config must be a dict or None, got {type(config).__name__}."
            )
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown UGDRDetect config keys: {unknown}.")

        cfg = dict(self._DEFAULT_CONFIG)
        cfg.update(config)

        reduction = int(cfg["reduction"])
        detail_kernel = int(cfg["detail_kernel"])
        support_kernel = int(cfg["support_kernel"])
        max_logit_delta = float(cfg["max_logit_delta"])
        uncertainty_power = float(cfg["uncertainty_power"])
        detail_power = float(cfg["detail_power"])
        eps = float(cfg["eps"])

        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        for name, value in (
            ("detail_kernel", detail_kernel),
            ("support_kernel", support_kernel),
        ):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value}.")
        if max_logit_delta <= 0.0:
            raise ValueError(
                f"max_logit_delta must be positive, got {max_logit_delta}."
            )
        if uncertainty_power <= 0.0 or detail_power <= 0.0:
            raise ValueError(
                "uncertainty_power and detail_power must be positive, got "
                f"{uncertainty_power}, {detail_power}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError(
                "UGDRDetect currently supports only the standard one-to-many path."
            )

        strengths = tuple(float(value) for value in cfg["level_strengths"])
        if len(strengths) != self.nl:
            raise ValueError(
                f"level_strengths must contain {self.nl} values, got {len(strengths)}."
            )
        if any(value < 0.0 or value > 1.0 for value in strengths):
            raise ValueError(
                f"Every level strength must be in [0,1], got {strengths}."
            )

        self.level_strengths = strengths
        adapters = []
        for channels, strength in zip(ch, strengths):
            if strength == 0.0:
                adapters.append(nn.Identity())
            else:
                adapters.append(
                    _UGDRLogitAdapter(
                        channels=channels,
                        reg_max=self.reg_max,
                        reduction=reduction,
                        detail_kernel=detail_kernel,
                        support_kernel=support_kernel,
                        max_logit_delta=max_logit_delta,
                        level_strength=strength,
                        uncertainty_power=uncertainty_power,
                        detail_power=detail_power,
                        detach_uncertainty=bool(cfg["detach_uncertainty"]),
                        detach_detail_gate=bool(cfg["detach_detail_gate"]),
                        finite_fallback=bool(cfg["finite_fallback"]),
                        eps=eps,
                    )
                )
        self.box_refine = nn.ModuleList(adapters)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            raise TypeError(f"UGDRDetect expects a {self.nl}-level feature list.")

        # Build a fresh output list instead of mutating the input feature list,
        # preserving Detect-compatible training/inference outputs without hidden
        # side effects on upstream feature references.
        outputs = []
        for index in range(self.nl):
            feature = x[index]
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise TypeError(f"UGDRDetect level {index} must be a 4D tensor.")
            box_logits = self.cv2[index](feature)
            adapter = self.box_refine[index]
            if not isinstance(adapter, nn.Identity):
                box_logits = box_logits + adapter(feature, box_logits)
            cls_logits = self.cv3[index](feature)
            outputs.append(torch.cat((box_logits, cls_logits), dim=1))

        if self.training:
            return outputs
        y = self._inference(outputs)
        return y if self.export else (y, outputs)
