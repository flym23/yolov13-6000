class BGDRP3Fuse(nn.Module):
    """
    Background-Guided Detail Rescue from backbone P2 to detection P3.

    The module anti-aliases P2 to P3 resolution, extracts local detail, and
    injects only detail that is simultaneously:
      1) semantically compatible with P3,
      2) locally salient,
      3) spatially coherent rather than an isolated underwater-noise spike.

    The final projection is zero-initialized and the projected correction is
    spatially centered and RMS-bounded per sample/channel, so the module is an
    exact P3 identity at initialization.

    Required existing symbols in block.py:
        torch, torch.nn as nn, torch.nn.functional as F
    """

    _DEFAULT_CONFIG = {
        "reduction": 4,
        "detail_kernel": 3,
        "support_kernel": 5,
        "agreement_power": 1.0,
        "support_power": 1.0,
        "coherence_power": 1.0,
        "coherence_floor": 0.25,
        "max_residual_ratio": 0.10,
        "detail_grad_clip": 1.0,
        "center_correction": True,
        "detach_gate": False,
        "detach_bound": True,
        "strict_scale": True,
        "finite_fallback": True,
        "eps": 1e-6,
    }

    def __init__(self, c_p2, c_p3, config=None):
        super().__init__()
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(
                f"BGDRP3Fuse config must be a dict or None, got {type(config).__name__}."
            )
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown BGDRP3Fuse config keys: {unknown}.")

        cfg = dict(self._DEFAULT_CONFIG)
        cfg.update(config)

        self.c_p2 = int(c_p2)
        self.c_p3 = int(c_p3)
        self.reduction = int(cfg["reduction"])
        self.detail_kernel = int(cfg["detail_kernel"])
        self.support_kernel = int(cfg["support_kernel"])
        self.agreement_power = float(cfg["agreement_power"])
        self.support_power = float(cfg["support_power"])
        self.coherence_power = float(cfg["coherence_power"])
        self.coherence_floor = float(cfg["coherence_floor"])
        self.max_residual_ratio = float(cfg["max_residual_ratio"])
        self.detail_grad_clip = float(cfg["detail_grad_clip"])
        self.center_correction = bool(cfg["center_correction"])
        self.detach_gate = bool(cfg["detach_gate"])
        self.detach_bound = bool(cfg["detach_bound"])
        self.strict_scale = bool(cfg["strict_scale"])
        self.finite_fallback = bool(cfg["finite_fallback"])
        self.eps = float(cfg["eps"])

        if self.c_p2 <= 0 or self.c_p3 <= 0:
            raise ValueError(
                f"c_p2 and c_p3 must be positive, got {self.c_p2}, {self.c_p3}."
            )
        if self.reduction <= 0:
            raise ValueError(f"reduction must be positive, got {self.reduction}.")
        for name, value in (
            ("detail_kernel", self.detail_kernel),
            ("support_kernel", self.support_kernel),
        ):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value}.")
        for name, value in (
            ("agreement_power", self.agreement_power),
            ("support_power", self.support_power),
            ("coherence_power", self.coherence_power),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if not 0.0 <= self.coherence_floor <= 1.0:
            raise ValueError(
                "coherence_floor must satisfy 0 <= floor <= 1, "
                f"got {self.coherence_floor}."
            )
        if not 0.0 < self.max_residual_ratio <= 1.0:
            raise ValueError(
                "max_residual_ratio must satisfy 0 < ratio <= 1, "
                f"got {self.max_residual_ratio}."
            )
        if not 0.0 < self.detail_grad_clip < float("inf"):
            raise ValueError(
                "detail_grad_clip must be a finite positive value, "
                f"got {self.detail_grad_clip}."
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

        hidden = max(16, min(96, self.c_p3 // self.reduction))
        self.hidden = hidden

        kernel = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        kernel /= kernel.sum()
        self.register_buffer("aa_kernel", kernel[None, None], persistent=False)

        with torch.random.fork_rng(devices=[], enabled=True):
            local_seed = (
                int(torch.initial_seed())
                + 65537 * self.c_p2
                + 8191 * self.c_p3
                + 257 * self.detail_kernel
                + 17 * self.support_kernel
            ) % (2**63 - 1)
            torch.manual_seed(local_seed)

            self.p2_proj = nn.Sequential(
                nn.Conv2d(self.c_p2, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            self.p3_proj = nn.Sequential(
                nn.Conv2d(self.c_p3, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            self.detail_refine = nn.Sequential(
                nn.Conv2d(
                    hidden,
                    hidden,
                    self.detail_kernel,
                    1,
                    self.detail_kernel // 2,
                    groups=hidden,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            self.detail_out = nn.Conv2d(hidden, self.c_p3, 1, 1, 0, bias=False)

        nn.init.zeros_(self.detail_out.weight)
        self.detail_out.weight.register_hook(self._sanitize_detail_gradient)

        # When the semantic gate is explicitly detached, p3_proj would otherwise
        # be a trainable-but-dead branch because it is used only by that gate.
        # Freeze it deliberately so optimizers and parameter audits reflect the
        # configured computation graph. The default keeps detach_gate=False, so
        # semantic agreement is learnable after the zero-start output branch opens.
        if self.detach_gate:
            self.p3_proj.requires_grad_(False)

    def _sanitize_detail_gradient(self, gradient):
        """Keep the zero-start residual projection numerically bounded during its warm-up."""
        limit = self.detail_grad_clip
        return torch.nan_to_num(
            gradient, nan=0.0, posinf=limit, neginf=-limit
        ).clamp(min=-limit, max=limit)

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

    @staticmethod
    def _centered_rms(x, eps):
        x_fp32 = x.float()
        centered = x_fp32 - x_fp32.mean(dim=(2, 3), keepdim=True)
        return centered.square().mean(dim=(2, 3), keepdim=True).add(eps).sqrt()

    def _anti_alias_downsample(self, x, target_size):
        if x.ndim != 4:
            raise ValueError(f"P2 feature must be 4D NCHW, got {tuple(x.shape)}.")
        weight = self.aa_kernel.to(device=x.device, dtype=x.dtype).repeat(
            x.shape[1], 1, 1, 1
        )
        filtered = F.conv2d(
            F.pad(x, (1, 1, 1, 1), mode="replicate"),
            weight,
            stride=2,
            padding=0,
            groups=x.shape[1],
        )
        if tuple(filtered.shape[-2:]) != tuple(target_size):
            if self.strict_scale:
                raise ValueError(
                    "BGDRP3Fuse spatial scale mismatch after anti-aliased downsampling: "
                    f"got {tuple(filtered.shape[-2:])}, expected {tuple(target_size)}."
                )
            filtered = F.interpolate(filtered, size=target_size, mode="nearest")
        return filtered

    def _validate_inputs(self, p2, p3):
        if not isinstance(p2, torch.Tensor) or not isinstance(p3, torch.Tensor):
            raise TypeError("BGDRP3Fuse expects tensor inputs [p2_feature, p3_feature].")
        if p2.ndim != 4 or p3.ndim != 4:
            raise ValueError(
                f"BGDRP3Fuse expects 4D NCHW tensors, got {tuple(p2.shape)} and {tuple(p3.shape)}."
            )
        if p2.shape[0] != p3.shape[0]:
            raise ValueError(f"Batch mismatch: P2={p2.shape[0]}, P3={p3.shape[0]}.")
        if p2.shape[1] != self.c_p2 or p3.shape[1] != self.c_p3:
            raise ValueError(
                "Channel mismatch: "
                f"P2={p2.shape[1]}/{self.c_p2}, P3={p3.shape[1]}/{self.c_p3}."
            )
        if p2.device != p3.device:
            raise ValueError(f"Device mismatch: P2={p2.device}, P3={p3.device}.")
        if p2.dtype != p3.dtype:
            raise ValueError(f"Dtype mismatch: P2={p2.dtype}, P3={p3.dtype}.")
        expected = (p3.shape[-2] * 2, p3.shape[-1] * 2)
        if self.strict_scale and tuple(p2.shape[-2:]) != expected:
            raise ValueError(
                "BGDRP3Fuse requires P2 spatial size to be exactly 2x P3, "
                f"got P2={tuple(p2.shape[-2:])}, P3={tuple(p3.shape[-2:])}."
            )

    def _compute_gate(self, p2_embed, p3_embed):
        if p2_embed.shape != p3_embed.shape:
            raise ValueError(
                f"Embedded P2/P3 shape mismatch: {tuple(p2_embed.shape)} vs {tuple(p3_embed.shape)}."
            )

        p2_fp32 = p2_embed.float()
        p3_fp32 = p3_embed.float()
        cosine = F.cosine_similarity(p2_fp32, p3_fp32, dim=1, eps=self.eps).unsqueeze(1)
        agreement = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0).pow(
            self.agreement_power
        )

        detail = p2_fp32 - self._replicate_avg_pool2d(
            p2_fp32, self.detail_kernel
        )
        magnitude = detail.abs().mean(dim=1, keepdim=True)
        local_mean = self._replicate_avg_pool2d(magnitude, self.support_kernel)
        local_second = self._replicate_avg_pool2d(
            magnitude.square(), self.support_kernel
        )
        local_std = (local_second - local_mean.square()).clamp_min(0.0).add(
            self.eps
        ).sqrt()

        # Salience preserves compact responses; coherence suppresses isolated
        # high-frequency background and suspended-particle noise.
        ratio = magnitude / local_mean.add(self.eps)
        salience = (ratio / (1.0 + ratio)).clamp(0.0, 1.0)
        local_support = self._replicate_avg_pool2d(salience, self.support_kernel)
        support = (salience * local_support).clamp(0.0, 1.0).sqrt().pow(
            self.support_power
        )
        coherence = (local_mean / (local_mean + local_std + self.eps)).clamp(
            0.0, 1.0
        )
        coherence = self.coherence_floor + (1.0 - self.coherence_floor) * coherence
        coherence = coherence.pow(self.coherence_power)

        gate = (agreement * support * coherence).clamp(0.0, 1.0)
        if self.finite_fallback:
            gate = torch.where(torch.isfinite(gate), gate, torch.zeros_like(gate))
            detail = torch.where(torch.isfinite(detail), detail, torch.zeros_like(detail))
        elif not torch.isfinite(gate).all() or not torch.isfinite(detail).all():
            raise RuntimeError("BGDRP3Fuse gate/detail contains non-finite values.")
        if self.detach_gate:
            gate = gate.detach()
        return gate, detail.to(dtype=p2_embed.dtype)

    def _bound_correction(self, base, correction):
        correction_fp32 = correction.float()
        if self.finite_fallback:
            correction_fp32 = torch.where(
                torch.isfinite(correction_fp32),
                correction_fp32,
                torch.zeros_like(correction_fp32),
            )
        elif not torch.isfinite(correction_fp32).all():
            raise RuntimeError("BGDRP3Fuse correction contains non-finite values.")

        if self.center_correction:
            correction_fp32 = correction_fp32 - correction_fp32.mean(
                dim=(2, 3), keepdim=True
            )

        base_rms = self._centered_rms(base, self.eps)
        correction_rms = self._centered_rms(correction_fp32, self.eps)
        bound = torch.minimum(
            torch.ones_like(correction_rms),
            self.max_residual_ratio
            * base_rms
            / correction_rms.clamp_min(self.eps),
        )
        if self.detach_bound:
            bound = bound.detach()
        bounded = correction_fp32 * bound

        if self.finite_fallback:
            bounded = torch.where(
                torch.isfinite(bounded), bounded, torch.zeros_like(bounded)
            )
        elif not torch.isfinite(bounded).all():
            raise RuntimeError("BGDRP3Fuse bounded correction contains non-finite values.")
        return bounded

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("BGDRP3Fuse expects input as [p2_feature, p3_feature].")
        p2, p3 = x
        self._validate_inputs(p2, p3)

        p2_down = self._anti_alias_downsample(p2, p3.shape[-2:])
        p2_embed = self.p2_proj(p2_down)
        p3_embed = self.p3_proj(p3)
        gate, detail = self._compute_gate(p2_embed, p3_embed)
        refined = self.detail_refine(detail * gate.to(dtype=detail.dtype))

        if self.detail_out.weight.dtype == torch.float32:
            with torch.autocast(device_type=p3.device.type, enabled=False):
                correction = self.detail_out(refined.float())
        else:
            correction = self.detail_out(
                refined.to(dtype=self.detail_out.weight.dtype)
            ).float()

        correction = self._bound_correction(p3, correction)
        output = p3.float() + correction
        if p3.dtype != torch.float32:
            output = output.to(dtype=p3.dtype)

        if tuple(output.shape) != tuple(p3.shape):
            raise RuntimeError(
                f"BGDRP3Fuse output mismatch: got {tuple(output.shape)}, expected {tuple(p3.shape)}."
            )
        if self.finite_fallback:
            output = torch.where(torch.isfinite(output), output, p3.to(dtype=output.dtype))
        elif not torch.isfinite(output).all():
            raise RuntimeError("BGDRP3Fuse output contains non-finite values.")
        return output
