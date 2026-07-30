class AMSCLCERDCRAUp(LCERDCRAUp):
    """
    Ambiguity-aware Multi-Scale Consensus LCER-DCRA upsampling.

    The module keeps the complete LCER-L3 moment-preserving and energy-routed
    correction path, but replaces its single entropy / single-window release
    evidence with two conservative additions:
      1) candidate ambiguity from the normalized top-1/top-2 margin;
      2) weighted multi-window spatial consensus.

    No trainable parameter or persistent buffer is added. When
    evidence_mode="entropy", consensus_kernels=[3], consensus_weights=[1.0]
    and the parent LCER configuration is L3, the forward is bitwise equivalent
    to LCER-L3 for finite inputs and an identical state_dict.

    Required existing symbols in block.py:
        math, torch, torch.nn.functional as F, LCERDCRAUp
    """

    _VALID_EVIDENCE_MODES = frozenset(("entropy", "entropy_margin"))
    _DEFAULT_CONFIG = dict(LCERDCRAUp._DEFAULT_CONFIG)
    _DEFAULT_CONFIG.update(
        {
            "evidence_mode": "entropy_margin",
            "margin_power": 1.0,
            "margin_floor": 0.50,
            "consensus_kernels": [3, 5],
            "consensus_weights": [0.65, 0.35],
            "detach_evidence": True,
            "finite_fallback": True,
        }
    )

    def __init__(self, c_deep, c_lateral, config=None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(
                f"AMSCLCERDCRAUp config must be a dict or None, got {type(config).__name__}."
            )
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown AMSCLCERDCRAUp config keys: {unknown}.")

        cfg = dict(self._DEFAULT_CONFIG)
        cfg.update(config)

        evidence_mode = str(cfg["evidence_mode"]).lower()
        margin_power = float(cfg["margin_power"])
        margin_floor = float(cfg["margin_floor"])
        kernels = tuple(int(value) for value in cfg["consensus_kernels"])
        weights = tuple(float(value) for value in cfg["consensus_weights"])

        if evidence_mode not in self._VALID_EVIDENCE_MODES:
            raise ValueError(
                "evidence_mode must be one of "
                f"{sorted(self._VALID_EVIDENCE_MODES)}, got {evidence_mode!r}."
            )
        if margin_power <= 0.0:
            raise ValueError(f"margin_power must be positive, got {margin_power}.")
        if not 0.0 <= margin_floor <= 1.0:
            raise ValueError(
                f"margin_floor must satisfy 0 <= floor <= 1, got {margin_floor}."
            )
        if not kernels:
            raise ValueError("consensus_kernels must contain at least one kernel.")
        if len(kernels) != len(weights):
            raise ValueError(
                "consensus_kernels and consensus_weights must have equal length, "
                f"got {len(kernels)} and {len(weights)}."
            )
        for kernel in kernels:
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError(
                    "Every consensus kernel must be a positive odd integer, "
                    f"got {kernels}."
                )
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"consensus_weights must be non-negative, got {weights}.")
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("consensus_weights must contain a positive value.")
        weights = tuple(weight / weight_sum for weight in weights)

        parent_cfg = {key: cfg[key] for key in LCERDCRAUp._DEFAULT_CONFIG}
        super().__init__(c_deep=c_deep, c_lateral=c_lateral, config=parent_cfg)

        self.evidence_mode = evidence_mode
        self.margin_power = margin_power
        self.margin_floor = margin_floor
        self.consensus_kernels = kernels
        self.consensus_weights = weights
        self.detach_evidence = bool(cfg["detach_evidence"])
        self.finite_fallback = bool(cfg["finite_fallback"])

    @property
    def _is_exact_l3_consensus_endpoint(self):
        return (
            self.evidence_mode == "entropy"
            and self.consensus_kernels == (self.consensus_kernel,)
            and len(self.consensus_weights) == 1
            and abs(self.consensus_weights[0] - 1.0) <= 1e-12
        )

    def _validate_candidate_weights(self, weights, confidence):
        if weights.ndim != 4:
            raise ValueError(
                f"weights must be a 4D [B,K,H,W] tensor, got {tuple(weights.shape)}."
            )
        if weights.shape[1] != self.num_candidates:
            raise ValueError(
                f"weights candidate count must be {self.num_candidates}, got {weights.shape[1]}."
            )
        self._validate_confidence(confidence, weights[:, :1])
        if weights.shape[0] != confidence.shape[0] or weights.shape[-2:] != confidence.shape[-2:]:
            raise ValueError(
                "weights and confidence must share batch and spatial dimensions, got "
                f"weights={tuple(weights.shape)}, confidence={tuple(confidence.shape)}."
            )

    def _compute_margin(self, weights):
        weights_fp32 = weights.float()
        if self.finite_fallback:
            weights_fp32 = torch.nan_to_num(
                weights_fp32, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
            weights_fp32 = weights_fp32 / weights_fp32.sum(
                dim=1, keepdim=True
            ).clamp_min(self.eps)
        elif not torch.isfinite(weights_fp32).all():
            raise RuntimeError("AMSCLCERDCRAUp candidate weights contain non-finite values.")

        top2 = torch.topk(
            weights_fp32, k=2, dim=1, largest=True, sorted=True
        ).values
        top1 = top2[:, :1]
        top2_value = top2[:, 1:2]
        # A normalized margin is less sensitive to the absolute sharpness of the
        # softmax distribution than the raw top1-top2 gap.
        margin = (top1 - top2_value) / top1.clamp_min(self.eps)
        return margin.clamp(0.0, 1.0).pow(self.margin_power)

    def _compute_release_evidence(self, weights, entropy_confidence):
        self._validate_candidate_weights(weights, entropy_confidence)
        entropy_fp32 = entropy_confidence.float().clamp(0.0, 1.0)
        if self.evidence_mode == "entropy":
            # The exact endpoint bypasses this method through super().forward().
            # For non-endpoint entropy-only configurations, still honor the
            # finite-fallback and detach_evidence contracts.
            evidence = entropy_fp32
        else:
            margin = self._compute_margin(weights)
            # The floor prevents ambiguous small-object locations from being erased;
            # the margin only attenuates the entropy evidence instead of replacing it.
            ambiguity_modifier = self.margin_floor + (1.0 - self.margin_floor) * margin
            evidence = entropy_fp32 * ambiguity_modifier

        if self.finite_fallback:
            evidence = torch.where(
                torch.isfinite(evidence), evidence, torch.zeros_like(evidence)
            )
        elif not torch.isfinite(evidence).all():
            raise RuntimeError("AMSCLCERDCRAUp release evidence contains non-finite values.")

        evidence = evidence.clamp(0.0, 1.0)
        if self.detach_evidence or self.detach_release:
            evidence = evidence.detach()
        return evidence

    def _compute_spatial_consensus(self, confidence, reference):
        if self._is_exact_l3_consensus_endpoint:
            return super()._compute_spatial_consensus(confidence, reference)

        self._validate_confidence(confidence, reference)
        confidence_fp32 = confidence.float().clamp(0.0, 1.0)
        consensus = torch.zeros_like(confidence_fp32)
        for kernel, weight in zip(self.consensus_kernels, self.consensus_weights):
            local_mean = self._replicate_avg_pool2d(confidence_fp32, kernel)
            local = (
                (confidence_fp32 * local_mean)
                .clamp(0.0, 1.0)
                .sqrt()
                .pow(self.spatial_power)
                .clamp(0.0, 1.0)
            )
            consensus = consensus + float(weight) * local

        if self.finite_fallback:
            consensus = torch.where(
                torch.isfinite(consensus), consensus, torch.zeros_like(consensus)
            )
        elif not torch.isfinite(consensus).all():
            raise RuntimeError("AMSCLCERDCRAUp spatial consensus contains non-finite values.")

        consensus = consensus.clamp(0.0, 1.0)
        return consensus.detach() if self.detach_release else consensus

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("AMSCLCERDCRAUp expects input as [deep_feature, lateral_feature].")

        if self._is_exact_l3_consensus_endpoint:
            # Avoid even harmless extra casts in the strict L3 endpoint.
            return super().forward(x)

        deep, lateral = x
        base, residual, weights, entropy_confidence = self._compute_alignment(deep, lateral)
        evidence = self._compute_release_evidence(weights, entropy_confidence)

        # DCRA's residual is already entropy-weighted. Reweight it without
        # reconstructing the ungated delta, which would be unstable at zero entropy.
        entropy_fp32 = entropy_confidence.float().clamp(0.0, 1.0)
        ratio = torch.where(
            entropy_fp32 > self.eps,
            evidence.float() / entropy_fp32.clamp_min(self.eps),
            torch.zeros_like(entropy_fp32),
        ).clamp(0.0, 1.0)
        if self.detach_evidence or self.detach_release:
            ratio = ratio.detach()
        residual = residual * ratio.to(device=residual.device, dtype=residual.dtype)

        if self.finite_fallback:
            residual = torch.where(torch.isfinite(residual), residual, torch.zeros_like(residual))
        elif not torch.isfinite(residual).all():
            raise RuntimeError("AMSCLCERDCRAUp residual contains non-finite values.")

        residual = self._moment_preserving_residual(base, residual)
        if self.residual_out.weight.dtype == torch.float32:
            with torch.autocast(device_type=deep.device.type, enabled=False):
                correction = self.residual_out(residual.float())
        else:
            correction = self.residual_out(
                residual.to(dtype=self.residual_out.weight.dtype)
            ).float()

        if self.finite_fallback:
            correction = torch.where(
                torch.isfinite(correction), correction, torch.zeros_like(correction)
            )
        elif not torch.isfinite(correction).all():
            raise RuntimeError("AMSCLCERDCRAUp correction contains non-finite values.")

        output = base.float() + self._route_output_correction(base, correction, evidence)
        if self.residual_out.weight.dtype != torch.float32:
            output = output.to(dtype=deep.dtype)

        expected_shape = (
            deep.shape[0],
            self.c_deep,
            lateral.shape[-2],
            lateral.shape[-1],
        )
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "AMSCLCERDCRAUp output-shape mismatch: "
                f"got {tuple(output.shape)}, expected {expected_shape}."
            )
        if self.finite_fallback:
            output = torch.where(torch.isfinite(output), output, base.float().to(dtype=output.dtype))
        elif not torch.isfinite(output).all():
            raise RuntimeError("AMSCLCERDCRAUp output contains non-finite values.")
        return output
