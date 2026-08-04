from __future__ import annotations

import ast
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parent


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = ((k - 1) * d) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def DWConv(c1, c2, k=1, s=1, d=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = int(c1)

    def forward(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        proj = torch.arange(self.c1, device=x.device, dtype=x.dtype).view(1, self.c1, 1, 1)
        return (x * proj).sum(1)


class DCRAUp(nn.Module):
    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        kernel_size=3,
        reduction=4,
        temperature=0.10,
        residual_groups=4,
        use_entropy=True,
        use_lateral_guidance=True,
        detach_confidence=True,
        strict_scale=True,
        eps=1e-6,
    ):
        super().__init__()
        self.c_deep = int(c_deep)
        self.c_lateral = int(c_lateral)
        self.scale = int(scale)
        self.kernel_size = int(kernel_size)
        self.reduction = int(reduction)
        self.temperature = float(temperature)
        self.use_entropy = bool(use_entropy)
        self.use_lateral_guidance = bool(use_lateral_guidance)
        self.detach_confidence = bool(detach_confidence)
        self.strict_scale = bool(strict_scale)
        self.eps = float(eps)
        if self.scale <= 1:
            raise ValueError
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError
        if self.reduction <= 0 or self.temperature <= 0.0 or self.eps <= 0.0:
            raise ValueError
        self.num_candidates = self.kernel_size**2
        self.embed_dim = max(16, min(64, min(self.c_deep, self.c_lateral) // self.reduction))
        self.residual_groups = math.gcd(self.c_deep, int(residual_groups))
        with torch.random.fork_rng(devices=[], enabled=True):
            local_seed = (
                int(torch.initial_seed())
                + 104729 * self.c_deep
                + 13007 * self.c_lateral
                + 1009 * self.kernel_size
                + 97 * self.scale
            ) % (2**63 - 1)
            torch.manual_seed(local_seed)
            self.key_proj = nn.Conv2d(self.c_deep, self.embed_dim, 1, bias=False)
            self.query_proj = (
                nn.Conv2d(self.c_lateral, self.embed_dim, 1, bias=False)
                if self.use_lateral_guidance
                else None
            )
            self.residual_out = nn.Conv2d(
                self.c_deep,
                self.c_deep,
                1,
                groups=self.residual_groups,
                bias=False,
            )
        nn.init.zeros_(self.residual_out.weight)

    def _validate_inputs(self, deep, lateral):
        if deep.ndim != 4 or lateral.ndim != 4:
            raise ValueError
        if deep.shape[0] != lateral.shape[0]:
            raise ValueError
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError
        if self.strict_scale and tuple(lateral.shape[-2:]) != (
            deep.shape[-2] * self.scale,
            deep.shape[-1] * self.scale,
        ):
            raise ValueError

    def _extract_local_patches(self, x):
        b, c, h, w = x.shape
        padding = self.kernel_size // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        patches = F.unfold(x, kernel_size=self.kernel_size)
        return patches.reshape(b, c, self.num_candidates, h, w)

    @staticmethod
    def _split_phases(x, low_size, scale):
        b, c = x.shape[:2]
        h, w = map(int, low_size)
        scale = int(scale)
        return x.reshape(b, c, h, scale, w, scale).permute(0, 1, 3, 5, 2, 4).contiguous()

    @staticmethod
    def _merge_phases(x):
        b, c, sh, sw, h, w = x.shape
        return x.permute(0, 1, 4, 2, 5, 3).contiguous().reshape(b, c, h * sh, w * sw)

    @staticmethod
    def _resize_patch_tensor(patches, target_size):
        b, c, k, h, w = patches.shape
        th, tw = map(int, target_size)
        y = patches.reshape(b, c * k, h, w)
        if (h, w) != (th, tw):
            y = F.interpolate(y, size=(th, tw), mode="nearest")
        return y.reshape(b, c, k, th, tw)

    def _phase_correlate_and_reassemble(self, query, key_patches, value_patches):
        with torch.autocast(device_type=query.device.type, enabled=False):
            low_size = key_patches.shape[-2:]
            query_phase = self._split_phases(
                F.normalize(query.float(), p=2, dim=1, eps=self.eps),
                low_size,
                self.scale,
            )
            key_norm = F.normalize(key_patches.float(), p=2, dim=1, eps=self.eps)
            logits = (query_phase.unsqueeze(2) * key_norm.unsqueeze(3).unsqueeze(3)).sum(dim=1)
            weights_phase = torch.softmax(logits / self.temperature, dim=1)
            reassembled_phase = torch.einsum(
                "bckhw,bkijhw->bcijhw", value_patches.float(), weights_phase
            )
            return self._merge_phases(reassembled_phase), self._merge_phases(weights_phase)

    def _fallback_correlate_and_reassemble(self, query, key_patches, value_patches, target_size):
        with torch.autocast(device_type=query.device.type, enabled=False):
            keys = self._resize_patch_tensor(key_patches.float(), target_size)
            query = F.normalize(query.float(), p=2, dim=1, eps=self.eps)
            keys = F.normalize(keys, p=2, dim=1, eps=self.eps)
            logits = (query.unsqueeze(2) * keys).sum(dim=1)
            weights = torch.softmax(logits / self.temperature, dim=1)
            values = self._resize_patch_tensor(value_patches.float(), target_size)
            return torch.einsum("bckhw,bkhw->bchw", values, weights), weights

    @staticmethod
    def _project_for_fp32_path(projection, feature):
        if projection.weight.dtype == torch.float32:
            with torch.autocast(device_type=feature.device.type, enabled=False):
                return projection(feature.float())
        return projection(feature.to(dtype=projection.weight.dtype)).float()

    def _compute_alignment(self, deep, lateral):
        self._validate_inputs(deep, lateral)
        target_size = tuple(lateral.shape[-2:])
        base = F.interpolate(deep, size=target_size, mode="nearest")
        key = self._project_for_fp32_path(self.key_proj, deep)
        query = (
            self._project_for_fp32_path(self.query_proj, lateral)
            if self.use_lateral_guidance
            else F.interpolate(key, size=target_size, mode="nearest")
        )
        key_patches = self._extract_local_patches(key)
        value_patches = self._extract_local_patches(deep)
        if target_size == (deep.shape[-2] * self.scale, deep.shape[-1] * self.scale):
            reassembled, weights = self._phase_correlate_and_reassemble(query, key_patches, value_patches)
        else:
            reassembled, weights = self._fallback_correlate_and_reassemble(
                query, key_patches, value_patches, target_size
            )
        reassembled = reassembled.to(dtype=deep.dtype)
        if self.use_entropy:
            entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=1, keepdim=True)
            confidence = (1.0 - entropy / math.log(float(self.num_candidates))).clamp(0.0, 1.0)
        else:
            confidence = torch.ones(
                (deep.shape[0], 1, *target_size),
                device=deep.device,
                dtype=torch.float32,
            )
        if self.detach_confidence:
            confidence = confidence.detach()
        residual = (reassembled - base) * confidence.to(dtype=deep.dtype)
        return base, residual, weights, confidence


class MEDCRAUp(DCRAUp):
    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        kernel_size=3,
        reduction=4,
        temperature=0.20,
        residual_groups=4,
        use_entropy=True,
        use_lateral_guidance=True,
        detach_confidence=True,
        strict_scale=True,
        preserve_moments=True,
        center_correction=True,
        use_energy_bound=False,
        max_residual_ratio=0.20,
        detach_moment_scale=True,
        detach_energy_scale=True,
        moment_scale_max=4.0,
        eps=1e-6,
    ):
        super().__init__(
            c_deep,
            c_lateral,
            scale,
            kernel_size,
            reduction,
            temperature,
            residual_groups,
            use_entropy,
            use_lateral_guidance,
            detach_confidence,
            strict_scale,
            eps,
        )
        self.preserve_moments = bool(preserve_moments)
        self.center_correction = bool(center_correction)
        self.use_energy_bound = bool(use_energy_bound)
        self.max_residual_ratio = float(max_residual_ratio)
        self.detach_moment_scale = bool(detach_moment_scale)
        self.detach_energy_scale = bool(detach_energy_scale)
        self.moment_scale_max = float(moment_scale_max)

    @staticmethod
    def _spatial_mean_and_centered_rms(x, eps):
        x = x.float()
        mean = x.mean(dim=(2, 3), keepdim=True)
        centered = x - mean
        rms = centered.square().mean(dim=(2, 3), keepdim=True).add(eps).sqrt()
        return mean, centered, rms

    def _moment_preserving_residual(self, base, residual):
        if not self.preserve_moments:
            return residual
        base_mean, _, base_rms = self._spatial_mean_and_centered_rms(base, self.eps)
        candidate = base.float() + residual.float()
        _, candidate_centered, candidate_rms = self._spatial_mean_and_centered_rms(candidate, self.eps)
        scale = (base_rms / candidate_rms.clamp_min(self.eps)).clamp(max=self.moment_scale_max)
        if self.detach_moment_scale:
            scale = scale.detach()
        return (candidate_centered * scale + base_mean - base.float()).to(dtype=base.dtype)


class LCERDCRAUp(MEDCRAUp):
    _VALID_RELEASE_MODES = frozenset(("local", "channel", "strict", "none"))
    _DEFAULT_CONFIG = {
        "scale": 2,
        "kernel_size": 3,
        "reduction": 4,
        "temperature": 0.20,
        "residual_groups": 4,
        "use_entropy": True,
        "use_lateral_guidance": True,
        "detach_confidence": True,
        "strict_scale": True,
        "preserve_moments": True,
        "center_correction": True,
        "release_mode": "local",
        "strict_ratio": 0.20,
        "channel_power": 2.0,
        "spatial_power": 1.0,
        "consensus_kernel": 3,
        "energy_weighted_channel": True,
        "detach_moment_scale": True,
        "detach_release": True,
        "moment_scale_max": 4.0,
        "eps": 1e-6,
    }

    def __init__(self, c_deep, c_lateral, config=None):
        config = {} if config is None else config
        unknown = sorted(set(config) - set(self._DEFAULT_CONFIG))
        if unknown:
            raise ValueError(unknown)
        cfg = dict(self._DEFAULT_CONFIG)
        cfg.update(config)
        super().__init__(
            c_deep=c_deep,
            c_lateral=c_lateral,
            scale=int(cfg["scale"]),
            kernel_size=int(cfg["kernel_size"]),
            reduction=int(cfg["reduction"]),
            temperature=float(cfg["temperature"]),
            residual_groups=int(cfg["residual_groups"]),
            use_entropy=bool(cfg["use_entropy"]),
            use_lateral_guidance=bool(cfg["use_lateral_guidance"]),
            detach_confidence=bool(cfg["detach_confidence"]),
            strict_scale=bool(cfg["strict_scale"]),
            preserve_moments=bool(cfg["preserve_moments"]),
            center_correction=bool(cfg["center_correction"]),
            use_energy_bound=False,
            max_residual_ratio=float(cfg["strict_ratio"]),
            detach_moment_scale=bool(cfg["detach_moment_scale"]),
            detach_energy_scale=True,
            moment_scale_max=float(cfg["moment_scale_max"]),
            eps=float(cfg["eps"]),
        )
        self.release_mode = str(cfg["release_mode"]).lower()
        self.strict_ratio = float(cfg["strict_ratio"])
        self.channel_power = float(cfg["channel_power"])
        self.spatial_power = float(cfg["spatial_power"])
        self.consensus_kernel = int(cfg["consensus_kernel"])
        self.energy_weighted_channel = bool(cfg["energy_weighted_channel"])
        self.detach_release = bool(cfg["detach_release"])

    @staticmethod
    def _validate_confidence(confidence, reference):
        if confidence.ndim != 4 or reference.ndim != 4:
            raise ValueError
        if confidence.shape[0] != reference.shape[0] or confidence.shape[1] != 1:
            raise ValueError
        if confidence.shape[-2:] != reference.shape[-2:] or confidence.device != reference.device:
            raise ValueError

    @staticmethod
    def _replicate_avg_pool2d(x, kernel_size):
        if kernel_size == 1:
            return x
        pad = kernel_size // 2
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size, stride=1)

    def _compute_strict_scale(self, base, correction):
        _, _, base_rms = self._spatial_mean_and_centered_rms(base, self.eps)
        correction_rms = correction.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        scale = torch.minimum(
            torch.ones_like(correction_rms),
            self.strict_ratio * base_rms / correction_rms.clamp_min(self.eps),
        )
        return scale.detach() if self.detach_release else scale

    def _compute_channel_eligibility(self, confidence, correction):
        self._validate_confidence(confidence, correction)
        confidence = confidence.float().clamp(0.0, 1.0)
        b, c = correction.shape[:2]
        mean_conf = confidence.mean(dim=(2, 3), keepdim=True).expand(b, c, 1, 1)
        if self.energy_weighted_channel:
            energy = correction.float().square()
            energy_sum = energy.sum(dim=(2, 3), keepdim=True)
            weighted = (energy * confidence).sum(dim=(2, 3), keepdim=True) / energy_sum.clamp_min(self.eps)
            eligibility = torch.where(energy_sum > self.eps, weighted, mean_conf)
        else:
            eligibility = mean_conf
        eligibility = eligibility.clamp(0.0, 1.0).pow(self.channel_power)
        return eligibility.detach() if self.detach_release else eligibility

    def _compute_spatial_consensus(self, confidence, reference):
        self._validate_confidence(confidence, reference)
        confidence = confidence.float().clamp(0.0, 1.0)
        local_mean = self._replicate_avg_pool2d(confidence, self.consensus_kernel)
        consensus = (confidence * local_mean).clamp(0.0, 1.0).sqrt().pow(self.spatial_power).clamp(0.0, 1.0)
        return consensus.detach() if self.detach_release else consensus

    def _route_output_correction(self, base, correction, confidence):
        correction = correction.float()
        if self.center_correction:
            correction = correction - correction.mean(dim=(2, 3), keepdim=True)
        if self.release_mode == "none":
            return correction
        strict = self._compute_strict_scale(base, correction)
        if self.release_mode == "strict":
            return correction * strict
        channel = self._compute_channel_eligibility(confidence, correction)
        release = channel if self.release_mode == "channel" else torch.minimum(
            channel, self._compute_spatial_consensus(confidence, correction)
        )
        scale = (strict + release * (1.0 - strict)).clamp(0.0, 1.0)
        return correction * (scale.detach() if self.detach_release else scale)

    def forward(self, x):
        deep, lateral = x
        base, residual, _, confidence = self._compute_alignment(deep, lateral)
        residual = self._moment_preserving_residual(base, residual)
        with torch.autocast(device_type=deep.device.type, enabled=False):
            correction = self.residual_out(residual.float())
        return base.float() + self._route_output_correction(base, correction, confidence)


class Detect(nn.Module):
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    legacy = False

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max)

    def _inference(self, x):
        shape = x[0].shape
        return torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), dim=1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


# Load the proposed classes into the isolated parent environment.
namespace = globals()
for filename in (
    "AMSCLCERDCRAUp.py",
    "BGDRP3Fuse.py",
    "UGDRDetect.py",
):
    exec((ROOT / filename).read_text(encoding="utf-8"), namespace)

AMSCLCERDCRAUp = namespace["AMSCLCERDCRAUp"]
BGDRP3Fuse = namespace["BGDRP3Fuse"]
UGDRDetect = namespace["UGDRDetect"]
_UGDRLogitAdapter = namespace["_UGDRLogitAdapter"]


def assert_finite(x):
    assert torch.isfinite(x).all(), "tensor contains non-finite values"


def round_one_static_checks():
    for filename in (
        "AMSCLCERDCRAUp.py",
        "BGDRP3Fuse.py",
        "UGDRDetect.py",
        "test_l3cru_modules_isolated.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        ast.parse(source, filename=filename)
        compile(source, filename, "exec")

    model = yaml.safe_load((ROOT / "yamls" / "yolov13-l3cru-t7_full.yaml").read_text(encoding="utf-8"))
    rows = model["backbone"] + model["head"]
    assert model["nc"] == 4 and model["scale"] == "n"
    assert len(model["backbone"]) == 9 and len(model["head"]) == 25
    assert rows[15][0] == [-1, 12] and rows[15][2] == "AMSCLCERDCRAUp"
    assert rows[16][0] == [-1, 12] and rows[16][2] == "Concat"
    assert rows[32][0] == [2, 23] and rows[32][2] == "BGDRP3Fuse"
    assert rows[33][0] == [32, 27, 31] and rows[33][2] == "UGDRDetect"

    amsc_cfg = rows[15][3][0]
    bgdr_cfg = rows[32][3][0]
    ugdr_cfg = rows[33][3][1]
    assert set(amsc_cfg) <= set(AMSCLCERDCRAUp._DEFAULT_CONFIG)
    assert set(bgdr_cfg) <= set(BGDRP3Fuse._DEFAULT_CONFIG)
    assert set(ugdr_cfg) <= set(UGDRDetect._DEFAULT_CONFIG)


def round_two_dynamic_checks():
    # 1) Strict LCER-L3 endpoint and no new state.
    l3_cfg = dict(LCERDCRAUp._DEFAULT_CONFIG)
    endpoint_cfg = dict(l3_cfg)
    endpoint_cfg.update(
        {
            "evidence_mode": "entropy",
            "consensus_kernels": [3],
            "consensus_weights": [1.0],
        }
    )
    for seed in (0, 7, 19):
        torch.manual_seed(seed)
        parent = LCERDCRAUp(64, 32, l3_cfg).eval()
        child = AMSCLCERDCRAUp(64, 32, endpoint_cfg).eval()
        child.load_state_dict(parent.state_dict(), strict=True)
        assert len(child.state_dict()) == len(parent.state_dict())
        deep = torch.randn(2, 64, 7, 9)
        lateral = torch.randn(2, 32, 14, 18)
        with torch.no_grad():
            assert torch.equal(parent([deep, lateral]), child([deep, lateral]))

    # 2) AMSC bounds, zero-start identity and two-step gradient launch.
    amsc = AMSCLCERDCRAUp(64, 32).train()
    deep = torch.randn(2, 64, 7, 9)
    lateral = torch.randn(2, 32, 14, 18)
    nearest = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest").float()
    assert torch.equal(amsc([deep, lateral]), nearest)
    base, residual, weights, entropy = amsc._compute_alignment(deep, lateral)
    evidence = amsc._compute_release_evidence(weights, entropy)
    assert_finite(evidence)
    assert evidence.min() >= 0.0 and evidence.max() <= 1.0
    assert torch.all(evidence <= entropy.float().clamp(0.0, 1.0) + 1e-6)

    # Entropy-only multi-scale mode is not the strict endpoint and must still
    # honor detach_evidence and finite fallback.
    entropy_multiscale = AMSCLCERDCRAUp(
        32,
        16,
        {
            "detach_confidence": False,
            "detach_release": False,
            "detach_evidence": True,
            "evidence_mode": "entropy",
            "consensus_kernels": [3, 5],
            "consensus_weights": [0.5, 0.5],
        },
    )
    random_weights = torch.softmax(torch.randn(2, 9, 10, 12, requires_grad=True), dim=1)
    random_entropy = torch.rand(2, 1, 10, 12, requires_grad=True)
    entropy_evidence = entropy_multiscale._compute_release_evidence(
        random_weights, random_entropy
    )
    assert not entropy_evidence.requires_grad
    corrupted_weights = random_weights.detach().clone()
    corrupted_weights[:, 0, 0, 0] = float("nan")
    recovered = entropy_multiscale._compute_release_evidence(
        corrupted_weights, random_entropy.detach()
    )
    assert_finite(recovered)

    optimizer = torch.optim.SGD(amsc.parameters(), lr=0.05)
    amsc([deep, lateral]).square().mean().backward()
    assert amsc.residual_out.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    amsc([deep, lateral]).square().mean().backward()
    assert amsc.key_proj.weight.grad.abs().sum() > 0
    assert amsc.query_proj.weight.grad.abs().sum() > 0

    # 3) BGDR identity, gate, coherence, bound, reload and gradients.
    bgdr = BGDRP3Fuse(32, 48).train()
    assert bgdr.detach_gate is False
    detached_bgdr = BGDRP3Fuse(32, 48, {"detach_gate": True})
    assert all(not parameter.requires_grad for parameter in detached_bgdr.p3_proj.parameters())
    p2 = torch.randn(2, 32, 28, 36)
    p3 = torch.randn(2, 48, 14, 18)
    assert torch.equal(bgdr([p2, p3]), p3)
    clone = BGDRP3Fuse(32, 48)
    clone.load_state_dict(bgdr.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(bgdr.eval()([p2, p3]), clone.eval()([p2, p3]))
    bgdr.train()
    p2_down = bgdr._anti_alias_downsample(p2, p3.shape[-2:])
    gate, detail = bgdr._compute_gate(bgdr.p2_proj(p2_down), bgdr.p3_proj(p3))
    assert_finite(gate)
    assert gate.min() >= 0.0 and gate.max() <= 1.0
    constant = torch.full_like(p3, 100.0)
    assert bgdr._bound_correction(p3, constant).abs().max().item() == 0.0
    optimizer = torch.optim.SGD(bgdr.parameters(), lr=0.05)
    bgdr([p2, p3]).square().mean().backward()
    assert bgdr.detail_out.weight.grad.abs().sum() > 0
    guarded = bgdr._sanitize_bgdr_gradient(
        torch.full_like(bgdr.detail_out.weight, float("nan"))
    )
    assert_finite(guarded)
    assert guarded.abs().max().item() == 0.0
    clipped = bgdr._sanitize_bgdr_gradient(
        torch.full_like(bgdr.detail_out.weight, bgdr.detail_grad_clip * 10.0)
    )
    assert clipped.abs().max().item() == bgdr.detail_grad_clip
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = bgdr([p2, p3])
    output.square().mean().backward()
    assert bgdr.p2_proj[0].weight.grad.abs().sum() > 0
    assert bgdr.p3_proj[0].weight.grad is not None
    assert bgdr.p3_proj[0].weight.grad.abs().sum() > 0
    correction = output.float() - p3.float()
    assert correction.mean(dim=(2, 3)).abs().max() < 1e-5
    assert torch.all(
        bgdr._centered_rms(correction, bgdr.eps)
        <= bgdr.max_residual_ratio * bgdr._centered_rms(p3, bgdr.eps) + 2e-3
    )

    for i in range(30):
        torch.manual_seed(100 + i)
        h, w = 5 + i % 8, 7 + (i * 3) % 9
        cp2 = (8, 16, 24, 32)[i % 4]
        cp3 = (16, 24, 32, 48)[i % 4]
        module = BGDRP3Fuse(cp2, cp3).eval()
        a = torch.randn(2, cp2, 2 * h, 2 * w)
        b = torch.randn(2, cp3, h, w)
        with torch.no_grad():
            y = module([a, b])
        assert torch.equal(y, b)
        assert_finite(y)

    # 4) UGDR exact Detect endpoint, uncertainty behavior and gradients.
    for seed in (1, 11, 29):
        channels = (32, 64, 128)
        torch.manual_seed(seed)
        base_head = Detect(4, channels).train()
        torch.manual_seed(seed)
        head = UGDRDetect(4, {"level_strengths": [1.0, 0.5, 0.0]}, channels).train()
        features = [
            torch.randn(2, channels[0], 20, 20),
            torch.randn(2, channels[1], 10, 10),
            torch.randn(2, channels[2], 5, 5),
        ]
        base_output = base_head([x.clone() for x in features])
        head_inputs = [x.clone() for x in features]
        snapshots = [x.clone() for x in head_inputs]
        head_output = head(head_inputs)
        assert all(torch.equal(a, b) for a, b in zip(base_output, head_output))
        assert all(torch.equal(a, b) for a, b in zip(head_inputs, snapshots))

    head = UGDRDetect(4, {"level_strengths": [1.0, 0.5, 0.0]}, (32, 64, 128)).train()
    features = [
        torch.randn(2, 32, 20, 20),
        torch.randn(2, 64, 10, 10),
        torch.randn(2, 128, 5, 5),
    ]
    active = [module for module in head.box_refine if isinstance(module, _UGDRLogitAdapter)]
    adapter = active[0]
    uniform = torch.zeros(2, 4 * adapter.reg_max, 6, 7)
    peaked = uniform.clone().reshape(2, 4, adapter.reg_max, 6, 7)
    peaked[:, :, 3] = 12.0
    peaked = peaked.reshape_as(uniform)
    assert adapter._distribution_uncertainty(uniform).mean() > adapter._distribution_uncertainty(peaked).mean()

    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    outputs = head([x.clone() for x in features])
    sum(output.square().mean() for output in outputs).backward()
    assert sum(module.out.weight.grad.abs().sum() for module in active) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    outputs = head([x.clone() for x in features])
    sum(output.square().mean() for output in outputs).backward()
    assert sum(module.depthwise.weight.grad.abs().sum() for module in active) > 0

    with torch.no_grad():
        for level, module in enumerate(active):
            feature = features[level]
            box_logits = head.cv2[level](feature)
            delta = module(feature, box_logits).float()
            assert_finite(delta)
            assert delta.abs().max() <= module.level_strength * module.max_logit_delta + 1e-6
            shaped = delta.reshape(delta.shape[0], 4, module.reg_max, *delta.shape[-2:])
            # Multiplication by per-edge uncertainty preserves the zero-bin-mean property.
            assert shaped.mean(dim=2).abs().max() < 3e-6

    for i in range(24):
        torch.manual_seed(200 + i)
        channels = (16 + 8 * (i % 3), 32 + 16 * (i % 3), 64 + 32 * (i % 3))
        module = UGDRDetect(4, {"level_strengths": [1.0, 0.5, 0.0]}, channels).eval()
        xs = [
            torch.randn(2, channels[0], 16, 18),
            torch.randn(2, channels[1], 8, 9),
            torch.randn(2, channels[2], 4, 5),
        ]
        with torch.no_grad():
            y = module(xs)[0]
        assert_finite(y)

    # 5) Configuration blocking.
    invalid_amsc = (
        {"evidence_mode": "invalid"},
        {"margin_power": 0.0},
        {"margin_floor": -0.1},
        {"consensus_kernels": [2]},
        {"consensus_kernels": [3], "consensus_weights": [0.5, 0.5]},
        {"consensus_weights": [0.0, 0.0]},
    )
    for cfg in invalid_amsc:
        try:
            AMSCLCERDCRAUp(16, 16, cfg)
            raise AssertionError(cfg)
        except ValueError:
            pass

    invalid_bgdr = (
        {"reduction": 0},
        {"detail_kernel": 2},
        {"support_kernel": 2},
        {"agreement_power": 0.0},
        {"coherence_floor": 1.1},
        {"max_residual_ratio": 0.0},
        {"detail_grad_clip": 0.0},
        {"eps": 0.0},
    )
    for cfg in invalid_bgdr:
        try:
            BGDRP3Fuse(16, 16, cfg)
            raise AssertionError(cfg)
        except ValueError:
            pass

    invalid_ugdr = (
        {"reduction": 0, "level_strengths": [0.0, 0.0, 0.0]},
        {"detail_kernel": 2, "level_strengths": [0.0, 0.0, 0.0]},
        {"support_kernel": 2, "level_strengths": [0.0, 0.0, 0.0]},
        {"max_logit_delta": 0.0, "level_strengths": [0.0, 0.0, 0.0]},
        {"uncertainty_power": 0.0, "level_strengths": [0.0, 0.0, 0.0]},
        {"level_strengths": [1.0, 0.5]},
        {"level_strengths": [1.1, 0.5, 0.0]},
    )
    for cfg in invalid_ugdr:
        try:
            UGDRDetect(4, cfg, (16, 32, 64))
            raise AssertionError(cfg)
        except ValueError:
            pass


if __name__ == "__main__":
    round_one_static_checks()
    print("ROUND 1 STATIC / DESIGN CHECKS PASSED")
    round_two_dynamic_checks()
    print("ROUND 2 DYNAMIC / ADVERSARIAL CHECKS PASSED")
