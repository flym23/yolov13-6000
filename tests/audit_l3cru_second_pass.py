from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import test_l3cru_modules_isolated as env

AMSCLCERDCRAUp = env.AMSCLCERDCRAUp
BGDRP3Fuse = env.BGDRP3Fuse
UGDRDetect = env.UGDRDetect
_UGDRLogitAdapter = env._UGDRLogitAdapter


def assert_finite(x):
    assert torch.isfinite(x).all(), "non-finite tensor"


# 1) CPU autocast finite behavior.
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    a = AMSCLCERDCRAUp(32, 16).train()
    deep = torch.randn(2, 32, 5, 7)
    lat = torch.randn(2, 16, 10, 14)
    y = a([deep, lat])
assert_finite(y)

with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    b = BGDRP3Fuse(16, 24).train()
    p2 = torch.randn(2, 16, 20, 28)
    p3 = torch.randn(2, 24, 10, 14)
    y = b([p2, p3])
assert_finite(y)

with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    h = UGDRDetect(4, {"level_strengths": [1.0, 0.5, 0.0]}, (24, 48, 96)).train()
    xs = [
        torch.randn(2, 24, 16, 18),
        torch.randn(2, 48, 8, 9),
        torch.randn(2, 96, 4, 5),
    ]
    ys = h(xs)
assert all(torch.isfinite(t).all() for t in ys)

# 2) AMSC non-endpoint entropy mode honors detach/fallback; corrupted weights remain conservative.
m = AMSCLCERDCRAUp(
    16,
    8,
    {
        "evidence_mode": "entropy",
        "consensus_kernels": [3, 5],
        "consensus_weights": [0.5, 0.5],
        "detach_confidence": False,
        "detach_release": False,
        "detach_evidence": True,
    },
)
w = torch.softmax(torch.randn(2, 9, 8, 10, requires_grad=True), dim=1)
c = torch.rand(2, 1, 8, 10, requires_grad=True)
e = m._compute_release_evidence(w, c)
assert not e.requires_grad
w_bad = w.detach().clone()
w_bad[:, 0, 0, 0] = float("nan")
e_bad = m._compute_release_evidence(w_bad, c.detach())
assert_finite(e_bad)
assert e_bad.min() >= 0 and e_bad.max() <= 1

# 3) BGDR coherence reduces an isolated spike relative to coherent 3x3 support.
m = BGDRP3Fuse(8, 8, {"support_kernel": 5, "coherence_floor": 0.0}).eval()
coherent = torch.zeros(1, m.hidden, 15, 15)
isolated = torch.zeros_like(coherent)
coherent[:, :, 6:9, 6:9] = 1.0
isolated[:, :, 7, 7] = 1.0
g_coh, _ = m._compute_gate(coherent, coherent)
g_iso, _ = m._compute_gate(isolated, isolated)
assert g_coh[:, :, 6:9, 6:9].mean() > g_iso[:, :, 7:8, 7:8].mean()

# 4) BGDR has no trainable dead semantic projection in default config.
torch.manual_seed(9)
b = BGDRP3Fuse(16, 24).train()
assert b.detach_gate is False
p2 = torch.randn(2, 16, 20, 28)
p3 = torch.randn(2, 24, 10, 14)
opt = torch.optim.SGD(b.parameters(), lr=0.05)
b([p2, p3]).square().mean().backward()
opt.step()
opt.zero_grad(set_to_none=True)
b([p2, p3]).square().mean().backward()
for parameter in (b.p2_proj[0].weight, b.p3_proj[0].weight, b.detail_refine[0].weight):
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0
frozen = BGDRP3Fuse(16, 24, {"detach_gate": True})
assert all(not p.requires_grad for p in frozen.p3_proj.parameters())

# 5) Classification and P5 box path remain untouched after activating UGDR weights.
torch.manual_seed(4)
h = UGDRDetect(4, {"level_strengths": [1.0, 0.5, 0.0]}, (24, 48, 96)).eval()
for mod in h.box_refine:
    if isinstance(mod, _UGDRLogitAdapter):
        nn.init.normal_(mod.out.weight, std=0.02)
        nn.init.normal_(mod.out.bias, std=0.02)
xs = [
    torch.randn(2, 24, 16, 18),
    torch.randn(2, 48, 8, 9),
    torch.randn(2, 96, 4, 5),
]
snapshots = [x.clone() for x in xs]
with torch.no_grad():
    raw_cls = [h.cv3[i](xs[i]) for i in range(3)]
    raw_p5_box = h.cv2[2](xs[2])
    out = h(xs)
    tensors = out[1]
    for i in range(3):
        assert torch.equal(tensors[i][:, 64:], raw_cls[i])
    assert torch.equal(tensors[2][:, :64], raw_p5_box)
    assert all(torch.equal(a, b) for a, b in zip(xs, snapshots))

# 6) Non-finite projected corrections fall back to safe endpoints.
b = BGDRP3Fuse(16, 24).eval()
with torch.no_grad():
    b.detail_out.weight.fill_(float("nan"))
p2 = torch.randn(2, 16, 20, 28)
p3 = torch.randn(2, 24, 10, 14)
with torch.no_grad():
    y = b([p2, p3])
assert torch.equal(y, p3)

a = AMSCLCERDCRAUp(32, 16).eval()
with torch.no_grad():
    a.residual_out.weight.fill_(float("nan"))
deep = torch.randn(2, 32, 5, 7)
lat = torch.randn(2, 16, 10, 14)
with torch.no_grad():
    y = a([deep, lat])
assert torch.equal(y, F.interpolate(deep, size=lat.shape[-2:], mode="nearest").float())

print("INDEPENDENT REVIEWED-V2 SECOND-PASS AUDIT PASSED")
