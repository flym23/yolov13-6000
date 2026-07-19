# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .conv import Conv, DSConv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "HyperACE",
    "DPRFuseModule",
    "DPRHyperACE",
    "DownsampleConv",
    "FullPAD_Tunnel",
    "RATunnel",
    "DSC3k2",
    "UWFeatCalib",
    "CtrlP2Fuse",
    "AADown",
    "ScaleRebalance",
    "WTFeatCalib",
    "EdgeConfidenceP2Fuse",
    "AARUp",
    "AARDown",
    "FastNormFusion",
    "LiteBiFPNNode",
    "UDCStem",
    "UCRA_SemUp",
    "UCRA_DetailUp",
    "SIRUCRA_DetailUp",
    "AARUpLite",
    "P2LiteGuide",
    "FAARUp",
    "DCRAUp",
    "WSDRFuse",
    "SCAFFuse",
    "RPSCAFFuse",
    "SFRSCAFFuse",
    "LGARUp",
    "RFABlock",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc ** 0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


import logging

logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func

        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")


class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes:
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes:
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2,
                                                                                                      shortcut, g) for _
            in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


class DSBottleneck(nn.Module):
    """
    An improved bottleneck block using depthwise separable convolutions (DSConv).

    This class implements a lightweight bottleneck module that replaces standard convolutions with depthwise
    separable convolutions to reduce parameters and computational cost.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to use a residual shortcut connection. The connection is only added if c1 == c2. Defaults to True.
        e (float, optional): Expansion ratio for the intermediate channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv layer. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv layer. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv layer. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSBottleneck module.

    Examples:
        >>> import torch
        >>> model = DSBottleneck(c1=64, c2=64, shortcut=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    """
    An improved C3k module using DSBottleneck blocks for lightweight feature extraction.

    This class extends the C3 module by replacing its standard bottleneck blocks with DSBottleneck blocks,
    which use depthwise separable convolutions.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of DSBottleneck blocks to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections within the DSBottlenecks. Defaults to True.
        g (int, optional): Number of groups for grouped convolution (passed to parent C3). Defaults to 1.
        e (float, optional): Expansion ratio for the C3 module's hidden channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv in each DSBottleneck. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in each DSBottleneck. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv in each DSBottleneck. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k module (inherited from C3).

    Examples:
        >>> import torch
        >>> model = DSC3k(c1=128, c2=128, n=2, k1=3, k2=7)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 64, 64])
    """

    def __init__(
            self,
            c1,
            c2,
            n=1,
            shortcut=True,
            g=1,
            e=0.5,
            k1=3,
            k2=5,
            d2=1
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                DSBottleneck(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )


class DSC3k2(C2f):
    """
    An improved C3k2 module that uses lightweight depthwise separable convolution blocks.

    This class redesigns C3k2 module, replacing its internal processing blocks with either DSBottleneck
    or DSC3k modules.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of internal processing blocks to stack. Defaults to 1.
        dsc3k (bool, optional): If True, use DSC3k as the internal block. If False, use DSBottleneck. Defaults to False.
        e (float, optional): Expansion ratio for the C2f module's hidden channels. Defaults to 0.5.
        g (int, optional): Number of groups for grouped convolution (passed to parent C2f). Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections in the internal blocks. Defaults to True.
        k1 (int, optional): Kernel size for the first DSConv in internal blocks. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in internal blocks. Defaults to 7.
        d2 (int, optional): Dilation for the second DSConv in internal blocks. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k2 module (inherited from C2f).

    Examples:
        >>> import torch
        >>> # Using DSBottleneck as internal block
        >>> model1 = DSC3k2(c1=64, c2=64, n=2, dsc3k=False)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output1 = model1(x)
        >>> print(f"With DSBottleneck: {output1.shape}")
        With DSBottleneck: torch.Size([2, 64, 128, 128])
        >>> # Using DSC3k as internal block
        >>> model2 = DSC3k2(c1=64, c2=64, n=1, dsc3k=True)
        >>> output2 = model2(x)
        >>> print(f"With DSC3k: {output2.shape}")
        With DSC3k: torch.Size([2, 64, 128, 128])
    """

    def __init__(
            self,
            c1,
            c2,
            n=1,
            dsc3k=False,
            e=0.5,
            g=1,
            shortcut=True,
            k1=3,
            k2=7,
            d2=1
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.c, self.c,
                    n=2,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(
                    self.c, self.c,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )


class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.

    Examples:
        >>> import torch
        >>> model = AdaHyperedgeGen(node_dim=64, num_hyperedges=16, num_heads=4)
        >>> x = torch.randn(2, 100, 64)  # (Batch, Num_Nodes, Node_Dim)
        >>> A = model(x)
        >>> print(A.shape)
        torch.Size([2, 100, 16])
    """

    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            avg_context = X.mean(dim=1)
            max_context, _ = X.max(dim=1)
            context_cat = torch.cat([avg_context, max_context], dim=-1)
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets

        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)

        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)

        logits = self.dropout(logits)

        return F.softmax(logits, dim=1)


class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    Examples:
        >>> import torch
        >>> model = AdaHGConv(embed_dim=128, num_hyperedges=16, num_heads=8)
        >>> x = torch.randn(2, 256, 128) # (Batch, Num_Nodes, Dim)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 256, 128])
    """

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, X):
        A = self.edge_generator(X)

        He = torch.bmm(A.transpose(1, 2), X)
        He = self.edge_proj(He)

        X_new = torch.bmm(A, He)
        X_new = self.node_proj(X_new)

        return X_new + X


class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens)
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out


class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    Examples:
        >>> import torch
        >>> model = C3AH(c1=64, c2=128, num_hyperedges=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])
    """

    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_,
                                  num_hyperedges=num_hyperedges,
                                  num_heads=num_heads,
                                  dropout=0.1,
                                  context=context)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FuseModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out


class DPRFuseModule(nn.Module):
    """Detail-preserving residual multi-scale fusion for HyperACE."""

    def __init__(
        self,
        c_in,
        channel_adjust=True,
        use_p3_detail=True,
        detail_alpha_max=0.10,
        p3_down_mode="spd",
        p3_fusion_mode="residual",
        use_p5_semantic=False,
        semantic_alpha_max=0.05,
    ):
        super().__init__()
        if p3_down_mode not in {"spd", "dwconv"}:
            raise ValueError(f"Unsupported p3_down_mode={p3_down_mode!r}; expected 'spd' or 'dwconv'.")
        if p3_fusion_mode not in {"residual", "replace"}:
            raise ValueError(f"Unsupported p3_fusion_mode={p3_fusion_mode!r}; expected 'residual' or 'replace'.")
        if int(c_in) <= 0:
            raise ValueError(f"c_in must be positive, got {c_in}.")
        if float(detail_alpha_max) < 0 or float(semantic_alpha_max) < 0:
            raise ValueError("detail_alpha_max and semantic_alpha_max must be non-negative.")

        self.c_in = int(c_in)
        self.channel_adjust = bool(channel_adjust)
        self.use_p3_detail = bool(use_p3_detail)
        self.detail_alpha_max = float(detail_alpha_max)
        self.p3_down_mode = str(p3_down_mode)
        self.p3_fusion_mode = str(p3_fusion_mode)
        self.use_p5_semantic = bool(use_p5_semantic)
        self.semantic_alpha_max = float(semantic_alpha_max)
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_channels = 2 * self.c_in if self.channel_adjust else self.c_in
        concat_channels = 4 * self.c_in if self.channel_adjust else 3 * self.c_in
        self.conv_out = Conv(concat_channels, self.c_in, 1)

        if self.p3_down_mode == "spd":
            self.p3_down = nn.Sequential(
                Conv(4 * self.c_in, self.c_in, 1, 1, act=False),
                Conv(self.c_in, self.c_in, 3, 1, g=self.c_in, act=False),
            )
        else:
            self.p3_down = nn.Sequential(
                Conv(self.c_in, self.c_in, 3, 2, g=self.c_in, act=False),
                Conv(self.c_in, self.c_in, 1, 1, act=False),
            )

        self.p3_refine = nn.Sequential(
            Conv(self.c_in, self.c_in, 3, 1, g=self.c_in, act=False),
            nn.Conv2d(self.c_in, self.c_in, 1, 1, 0, bias=True),
        )
        nn.init.normal_(self.p3_refine[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.p3_refine[-1].bias)
        self.detail_alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.p5_refine = nn.Sequential(
            Conv(self.p5_channels, self.p5_channels, 5, 1, g=self.p5_channels, act=False),
            nn.Conv2d(self.p5_channels, self.p5_channels, 1, 1, 0, bias=True),
        )
        nn.init.normal_(self.p5_refine[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.p5_refine[-1].bias)
        self.semantic_alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _p3_detail_downsample(self, p3):
        if self.p3_down_mode == "spd":
            pad_h = p3.shape[-2] % 2
            pad_w = p3.shape[-1] % 2
            if pad_h or pad_w:
                p3 = F.pad(p3, (0, pad_w, 0, pad_h), mode="replicate")
            p3 = F.pixel_unshuffle(p3, downscale_factor=2)
        return self.p3_down(p3)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise TypeError("DPRFuseModule expects input as [p3, p4, p5].")
        p3, p4, p5 = x
        if any(tensor.ndim != 4 for tensor in (p3, p4, p5)):
            raise ValueError("DPRFuseModule inputs must be 4D NCHW tensors.")
        if p3.shape[1] != self.c_in or p4.shape[1] != self.c_in or p5.shape[1] != self.p5_channels:
            raise ValueError(
                "DPRFuseModule channel mismatch: "
                f"expected ({self.c_in}, {self.c_in}, {self.p5_channels}), "
                f"got ({p3.shape[1]}, {p4.shape[1]}, {p5.shape[1]})."
            )

        target_size = p4.shape[-2:]
        p3_avg = self.downsample(p3)
        if p3_avg.shape[-2:] != target_size:
            p3_avg = F.interpolate(p3_avg, size=target_size, mode="nearest")

        if self.use_p3_detail:
            p3_detail = self._p3_detail_downsample(p3)
            if p3_detail.shape[-2:] != target_size:
                p3_detail = F.interpolate(p3_detail, size=target_size, mode="nearest")
            if self.p3_fusion_mode == "replace":
                p3_aligned = p3_detail
            else:
                detail_delta = self.p3_refine(p3_detail - p3_avg)
                detail_alpha = self.detail_alpha_max * torch.tanh(self.detail_alpha_raw)
                p3_aligned = p3_avg + detail_alpha * detail_delta
        else:
            p3_aligned = p3_avg

        p5_base = self.upsample(p5)
        if p5_base.shape[-2:] != target_size:
            p5_base = F.interpolate(p5_base, size=target_size, mode="nearest")
        if self.use_p5_semantic:
            p5_context = self.p5_refine(p5_base)
            semantic_alpha = self.semantic_alpha_max * torch.tanh(self.semantic_alpha_raw)
            p5_aligned = p5_base + semantic_alpha * (p5_context - p5_base)
        else:
            p5_aligned = p5_base

        return self.conv_out(torch.cat((p3_aligned, p4, p5_aligned), dim=1))


class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """

    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both",
                 channel_adjust=True):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class DPRHyperACE(HyperACE):
    """HyperACE variant that only replaces its internal fusion module with DPRFuseModule."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1,
        context="both",
        channel_adjust=True,
        use_p3_detail=True,
        detail_alpha_max=0.10,
        p3_down_mode="spd",
        p3_fusion_mode="residual",
        use_p5_semantic=False,
        semantic_alpha_max=0.05,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            num_hyperedges=num_hyperedges,
            dsc3k=dsc3k,
            shortcut=shortcut,
            e1=e1,
            e2=e2,
            context=context,
            channel_adjust=channel_adjust,
        )
        self.fuse = DPRFuseModule(
            c_in=c1,
            channel_adjust=channel_adjust,
            use_p3_detail=use_p3_detail,
            detail_alpha_max=detail_alpha_max,
            p3_down_mode=p3_down_mode,
            p3_fusion_mode=p3_fusion_mode,
            use_p5_semantic=use_p5_semantic,
            semantic_alpha_max=semantic_alpha_max,
        )


class DownsampleConv(nn.Module):
    """
    A simple downsampling block with optional channel adjustment.

    This module uses average pooling to reduce the spatial dimensions (H, W) by a factor of 2. It can
    optionally include a 1x1 convolution to adjust the number of channels, typically doubling them.

    Attributes:
        in_channels (int): The number of input channels.
        channel_adjust (bool, optional): If True, a 1x1 convolution doubles the channel dimension. Defaults to True.

    Methods:
        forward: Performs the downsampling and optional channel adjustment.

    Examples:
        >>> import torch
        >>> model = DownsampleConv(in_channels=64, channel_adjust=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """

    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
        else:
            self.channel_adjust = nn.Identity()

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))


class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out


class RATunnel(nn.Module):
    """Reliability- and amplitude-aware residual tunnel with a zero-start injection path."""

    def __init__(
        self,
        c_original,
        c_enhanced,
        mode="p4",
        gamma_max=None,
        reduction=4,
        use_amplitude=True,
        use_channel=True,
        use_spatial=None,
        ratio_min=0.5,
        ratio_max=2.0,
        eps=1e-6,
    ):
        super().__init__()
        if mode not in {"p3", "p4", "p5"}:
            raise ValueError(f"Unsupported RATunnel mode: {mode}. Expected 'p3', 'p4', or 'p5'.")
        if int(reduction) <= 0:
            raise ValueError("RATunnel reduction must be a positive integer.")
        if ratio_min <= 0.0 or ratio_max < ratio_min:
            raise ValueError("RATunnel requires 0 < ratio_min <= ratio_max.")

        self.c_original = int(c_original)
        self.c_enhanced = int(c_enhanced)
        self.mode = str(mode)
        self.use_amplitude = bool(use_amplitude)
        self.use_channel = bool(use_channel)
        self.use_spatial = (self.mode == "p4") if use_spatial is None else bool(use_spatial)
        self.ratio_min, self.ratio_max, self.eps = float(ratio_min), float(ratio_max), float(eps)
        self.gamma_max = float(0.15 if self.mode == "p4" else 0.10) if gamma_max is None else float(gamma_max)

        self.enhanced_proj = (
            nn.Identity() if self.c_enhanced == self.c_original else Conv(self.c_enhanced, self.c_original, 1, 1)
        )
        hidden = max(self.c_original // int(reduction), 16)
        if self.use_channel:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.c_original * 4, hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, self.c_original, 1, bias=True),
            )
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.zeros_(self.channel_gate[-1].bias)
        else:
            self.channel_gate = None
        if self.use_spatial:
            spatial_hidden = max(hidden // 2, 8)
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(3, spatial_hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(spatial_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(spatial_hidden, 1, 1, bias=True),
            )
            nn.init.zeros_(self.spatial_gate[-1].weight)
            nn.init.zeros_(self.spatial_gate[-1].bias)
        else:
            self.spatial_gate = None
        self.gamma_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @staticmethod
    def _spatial_standardize(x, eps):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)

    def _align_amplitude(self, original, enhanced):
        if not self.use_amplitude:
            return enhanced
        rms_original = original.float().pow(2).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        rms_enhanced = enhanced.float().pow(2).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        ratio = (rms_original / rms_enhanced).detach().clamp(self.ratio_min, self.ratio_max)
        return enhanced * ratio.to(dtype=enhanced.dtype)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("RATunnel expects input as [original_feature, enhanced_feature].")
        original, enhanced = x
        if original.ndim != 4 or enhanced.ndim != 4:
            raise ValueError("RATunnel inputs must be 4D NCHW tensors.")
        if original.shape[-2:] != enhanced.shape[-2:]:
            enhanced = F.interpolate(enhanced, size=original.shape[-2:], mode="nearest")
        enhanced = self.enhanced_proj(enhanced)
        enhanced_aligned = self._align_amplitude(original, enhanced)
        norm_original = self._spatial_standardize(original, self.eps)
        norm_enhanced = self._spatial_standardize(enhanced_aligned, self.eps)
        diff = (norm_original - norm_enhanced).abs()
        local = (original - F.avg_pool2d(original, kernel_size=3, stride=1, padding=1)).abs()

        if self.channel_gate is None:
            channel_reliability = original.new_ones((original.shape[0], self.c_original, 1, 1))
        else:
            channel_reliability = torch.sigmoid(self.channel_gate(torch.cat((original, enhanced_aligned, diff, local), dim=1)))
        if self.spatial_gate is None:
            spatial_reliability = original.new_ones((original.shape[0], 1, original.shape[2], original.shape[3]))
        else:
            descriptor = torch.cat((diff.mean(1, keepdim=True), diff.amax(1, keepdim=True), local.mean(1, keepdim=True)), dim=1)
            spatial_reliability = torch.sigmoid(self.spatial_gate(descriptor))
            if self.mode == "p4":
                spatial_reliability = F.avg_pool2d(spatial_reliability, kernel_size=3, stride=1, padding=1)

        gamma = self.gamma_max * torch.tanh(self.gamma_raw)
        return original + gamma * channel_reliability * spatial_reliability * enhanced_aligned


class UWFeatCalib(nn.Module):
    """Detector-oriented underwater feature calibration for shallow feature maps."""

    def __init__(self, c1, c2=None, r=4):
        super().__init__()
        c2 = c1 if c2 is None else int(c2)
        hidden = max(c2 // int(r), 16)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.norm = nn.InstanceNorm2d(c2, affine=False)
        self.detail_dw = Conv(c2, c2, 3, 1, g=c2)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.pa = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.Conv2d(c2, c2, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.out = Conv(c2 * 3, c2, 1, 1)

    def forward(self, x):
        x = self.proj(x)
        x_norm = self.norm(x)
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = self.detail_dw(x - low)
        cast = x_norm * self.ca(x_norm)
        detail = detail * self.pa(torch.abs(detail))
        y = x + self.alpha.tanh() * cast + self.beta.tanh() * detail
        return self.out(torch.cat((x, y, detail), dim=1))


class CtrlP2Fuse(nn.Module):
    """Controlled P2 fusion where deep semantics gate shallow detail injection."""

    def __init__(self, c_deep, c_skip, c_out, hidden=None):
        super().__init__()
        hidden = max(int(c_out if hidden is None else hidden), 32)
        self.deep_proj = Conv(c_deep, hidden, 1, 1)
        self.skip_proj = Conv(c_skip, hidden, 1, 1)
        self.detail = nn.Sequential(
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1),
        )
        self.gate = nn.Sequential(
            Conv(hidden * 4, hidden, 1, 1),
            nn.Conv2d(hidden, hidden, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(hidden * 3, c_out, 1, 1)
        self.detail_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"CtrlP2Fuse expects [deep, skip], got {type(x).__name__}.")
        deep, skip = x
        if deep.shape[-2:] != skip.shape[-2:]:
            deep = F.interpolate(deep, size=skip.shape[-2:], mode="nearest")
        deep_embed = self.deep_proj(deep)
        skip_embed = self.skip_proj(skip)
        blur = F.avg_pool2d(skip_embed, kernel_size=3, stride=1, padding=1)
        detail = self.detail(skip_embed - blur)
        diff = torch.abs(deep_embed - skip_embed)
        gate = self.gate(torch.cat((deep_embed, skip_embed, diff, detail), dim=1))
        detail = self.detail_scale.tanh() * detail
        fused = skip_embed + gate * (deep_embed + detail - skip_embed)
        return self.out(torch.cat((fused, skip_embed, detail), dim=1))


class AADown(nn.Module):
    """Local anti-aliased downsampling for the UFCR P2-to-P3 return path."""

    def __init__(self, c1, c2):
        super().__init__()
        kernel = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        kernel /= kernel.sum()
        self.register_buffer("aa_kernel", kernel[None, None], persistent=False)
        self.mix = Conv(c1, c1, 3, 1, g=c1)
        self.proj = Conv(c1, c2, 1, 1)

    def forward(self, x):
        c = x.shape[1]
        weight = self.aa_kernel.repeat(c, 1, 1, 1).to(dtype=x.dtype)
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        x = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        return self.proj(self.mix(x))


class ScaleRebalance(nn.Module):
    """Recover P3 semantic dominance after adding and returning the P2 detail branch."""

    def __init__(self, c_low, c_cur, c_out, hidden=None):
        super().__init__()
        hidden = max(int(c_out if hidden is None else hidden), 32)
        self.low_proj = Conv(c_low, hidden, 1, 1)
        self.cur_proj = Conv(c_cur, hidden, 1, 1)
        mid = max(hidden // 4, 16)
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 2, mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 2, 1, bias=True),
        )
        self.edge = nn.Sequential(
            Conv(hidden * 2, hidden, 3, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
        )
        self.out = Conv(hidden * 2, c_out, 1, 1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"ScaleRebalance expects [low, current], got {type(x).__name__}.")
        low, cur = x
        if low.shape[-2:] != cur.shape[-2:]:
            low = F.interpolate(low, size=cur.shape[-2:], mode="nearest")
        low = self.low_proj(low)
        cur = self.cur_proj(cur)
        logits = self.weight(torch.cat((low, cur), dim=1))
        weights = torch.softmax(logits.view(logits.shape[0], 2, 1, 1, 1), dim=1)
        fused = weights[:, 0] * low + weights[:, 1] * cur
        residual = self.edge(torch.cat((torch.abs(low - cur), cur), dim=1))
        return self.out(torch.cat((fused, residual), dim=1))


class WTFeatCalib(nn.Module):
    """Water-type aware feature calibration with conservative FiLM and local detail residuals."""

    def __init__(self, c1, c2=None, token_dim=16, reduction=4):
        super().__init__()
        c2 = c1 if c2 is None else int(c2)
        token_dim = max(int(token_dim), 4)
        hidden = max(c2 // int(reduction), token_dim, 8)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.token = nn.Sequential(
            nn.Conv2d(c2 * 5, hidden, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, token_dim, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
        )
        self.film = nn.Conv2d(token_dim, c2 * 2, 1, 1, 0, bias=True)
        self.detail_dw = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.Conv2d(c2, c2, 1, 1, 0, bias=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x):
        x = self.proj(x)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True).add(1e-6).sqrt()
        max_v = torch.amax(x, dim=(2, 3), keepdim=True)
        min_v = torch.amin(x, dim=(2, 3), keepdim=True)
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        contrast = (x - low).abs().mean(dim=(2, 3), keepdim=True)
        token = self.token(torch.cat((mean, std, max_v, min_v, contrast), dim=1))
        gamma, beta = self.film(token).chunk(2, dim=1)
        film = x * (0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        detail = self.detail_dw(x - low)
        return x + film + torch.tanh(self.alpha) * detail


class EdgeConfidenceP2Fuse(nn.Module):
    """Fuse P2 detail with deep semantics through edge confidence and cross-scale consistency gates."""

    def __init__(self, c_deep, c_skip, c_out, hidden=None):
        super().__init__()
        hidden = max(int(c_out if hidden is None else hidden), 32)
        self.deep_proj = Conv(c_deep, hidden, 1, 1)
        self.skip_proj = Conv(c_skip, hidden, 1, 1)
        self.edge_branch = nn.Sequential(
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1),
        )
        self.edge_logits = nn.Conv2d(hidden, hidden, 1, 1, 0, bias=True)
        self.conf_branch = nn.Sequential(
            Conv(hidden * 4, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
        )
        self.conf_logits = nn.Conv2d(hidden, hidden, 1, 1, 0, bias=True)
        self.detail_scale = nn.Parameter(torch.zeros(1))
        self.out = Conv(hidden * 4, c_out, 1, 1)
        nn.init.constant_(self.edge_logits.bias, -1.0)
        nn.init.constant_(self.conf_logits.bias, -1.0)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"EdgeConfidenceP2Fuse expects [deep, skip], got {type(x).__name__}.")
        deep, skip = x
        if deep.shape[-2:] != skip.shape[-2:]:
            deep = F.interpolate(deep, size=skip.shape[-2:], mode="nearest")
        d = self.deep_proj(deep)
        s = self.skip_proj(skip)
        blur = F.avg_pool2d(s, kernel_size=3, stride=1, padding=1)
        hf = s - blur
        edge_feat = self.edge_branch(torch.abs(hf))
        edge_gate = torch.sigmoid(self.edge_logits(edge_feat))
        diff = torch.abs(d - s)
        local_var = torch.abs(hf)
        conf_feat = self.conf_branch(torch.cat((d, s, diff, local_var), dim=1))
        conf_gate = torch.sigmoid(self.conf_logits(conf_feat))
        gate = edge_gate * conf_gate
        candidate = d + torch.tanh(self.detail_scale) * edge_feat
        fused = s + gate * (candidate - s)
        return self.out(torch.cat((fused, s, edge_feat, diff), dim=1))


class AARUp(nn.Module):
    """Anti-aliased reassembly upsampling for the P3-to-P2 path."""

    def __init__(self, c1, c2=None, scale=2):
        super().__init__()
        c2 = c1 if c2 is None else int(c2)
        self.scale = int(scale)
        kernel = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        kernel /= kernel.sum()
        self.register_buffer("aa_kernel", kernel[None, None], persistent=False)
        self.mix = Conv(c1, c1, 3, 1, g=c1)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        c = x.shape[1]
        weight = self.aa_kernel.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        x = F.conv2d(x, weight, stride=1, padding=0, groups=c)
        return self.proj(self.mix(x))


class AARDown(nn.Module):
    """Paired anti-aliased downsampling with weak local-detail residual injection."""

    def __init__(self, c1, c2):
        super().__init__()
        kernel = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        kernel /= kernel.sum()
        self.register_buffer("aa_kernel", kernel[None, None], persistent=False)
        self.mix = Conv(c1, c1, 3, 1, g=c1)
        self.proj = Conv(c1, c2, 1, 1)
        self.detail_proj = Conv(c1, c2, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1))

    def _blur_down(self, x):
        c = x.shape[1]
        weight = self.aa_kernel.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, weight, stride=2, padding=0, groups=c)

    def forward(self, x):
        low = self._blur_down(x)
        out = self.proj(self.mix(low))
        base = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = self.detail_proj(self._blur_down(x - base))
        return out + torch.tanh(self.beta) * detail


class AARUpLite(nn.Module):
    """Lightweight anti-aliased adaptive reassembly upsampling for YOLOv13 three-scale necks."""

    def __init__(self, c1, scale=2, r=4):
        super().__init__()
        self.scale = int(scale)
        hidden = max(int(c1) // int(r), 16)
        self.lowpass = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c1 * 2, hidden, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        low = self.lowpass(y)
        high = y - F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        gate = self.gate(torch.cat((low, high.abs()), dim=1))
        return y + self.alpha.tanh() * gate * (low - y) + self.beta.tanh() * gate * high


class FAARUp(nn.Module):
    """Scale-specific frequency-aware anti-aliased upsampling for YOLOv13 three-scale necks."""

    def __init__(self, c1, scale=2, mode="semantic", groups=4):
        super().__init__()
        if mode not in {"semantic", "detail"}:
            raise ValueError(f"Unsupported FAARUp mode: {mode}. Expected 'semantic' or 'detail'.")
        c1 = int(c1)
        groups = max(1, min(int(groups), c1))
        if c1 % groups != 0:
            groups = 1
        self.c1 = c1
        self.scale = int(scale)
        self.mode = mode
        self.lowpass = nn.Sequential(
            Conv(c1, c1, 3, 1, g=c1),
            Conv(c1, c1, 1, 1),
        )
        self.highpass = nn.Sequential(
            Conv(c1, c1, 3, 1, g=c1),
            Conv(c1, c1, 1, 1),
        )
        self.hp_gate = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.Conv2d(c1, c1, 1, 1, 0, groups=groups, bias=True),
            nn.Sigmoid(),
        )
        if mode == "semantic":
            self.lp_strength = 1.0
            self.hp_strength = 0.25
            init_alpha = 0.02
        else:
            self.lp_strength = 0.75
            self.hp_strength = 1.0
            init_alpha = 0.05
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        blur = F.avg_pool2d(up, kernel_size=3, stride=1, padding=1)
        low = self.lowpass(up)
        lp = 0.5 * low + 0.5 * blur
        high = up - blur
        hp = self.highpass(high)
        gate = self.hp_gate(high.abs())
        if self.mode == "semantic":
            return lp + self.alpha.tanh() * self.hp_strength * gate * hp
        base = self.lp_strength * lp + (1.0 - self.lp_strength) * up
        return base + self.alpha.tanh() * self.hp_strength * gate * hp


class DCRAUp(nn.Module):
    """
    Discrete Correlation Residual Alignment Upsampling.

    The exact nearest-neighbor upsample is retained as the base path. The lateral feature is used only to query a
    replicate-padded discrete neighborhood sampled from the deep feature, and a zero-initialized projection adds the
    resulting residual correction.
    """

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

        if self.c_deep <= 0:
            raise ValueError(f"c_deep must be positive, got {self.c_deep}.")
        if self.c_lateral <= 0:
            raise ValueError(f"c_lateral must be positive, got {self.c_lateral}.")
        if self.scale <= 1:
            raise ValueError(f"scale must be greater than 1, got {self.scale}.")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be an odd integer greater than or equal "
                f"to 3, got {self.kernel_size}."
            )
        if self.reduction <= 0:
            raise ValueError(f"reduction must be positive, got {self.reduction}.")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}.")
        if int(residual_groups) <= 0:
            raise ValueError(f"residual_groups must be positive, got {residual_groups}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

        self.num_candidates = self.kernel_size**2
        self.embed_dim = max(16, min(64, min(self.c_deep, self.c_lateral) // self.reduction))
        self.residual_groups = math.gcd(self.c_deep, int(residual_groups))

        with torch.random.fork_rng(devices=[], enabled=True):
            # Deterministic module-local initialization avoids reusing the exact random stream consumed by the next
            # original layer while fork_rng still restores the global CPU RNG state.
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

        # Exact D0-equivalent initialization; zeros_ does not consume RNG.
        nn.init.zeros_(self.residual_out.weight)

    def _validate_inputs(self, deep, lateral):
        """Validate feature shape, channels, device, dtype, and configured spatial scale."""
        if deep.ndim != 4 or lateral.ndim != 4:
            raise ValueError(
                "DCRAUp expects 4D NCHW feature maps, got "
                f"deep={tuple(deep.shape)}, lateral={tuple(lateral.shape)}."
            )
        if deep.shape[0] != lateral.shape[0]:
            raise ValueError(f"Batch-size mismatch: deep={deep.shape[0]}, lateral={lateral.shape[0]}.")
        if deep.shape[1] != self.c_deep:
            raise ValueError(f"Deep-channel mismatch: got {deep.shape[1]}, expected {self.c_deep}.")
        if lateral.shape[1] != self.c_lateral:
            raise ValueError(f"Lateral-channel mismatch: got {lateral.shape[1]}, expected {self.c_lateral}.")
        if deep.device != lateral.device:
            raise ValueError(f"Device mismatch: deep={deep.device}, lateral={lateral.device}.")
        if deep.dtype != lateral.dtype:
            raise ValueError(f"Dtype mismatch: deep={deep.dtype}, lateral={lateral.dtype}.")

        expected_size = (deep.shape[-2] * self.scale, deep.shape[-1] * self.scale)
        actual_size = tuple(lateral.shape[-2:])
        if self.strict_scale and actual_size != expected_size:
            raise ValueError(
                "DCRAUp spatial scale mismatch: "
                f"deep={tuple(deep.shape[-2:])}, lateral={actual_size}, scale={self.scale}, "
                f"expected lateral={expected_size}."
            )

    def _extract_local_patches(self, x):
        """Return replicate-padded local patches as ``[B, C, K*K, H, W]``."""
        if x.ndim != 4:
            raise ValueError(f"Patch extraction expects a 4D tensor, got shape {tuple(x.shape)}.")
        batch, channels, height, width = x.shape
        padding = self.kernel_size // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        patches = F.unfold(x, kernel_size=self.kernel_size, dilation=1, padding=0, stride=1)
        expected_locations = height * width
        if patches.shape[-1] != expected_locations:
            raise RuntimeError(
                f"Unexpected patch count: got {patches.shape[-1]}, expected {expected_locations}."
            )
        return patches.reshape(batch, channels, self.num_candidates, height, width)

    @staticmethod
    def _split_phases(x, low_size, scale):
        """Convert ``[B, C, H*s, W*s]`` to ``[B, C, s, s, H, W]``."""
        if x.ndim != 4:
            raise ValueError(f"Phase split expects 4D input, got {tuple(x.shape)}.")
        low_h, low_w = int(low_size[0]), int(low_size[1])
        scale = int(scale)
        expected = (low_h * scale, low_w * scale)
        if tuple(x.shape[-2:]) != expected:
            raise ValueError(
                f"Phase split shape mismatch: input={tuple(x.shape[-2:])}, expected={expected}."
            )
        batch, channels = x.shape[:2]
        return (
            x.reshape(batch, channels, low_h, scale, low_w, scale)
            .permute(0, 1, 3, 5, 2, 4)
            .contiguous()
        )

    @staticmethod
    def _merge_phases(x):
        """Convert ``[B, C, s_h, s_w, H, W]`` to ``[B, C, H*s_h, W*s_w]``."""
        if x.ndim != 6:
            raise ValueError(f"Phase merge expects 6D input, got {tuple(x.shape)}.")
        batch, channels, scale_h, scale_w, low_h, low_w = x.shape
        return (
            x.permute(0, 1, 4, 2, 5, 3)
            .contiguous()
            .reshape(batch, channels, low_h * scale_h, low_w * scale_w)
        )

    @staticmethod
    def _resize_patch_tensor(patches, target_size):
        """Explicit high-resolution candidate fallback used only for non-exact scale."""
        if patches.ndim != 5:
            raise ValueError(f"Patch resize expects a 5D tensor, got shape {tuple(patches.shape)}.")
        batch, channels, candidates, height, width = patches.shape
        target_h, target_w = (int(target_size[0]), int(target_size[1]))
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid target size {(target_h, target_w)}.")
        patches_4d = patches.reshape(batch, channels * candidates, height, width)
        if (height, width) != (target_h, target_w):
            patches_4d = F.interpolate(patches_4d, size=(target_h, target_w), mode="nearest")
        return patches_4d.reshape(batch, channels, candidates, target_h, target_w)

    def _phase_correlate_and_reassemble(self, query, key_patches, value_patches):
        """Memory-efficient exact-scale correlation and value reassembly."""
        # CUDA autocast treats einsum as FP16-eligible even when its arguments were explicitly cast with .float().
        # Disable autocast around the complete correlation path so correlation, softmax, and aggregation are truly
        # FP32 and their small second-step gradients cannot underflow.
        with torch.autocast(device_type=query.device.type, enabled=False):
            low_size = key_patches.shape[-2:]
            query_fp32 = F.normalize(query.float(), p=2, dim=1, eps=self.eps)
            query_phases = self._split_phases(query_fp32, low_size, self.scale)
            keys_fp32 = F.normalize(key_patches.float(), p=2, dim=1, eps=self.eps)

            logits_phases = (
                query_phases.unsqueeze(2) * keys_fp32.unsqueeze(3).unsqueeze(3)
            ).sum(dim=1)
            weights_phases = torch.softmax(logits_phases / self.temperature, dim=1)
            reassembled_phases = torch.einsum(
                "bckhw,bkijhw->bcijhw",
                value_patches.float(),
                weights_phases,
            )
            return self._merge_phases(reassembled_phases), self._merge_phases(weights_phases)

    def _fallback_correlate_and_reassemble(self, query, key_patches, value_patches, target_size):
        """General-size fallback used only when ``strict_scale=False``."""
        with torch.autocast(device_type=query.device.type, enabled=False):
            key_patches_high = self._resize_patch_tensor(key_patches.float(), target_size)
            query_fp32 = F.normalize(query.float(), p=2, dim=1, eps=self.eps)
            keys_fp32 = F.normalize(key_patches_high, p=2, dim=1, eps=self.eps)
            logits = (query_fp32.unsqueeze(2) * keys_fp32).sum(dim=1)
            weights = torch.softmax(logits / self.temperature, dim=1)
            value_patches_high = self._resize_patch_tensor(value_patches.float(), target_size)
            reassembled = torch.einsum("bckhw,bkhw->bchw", value_patches_high, weights)
            return reassembled, weights

    @staticmethod
    def _project_for_fp32_path(projection, feature):
        """Project in FP32 during training, while remaining compatible with validator-created FP16 models."""
        if projection.weight.dtype == torch.float32:
            with torch.autocast(device_type=feature.device.type, enabled=False):
                return projection(feature.float())
        # Ultralytics validation may call model.half() outside autocast. Use the stored parameter dtype for the
        # convolution, then promote its result before correlation. This branch runs without gradients in validation.
        return projection(feature.to(dtype=projection.weight.dtype)).float()

    def _compute_alignment(self, deep, lateral):
        """Compute the nearest base, discrete local residual, FP32 weights, and FP32 confidence."""
        self._validate_inputs(deep, lateral)
        target_size = tuple(lateral.shape[-2:])
        base = F.interpolate(deep, size=target_size, mode="nearest")

        # Projection outputs and their backward paths remain FP32; casting an autocast FP16 result afterwards can
        # underflow the required second-step key/query gradients before GradScaler is involved.
        key_low = self._project_for_fp32_path(self.key_proj, deep)
        query = (
            self._project_for_fp32_path(self.query_proj, lateral)
            if self.use_lateral_guidance
            else F.interpolate(key_low, size=target_size, mode="nearest")
        )
        key_patches = self._extract_local_patches(key_low)
        value_patches = self._extract_local_patches(deep)

        exact_scale = target_size == (
            deep.shape[-2] * self.scale,
            deep.shape[-1] * self.scale,
        )
        if exact_scale:
            reassembled_fp32, weights = self._phase_correlate_and_reassemble(
                query,
                key_patches,
                value_patches,
            )
        else:
            reassembled_fp32, weights = self._fallback_correlate_and_reassemble(
                query,
                key_patches,
                value_patches,
                target_size,
            )
        reassembled = reassembled_fp32.to(dtype=deep.dtype)

        if self.use_entropy:
            entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=1, keepdim=True)
            confidence = (1.0 - entropy / math.log(float(self.num_candidates))).clamp(0.0, 1.0)
        else:
            confidence = torch.ones(
                (deep.shape[0], 1, target_size[0], target_size[1]),
                device=deep.device,
                dtype=torch.float32,
            )
        if self.detach_confidence:
            confidence = confidence.detach()

        residual = (reassembled - base) * confidence.to(dtype=deep.dtype)
        return base, residual, weights, confidence

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DCRAUp expects input as [deep_feature, lateral_feature].")
        deep, lateral = x
        base, residual, _, _ = self._compute_alignment(deep, lateral)
        # Keep the zero-initialized residual projection in FP32. With a tiny first SGD update, autocast FP16
        # convolution can underflow its input gradient to zero on the required second backward pass.
        if self.residual_out.weight.dtype == torch.float32:
            with torch.autocast(device_type=deep.device.type, enabled=False):
                correction = self.residual_out(residual.float())
                output = base.float() + correction
        else:
            correction = self.residual_out(residual.to(dtype=self.residual_out.weight.dtype)).float()
            # A validator-created model.half() executes outside autocast, so the next original neck layer requires
            # a half output. Accumulate the correction in FP32, then restore the model activation dtype.
            output = (base.float() + correction).to(dtype=deep.dtype)
        expected_shape = (deep.shape[0], self.c_deep, lateral.shape[-2], lateral.shape[-1])
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"DCRAUp output-shape mismatch: got {tuple(output.shape)}, expected {expected_shape}."
            )
        return output


class WSDRFuse(nn.Module):
    """Wavelet Semantic-Detail Recomposition fusion for the P5-to-P4 neck node."""

    def __init__(
        self,
        c_deep,
        c_lat,
        g_max=0.25,
        reduction=4,
        adaptive=True,
        fixed_gate=0.125,
        use_hf_energy=True,
        decomposition="haar",
        hf_reweight=False,
        faar_groups=4,
        eps=1e-4,
    ):
        super().__init__()
        self.c_deep = int(c_deep)
        self.c_lat = int(c_lat)
        self.g_max = float(g_max)
        self.reduction = int(reduction)
        self.adaptive = bool(adaptive)
        self.fixed_gate = float(fixed_gate)
        self.use_hf_energy = bool(use_hf_energy)
        self.decomposition = str(decomposition).lower()
        self.hf_reweight = bool(hf_reweight)
        self.faar_groups = int(faar_groups)
        self.eps = float(eps)

        if self.c_deep <= 0:
            raise ValueError(f"c_deep must be positive, got {self.c_deep}.")
        if self.c_lat <= 0:
            raise ValueError(f"c_lat must be positive, got {self.c_lat}.")
        if not (0.0 < self.g_max <= 1.0):
            raise ValueError(f"g_max must satisfy 0 < g_max <= 1, got {self.g_max}.")
        if not (0.0 <= self.fixed_gate <= self.g_max):
            raise ValueError(
                "fixed_gate must satisfy 0 <= fixed_gate <= g_max, "
                f"got fixed_gate={self.fixed_gate}, g_max={self.g_max}."
            )
        if self.reduction <= 0:
            raise ValueError(f"reduction must be positive, got {self.reduction}.")
        if self.decomposition not in {"haar", "avgpool"}:
            raise ValueError(
                "decomposition must be 'haar' or 'avgpool', "
                f"got {self.decomposition!r}."
            )
        if self.hf_reweight and self.decomposition != "haar":
            raise ValueError("hf_reweight=True is defined only for decomposition='haar'.")
        if self.faar_groups <= 0:
            raise ValueError(f"faar_groups must be positive, got {self.faar_groups}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

        self.deep_up = FAARUp(
            self.c_deep,
            scale=2,
            mode="semantic",
            groups=self.faar_groups,
        )
        self.deep_proj = Conv(self.c_deep, self.c_lat, 1, 1, act=False)
        hidden = max(self.c_lat // self.reduction, 16)

        if self.adaptive:
            gate_in_channels = 3 * self.c_lat + (1 if self.use_hf_energy else 0)
            self.gate_head = nn.Sequential(
                nn.Conv2d(gate_in_channels, hidden, 1, 1, 0, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
            )
            nn.init.zeros_(self.gate_head[-1].bias)
        else:
            self.gate_head = None

        if self.hf_reweight:
            self.hf_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(3 * self.c_lat, hidden, 1, 1, 0, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 3 * self.c_lat, 1, 1, 0, bias=True),
            )
            nn.init.zeros_(self.hf_gate[-1].bias)
        else:
            self.hf_gate = None

    @staticmethod
    def _pad_to_even(x):
        """Replicate-pad only the right and bottom borders when needed."""
        h, w = x.shape[-2:]
        pad_h, pad_w = h & 1, w & 1
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)

    @classmethod
    def _haar_decompose(cls, x):
        """Apply an orthonormal 2D Haar decomposition."""
        x, original_size = cls._pad_to_even(x)
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (-x00 - x01 + x10 + x11)
        hl = 0.5 * (-x00 + x01 - x10 + x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, (lh, hl, hh), original_size

    @staticmethod
    def _haar_reconstruct(ll, details, original_size):
        """Invert the Haar decomposition and crop to the original size."""
        if not isinstance(details, (list, tuple)) or len(details) != 3:
            raise ValueError("Haar reconstruction requires exactly (lh, hl, hh).")
        lh, hl, hh = details
        expected_shape = ll.shape
        if lh.shape != expected_shape or hl.shape != expected_shape or hh.shape != expected_shape:
            raise ValueError(
                "All Haar subbands must have identical shapes, got "
                f"LL={tuple(ll.shape)}, LH={tuple(lh.shape)}, "
                f"HL={tuple(hl.shape)}, HH={tuple(hh.shape)}."
            )
        x00 = 0.5 * (ll - lh - hl + hh)
        x01 = 0.5 * (ll - lh + hl - hh)
        x10 = 0.5 * (ll + lh - hl - hh)
        x11 = 0.5 * (ll + lh + hl + hh)
        b, c, h, w = ll.shape
        x = torch.stack((x00, x01, x10, x11), dim=-1).reshape(b, c, h, w, 2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h * 2, w * 2)
        original_h, original_w = original_size
        return x[..., :original_h, :original_w]

    @staticmethod
    def _avgpool_decompose(x):
        """Build an exactly reconstructible AvgPool-plus-residual decomposition."""
        h, w = x.shape[-2:]
        low = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)
        base = F.interpolate(low, size=(h, w), mode="nearest")
        return low, (x - base,), (h, w)

    @staticmethod
    def _avgpool_reconstruct(low, details, original_size):
        if not isinstance(details, (list, tuple)) or len(details) != 1:
            raise ValueError("AvgPool reconstruction requires one residual tensor.")
        residual = details[0]
        base = F.interpolate(low, size=original_size, mode="nearest")
        if base.shape != residual.shape:
            raise ValueError(
                "AvgPool reconstruction shape mismatch: "
                f"base={tuple(base.shape)}, residual={tuple(residual.shape)}."
            )
        return base + residual

    @staticmethod
    def _spatial_rms_norm(x, eps):
        """Apply per-sample, per-channel spatial RMS normalization in FP32."""
        rms = x.float().square().mean(dim=(2, 3), keepdim=True).add(float(eps)).sqrt()
        return x / rms.to(dtype=x.dtype)

    def _decompose(self, x):
        return self._haar_decompose(x) if self.decomposition == "haar" else self._avgpool_decompose(x)

    def _reconstruct(self, low, details, original_size):
        if self.decomposition == "haar":
            return self._haar_reconstruct(low, details, original_size)
        return self._avgpool_reconstruct(low, details, original_size)

    def _detail_energy(self, details, low_size):
        """Build one normalized spatial detail-energy map at low resolution."""
        if self.decomposition == "haar":
            lh, hl, hh = details
            energy = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True) / 3.0
        else:
            energy = details[0].abs().mean(dim=1, keepdim=True)
            energy = F.adaptive_avg_pool2d(energy, low_size)
        return self._spatial_rms_norm(energy, self.eps)

    def _compute_gate(self, semantic, low, details):
        """Return a bounded spatial gate with shape [B, 1, H_low, W_low]."""
        if semantic.shape != low.shape:
            raise ValueError(
                "semantic and low must have identical shapes before gate computation, "
                f"got semantic={tuple(semantic.shape)}, low={tuple(low.shape)}."
            )
        if not self.adaptive:
            return low.new_full((low.shape[0], 1, low.shape[2], low.shape[3]), self.fixed_gate)
        semantic_norm = self._spatial_rms_norm(semantic, self.eps)
        low_norm = self._spatial_rms_norm(low, self.eps)
        gate_inputs = [semantic_norm, low_norm, (semantic_norm - low_norm).abs()]
        if self.use_hf_energy:
            gate_inputs.append(self._detail_energy(details, low.shape[-2:]))
        return self.g_max * torch.sigmoid(self.gate_head(torch.cat(gate_inputs, dim=1)))

    def _reweight_details(self, details):
        """Optionally apply the W6-only high-frequency recalibration."""
        if not self.hf_reweight:
            return details
        lh, hl, hh = details
        logits = self.hf_gate(torch.cat((lh.abs(), hl.abs(), hh.abs()), dim=1))
        scales = 1.0 + 0.25 * torch.tanh(logits.float())
        scales = scales.to(dtype=lh.dtype).chunk(3, dim=1)
        return tuple(detail * scale for detail, scale in zip((lh, hl, hh), scales))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("WSDRFuse expects [deep_feature, lateral_feature].")
        deep, lateral = x
        if deep.ndim != 4 or lateral.ndim != 4:
            raise ValueError(
                "WSDRFuse expects 4D feature maps, got "
                f"deep={tuple(deep.shape)}, lateral={tuple(lateral.shape)}."
            )
        if deep.shape[0] != lateral.shape[0]:
            raise ValueError(f"Batch-size mismatch: deep B={deep.shape[0]}, lateral B={lateral.shape[0]}.")
        if deep.shape[1] != self.c_deep:
            raise ValueError(f"Deep channel mismatch: got {deep.shape[1]}, expected {self.c_deep}.")
        if lateral.shape[1] != self.c_lat:
            raise ValueError(f"Lateral channel mismatch: got {lateral.shape[1]}, expected {self.c_lat}.")

        deep_up = self.deep_up(deep)
        if deep_up.shape[-2:] != lateral.shape[-2:]:
            deep_up = F.interpolate(deep_up, size=lateral.shape[-2:], mode="nearest")
        low, details, original_size = self._decompose(lateral)
        semantic = self.deep_proj(deep)
        if semantic.shape[-2:] != low.shape[-2:]:
            semantic = F.interpolate(semantic, size=low.shape[-2:], mode="bilinear", align_corners=False)
        gate = self._compute_gate(semantic, low, details)
        low_new = low + gate * (semantic - low)
        lateral_new = self._reconstruct(low_new, self._reweight_details(details), original_size)
        if lateral_new.shape != lateral.shape:
            raise RuntimeError(
                "WSDRFuse reconstruction changed the lateral shape: "
                f"input={tuple(lateral.shape)}, output={tuple(lateral_new.shape)}."
            )
        if deep_up.shape[-2:] != lateral_new.shape[-2:]:
            raise RuntimeError(
                "WSDRFuse concatenation spatial mismatch: "
                f"deep_up={tuple(deep_up.shape)}, lateral_new={tuple(lateral_new.shape)}."
            )
        return torch.cat((deep_up, lateral_new), dim=1)


class SCAFFuse(nn.Module):
    """
    Scale-Consistent Adaptive Fusion.

    This module replaces a Concat node after FAARUp. It performs bounded multiplicative
    recalibration on the upsampled deep feature and the lateral feature, then concatenates
    them along the channel dimension.
    """

    def __init__(
        self,
        c_up,
        c_lat,
        mode="detail",
        alpha=0.15,
        reduction=4,
        use_consistency=True,
        channel_only=False,
        eps=1e-4,
    ):
        super().__init__()
        if mode not in {"semantic", "detail"}:
            raise ValueError(f"Unsupported SCAFFuse mode: {mode}. Expected 'semantic' or 'detail'.")
        self.c_up = int(c_up)
        self.c_lat = int(c_lat)
        self.mode = mode
        self.use_consistency = bool(use_consistency)
        self.channel_only = bool(channel_only)
        self.eps = float(eps)

        c_mid = max(min(self.c_up, self.c_lat) // int(reduction), 16)
        self.up_proj = Conv(self.c_up, c_mid, 1, 1)
        self.lat_proj = Conv(self.c_lat, c_mid, 1, 1)
        self.branch_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_mid * 3, c_mid, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 2, 1, 1, 0, bias=True),
        )
        self.consistency_head = (
            None
            if self.channel_only
            else nn.Sequential(
                Conv(c_mid, c_mid, 3, 1),
                nn.Conv2d(c_mid, 1, 1, 1, 0, bias=True),
            )
        )
        self.alpha_base = float(alpha)
        self.alpha_raw = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.branch_mlp[-1].weight)
        nn.init.zeros_(self.branch_mlp[-1].bias)
        if self.consistency_head is not None:
            nn.init.zeros_(self.consistency_head[-1].weight)
            nn.init.zeros_(self.consistency_head[-1].bias)

    @staticmethod
    def _norm_feat(x, eps=1e-4):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SCAFFuse expects input as [x_up, x_lat].")

        x_up, x_lat = x
        if x_up.shape[-2:] != x_lat.shape[-2:]:
            x_up = F.interpolate(x_up, size=x_lat.shape[-2:], mode="nearest")

        p_up = self.up_proj(x_up)
        p_lat = self.lat_proj(x_lat)
        n_up = self._norm_feat(p_up, self.eps)
        n_lat = self._norm_feat(p_lat, self.eps)
        diff = (n_up - n_lat).abs()

        branch_logits = self.branch_mlp(torch.cat((p_up, p_lat, diff), dim=1))
        branch_weight = torch.softmax(branch_logits, dim=1)
        w_up = branch_weight[:, 0:1]
        w_lat = branch_weight[:, 1:2]

        if self.use_consistency:
            consistency = torch.exp(-diff.mean(dim=1, keepdim=True).clamp(min=0.0, max=10.0))
            if self.consistency_head is not None:
                consistency = consistency * torch.sigmoid(self.consistency_head(diff))
            if self.mode == "semantic":
                consistency = F.avg_pool2d(consistency, kernel_size=5, stride=1, padding=2)
            else:
                consistency = F.avg_pool2d(consistency, kernel_size=3, stride=1, padding=1)
        else:
            consistency = torch.ones_like(w_up)

        alpha = self.alpha_base * torch.tanh(self.alpha_raw)
        x_up = x_up * (1.0 + alpha * consistency * w_up)
        x_lat = x_lat * (1.0 + alpha * consistency * w_lat)
        return torch.cat((x_up, x_lat), dim=1)


class RPSCAFFuse(nn.Module):
    """
    Recall-Preserving Scale-Consistent Adaptive Fusion.

    This module is designed for the P5->P4 semantic fusion position. It only
    applies a bounded positive enhancement to the upsampled semantic feature and
    keeps the lateral feature unchanged, so the output remains equivalent to a
    Concat node in channel count: c_up + c_lat.
    """

    def __init__(
        self,
        c_up,
        c_lat,
        mode="semantic",
        alpha=0.05,
        use_consistency=True,
        channel_only=False,
        c_floor=0.35,
        tau=1.0,
        reduction=4,
    ):
        super().__init__()
        if not isinstance(mode, str):
            compact_alpha = float(mode)
            compact_use_consistency = bool(alpha)
            compact_gate_type = str(use_consistency).lower()
            mode = "semantic"
            alpha = compact_alpha
            use_consistency = compact_use_consistency
            channel_only = compact_gate_type in {"channel", "channel_only", "channel-only"}
        if mode not in {"semantic", "detail"}:
            raise ValueError(f"Unsupported RPSCAFFuse mode: {mode}. Expected 'semantic' or 'detail'.")
        self.c_up = int(c_up)
        self.c_lat = int(c_lat)
        self.mode = str(mode)
        self.alpha_max = float(alpha)
        self.use_consistency = bool(use_consistency)
        self.channel_only = bool(channel_only)
        self.c_floor = float(c_floor)
        self.tau = float(tau)
        self.eps = 1e-6

        c_mid = max(min(min(self.c_up, self.c_lat) // int(reduction), 64), 16)
        self.c_mid = c_mid
        self.up_proj = Conv(self.c_up, c_mid, 1, 1)
        self.lat_proj = Conv(self.c_lat, c_mid, 1, 1)

        if self.channel_only:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c_mid * 3, c_mid, 1, 1, 0, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_mid, self.c_up, 1, 1, 0, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Sequential(
                nn.Conv2d(c_mid * 3, c_mid, 3, 1, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_mid, 1, 1, 1, 0, bias=True),
                nn.Sigmoid(),
            )

        self.alpha_raw = nn.Parameter(torch.tensor(-6.0, dtype=torch.float32))

    @staticmethod
    def _safe_norm(x, eps=1e-6):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)

    def _consistency(self, p_up, p_lat, diff):
        if not self.use_consistency:
            if self.channel_only:
                return torch.ones((p_up.shape[0], 1, 1, 1), device=p_up.device, dtype=p_up.dtype)
            return torch.ones((p_up.shape[0], 1, p_up.shape[2], p_up.shape[3]), device=p_up.device, dtype=p_up.dtype)

        c = torch.exp(-self.tau * diff.mean(dim=1, keepdim=True))
        c = self.c_floor + (1.0 - self.c_floor) * c
        if self.channel_only:
            c = F.adaptive_avg_pool2d(c, 1)
        elif self.mode == "semantic":
            c = F.avg_pool2d(c, kernel_size=3, stride=1, padding=1)
        return c.clamp(min=self.c_floor, max=1.0)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("RPSCAFFuse expects input as [x_up, x_lat].")

        x_up, x_lat = x
        if x_up.shape[-2:] != x_lat.shape[-2:]:
            x_up = F.interpolate(x_up, size=x_lat.shape[-2:], mode="nearest")

        p_up = self.up_proj(x_up)
        p_lat = self.lat_proj(x_lat)
        n_up = self._safe_norm(p_up, self.eps)
        n_lat = self._safe_norm(p_lat, self.eps)
        diff = (n_up - n_lat).abs()

        gate = self.gate(torch.cat((p_up, p_lat, diff), dim=1))
        consistency = self._consistency(p_up, p_lat, diff)
        alpha_eff = self.alpha_max * torch.sigmoid(self.alpha_raw)
        x_up = x_up * (1.0 + alpha_eff * consistency * gate)
        return torch.cat((x_up, x_lat), dim=1)


class SFRSCAFFuse(nn.Module):
    """Semantic-frequency routed scale-consistent fusion at the P5-to-P4 node."""

    def __init__(
        self,
        c_up,
        c_lat,
        alpha_cons=0.05,
        alpha_sem=0.05,
        reduction=4,
        route_mode="full",
        fusion_mode="hybrid",
        tau=1.0,
        route_smooth=3,
        eps=1e-4,
    ):
        super().__init__()
        valid_route_modes = {"full", "detail_only", "fixed"}
        valid_fusion_modes = {"hybrid", "consistency_only", "semantic_only"}
        if route_mode not in valid_route_modes:
            raise ValueError(f"Unsupported route_mode={route_mode}, expected one of {valid_route_modes}")
        if fusion_mode not in valid_fusion_modes:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}, expected one of {valid_fusion_modes}")
        if route_smooth < 1 or route_smooth % 2 == 0:
            raise ValueError("route_smooth must be an odd integer >= 1")

        self.c_up = int(c_up)
        self.c_lat = int(c_lat)
        self.alpha_cons_max = float(alpha_cons)
        self.alpha_sem_max = float(alpha_sem)
        self.route_mode = route_mode
        self.fusion_mode = fusion_mode
        self.tau = float(tau)
        self.route_smooth = int(route_smooth)
        self.eps = float(eps)

        c_mid = max(min(self.c_up, self.c_lat) // int(reduction), 16)
        self.up_proj = Conv(self.c_up, c_mid, 1, 1)
        self.lat_proj = Conv(self.c_lat, c_mid, 1, 1)
        self.detail_head = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, 1, 1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 1, 1, 1, 0, bias=True),
        )
        self.branch_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_mid * 3, c_mid, 1, 1, 0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 2, 1, 1, 0, bias=True),
        )
        self.semantic_head = nn.Sequential(
            nn.Conv2d(c_mid * 3, c_mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 1, 1, 1, 0, bias=True),
        )

        self.alpha_cons_raw = nn.Parameter(torch.zeros(1))
        self.alpha_sem_raw = nn.Parameter(torch.full((1,), -6.0))
        nn.init.zeros_(self.detail_head[-1].weight)
        nn.init.zeros_(self.detail_head[-1].bias)
        nn.init.zeros_(self.branch_mlp[-1].weight)
        nn.init.zeros_(self.branch_mlp[-1].bias)
        nn.init.zeros_(self.semantic_head[-1].weight)
        nn.init.zeros_(self.semantic_head[-1].bias)

    @staticmethod
    def _spatial_standardize(x, eps):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)

    def _semantic_consistency(self, p_up, p_lat):
        n_up = self._spatial_standardize(p_up, self.eps)
        n_lat = self._spatial_standardize(p_lat, self.eps)
        diff = (n_up - n_lat).abs()
        consistency = torch.exp(-self.tau * diff.mean(dim=1, keepdim=True).clamp(min=0.0, max=10.0))
        return diff, consistency

    def _build_route(self, detail_map, consistency):
        if self.fusion_mode == "consistency_only":
            return torch.ones_like(detail_map)
        if self.fusion_mode == "semantic_only":
            return torch.zeros_like(detail_map)
        if self.route_mode == "full":
            route = detail_map * consistency
        elif self.route_mode == "detail_only":
            route = detail_map
        else:
            route = torch.full_like(detail_map, 0.5)
        if self.route_smooth > 1:
            route = F.avg_pool2d(route, self.route_smooth, stride=1, padding=self.route_smooth // 2)
        return route.clamp(0.0, 1.0)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SFRSCAFFuse expects input as [x_up, x_lat].")
        x_up, x_lat = x
        if x_up.ndim != 4 or x_lat.ndim != 4:
            raise ValueError("SFRSCAFFuse inputs must be 4D NCHW tensors.")
        if x_up.shape[-2:] != x_lat.shape[-2:]:
            x_up = F.interpolate(x_up, size=x_lat.shape[-2:], mode="nearest")

        p_up, p_lat = self.up_proj(x_up), self.lat_proj(x_lat)
        diff, consistency = self._semantic_consistency(p_up, p_lat)
        low_lat = F.avg_pool2d(p_lat, kernel_size=3, stride=1, padding=1)
        detail_map = torch.sigmoid(self.detail_head((p_lat - low_lat).abs()))
        route = self._build_route(detail_map, consistency)
        descriptor = torch.cat((p_up, p_lat, diff), dim=1)
        branch_weights = torch.softmax(self.branch_mlp(descriptor), dim=1)
        semantic_gate = torch.sigmoid(self.semantic_head(descriptor))
        alpha_cons = self.alpha_cons_max * torch.tanh(self.alpha_cons_raw)
        alpha_sem = self.alpha_sem_max * torch.sigmoid(self.alpha_sem_raw)

        if self.fusion_mode == "consistency_only":
            cons_mask, sem_mask = consistency, torch.zeros_like(route)
        elif self.fusion_mode == "semantic_only":
            cons_mask, sem_mask = torch.zeros_like(route), semantic_gate
        else:
            cons_mask, sem_mask = route * consistency, (1.0 - route) * semantic_gate

        x_up_out = x_up * (1.0 + alpha_cons * cons_mask * branch_weights[:, 0:1])
        x_lat_out = x_lat * (1.0 + alpha_cons * cons_mask * branch_weights[:, 1:2])
        x_up_out = x_up_out * (1.0 + alpha_sem * sem_mask)
        return torch.cat((x_up_out, x_lat_out), dim=1)


class LGARUp(nn.Module):
    """Lateral-guided adaptive reassembly upsampling with zero-start residual output."""

    def __init__(
        self,
        c_deep,
        c_lat,
        mode="semantic",
        groups=4,
        max_offset=None,
        gamma_max=None,
        reduction=4,
        tau=1.0,
        use_lateral=True,
        use_offset=True,
        use_confidence=True,
    ):
        super().__init__()
        if mode not in {"semantic", "detail"}:
            raise ValueError(f"Unsupported LGARUp mode: {mode}. Expected 'semantic' or 'detail'.")
        self.c_deep, self.c_lat = int(c_deep), int(c_lat)
        self.mode = mode
        self.use_lateral, self.use_offset, self.use_confidence = bool(use_lateral), bool(use_offset), bool(use_confidence)
        self.tau, self.eps = float(tau), 1e-6
        groups = max(1, min(int(groups), self.c_deep))
        self.groups = groups if self.c_deep % groups == 0 else 1
        self.max_offset = float(0.50 if max_offset is None and mode == "semantic" else 0.25 if max_offset is None else max_offset)
        self.gamma_max = float(0.10 if gamma_max is None and mode == "semantic" else 0.05 if gamma_max is None else gamma_max)
        c_mid = max(min(self.c_deep, self.c_lat) // int(reduction), 16)
        self.deep_proj, self.lat_proj = Conv(self.c_deep, c_mid, 1, 1), Conv(self.c_lat, c_mid, 1, 1)
        descriptor_channels = c_mid * 4
        self.offset_head = nn.Sequential(
            nn.Conv2d(descriptor_channels, c_mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 2 * self.groups, 1, 1, 0, bias=True),
        )
        self.detail_head = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, 1, 1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 1, 1, 1, 0, bias=True),
        )
        self.gamma_raw = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.detail_head[-1].weight)
        nn.init.zeros_(self.detail_head[-1].bias)

    @staticmethod
    def _spatial_standardize(x, eps):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)

    @staticmethod
    def _base_grid(batch, groups, out_h, out_w, device, dtype):
        y = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) * (2.0 / float(out_h)) - 1.0
        x = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) * (2.0 / float(out_w)) - 1.0
        yy, xx = torch.meshgrid(y, x)
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).unsqueeze(0)
        return grid.expand(batch, groups, out_h, out_w, 2).reshape(batch * groups, out_h, out_w, 2)

    def _dynamic_sample(self, x_deep, offset, target_size):
        batch, channels, src_h, src_w = x_deep.shape
        out_h, out_w, groups = target_size[0], target_size[1], self.groups
        offset = offset.reshape(batch, groups, 2, out_h, out_w).permute(0, 1, 3, 4, 2).contiguous()
        normalized = torch.empty_like(offset)
        normalized[..., 0] = offset[..., 0] * (2.0 / max(float(src_w), 1.0))
        normalized[..., 1] = offset[..., 1] * (2.0 / max(float(src_h), 1.0))
        grid = self._base_grid(batch, groups, out_h, out_w, x_deep.device, x_deep.dtype) + normalized.reshape(batch * groups, out_h, out_w, 2)
        grouped = x_deep.reshape(batch * groups, channels // groups, src_h, src_w)
        sampled = F.grid_sample(grouped, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.reshape(batch, channels, out_h, out_w)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("LGARUp expects input as [x_deep, x_lat].")
        x_deep, x_lat = x
        if x_deep.ndim != 4 or x_lat.ndim != 4:
            raise ValueError("LGARUp inputs must be 4D NCHW tensors.")
        target_size = x_lat.shape[-2:]
        x_base = F.interpolate(x_deep, size=target_size, mode="nearest")
        deep_embed = self.deep_proj(x_base)
        if self.use_lateral:
            lat_embed = self.lat_proj(x_lat)
            diff = (self._spatial_standardize(deep_embed, self.eps) - self._spatial_standardize(lat_embed, self.eps)).abs()
            detail = (lat_embed - F.avg_pool2d(lat_embed, 3, 1, 1)).abs()
        else:
            lat_embed, diff = deep_embed, torch.zeros_like(deep_embed)
            detail = (deep_embed - F.avg_pool2d(deep_embed, 3, 1, 1)).abs()
        descriptor = torch.cat((deep_embed, lat_embed, diff, detail), dim=1)
        if self.use_offset:
            offset = self.max_offset * torch.tanh(self.offset_head(descriptor))
            x_dynamic = self._dynamic_sample(x_deep, offset, target_size)
        else:
            x_dynamic = F.interpolate(x_deep, size=target_size, mode="bilinear", align_corners=False)
        if self.use_confidence:
            confidence = torch.sigmoid(self.detail_head(detail))
            if self.use_lateral:
                confidence = confidence * torch.exp(-self.tau * diff.mean(dim=1, keepdim=True).clamp(0.0, 10.0))
            if self.mode == "semantic":
                confidence = F.avg_pool2d(confidence, 3, 1, 1)
        else:
            confidence = torch.ones((x_deep.shape[0], 1, *target_size), device=x_deep.device, dtype=x_deep.dtype)
        gamma = self.gamma_max * torch.tanh(self.gamma_raw)
        return x_base + gamma * confidence * (x_dynamic - x_base)


class RFABlock(nn.Module):
    """Reliability-guided feature recalibration before YOLOv13 Detect inputs."""

    def __init__(self, c1, c2=None, mode="p3", ratio=4):
        super().__init__()
        if mode not in {"p3", "p4"}:
            raise ValueError(f"Unsupported RFABlock mode: {mode}. Expected 'p3' or 'p4'.")
        c1 = int(c1)
        c2 = c1 if c2 is None else int(c2)
        self.mode = mode
        if mode == "p3":
            ratio = 4
            init_alpha = 0.05
        else:
            ratio = 8
            init_alpha = 0.02
        hidden = max(c2 // int(ratio), 32)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.gate = nn.Sequential(
            Conv(c2 * 3, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            nn.Conv2d(hidden, c2, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.enhance = nn.Sequential(
            Conv(c2, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, c2, 1, 1),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x):
        x = self.proj(x)
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        res = x - low
        gate_in = torch.cat((x, low, res.abs()), dim=1)
        g = self.gate(gate_in) * self.channel_gate(res.abs())
        e = self.enhance(res)
        return x + self.alpha.tanh() * g * e


class P2LiteGuide(nn.Module):
    """Compress P2 details into P3-resolution guidance without adding a P2 Detect head."""

    def __init__(self, c_p2, c_p3, c_out, r=4):
        super().__init__()
        c_out = int(c_out)
        hidden = max(c_out // int(r), 32)
        self.p2_reduce = nn.Sequential(
            nn.Conv2d(c_p2, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.low_down = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 2, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        self.detail_down = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 2, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        self.p3_proj = (
            nn.Identity()
            if c_p3 == c_out
            else nn.Sequential(
                nn.Conv2d(c_p3, c_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
            )
        )
        gate_hidden = max(c_out // int(r), 32)
        self.gate = nn.Sequential(
            nn.Conv2d(c_out * 4, gate_hidden, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, c_out, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"P2LiteGuide expects [p2, p3], got {type(x).__name__}.")
        p2, p3 = x
        p3 = self.p3_proj(p3)
        p2 = self.p2_reduce(p2)
        p2_blur = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)
        p2_low = self.low_down(p2_blur)
        p2_detail = self.detail_down(p2 - p2_blur)
        if p2_low.shape[-2:] != p3.shape[-2:]:
            p2_low = F.interpolate(p2_low, size=p3.shape[-2:], mode="nearest")
        if p2_detail.shape[-2:] != p3.shape[-2:]:
            p2_detail = F.interpolate(p2_detail, size=p3.shape[-2:], mode="nearest")
        diff = (p3 - p2_low).abs()
        gate = self.gate(torch.cat((p3, p2_low, diff, p2_detail), dim=1))
        y = p3 + self.alpha.tanh() * gate * p2_detail + self.beta.tanh() * gate * (p2_low - p3)
        return self.out(y)


class FastNormFusion(nn.Module):
    """Fuse feature maps with non-negative, normalized learnable weights."""

    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        if int(n_inputs) < 2:
            raise ValueError(f"FastNormFusion requires at least two inputs, got {n_inputs}.")
        if float(eps) <= 0:
            raise ValueError(f"FastNormFusion eps must be positive, got {eps}.")
        self.n_inputs = int(n_inputs)
        self.eps = float(eps)
        self.weights = nn.Parameter(torch.ones(self.n_inputs, dtype=torch.float32))

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != self.n_inputs:
            length = len(xs) if isinstance(xs, (list, tuple)) else "non-sequence"
            raise ValueError(f"FastNormFusion expected {self.n_inputs} tensors, got {length}.")
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        out = xs[0] * weights[0].to(dtype=xs[0].dtype)
        for i in range(1, self.n_inputs):
            out = out + xs[i] * weights[i].to(dtype=xs[i].dtype)
        return out


class LiteBiFPNNode(nn.Module):
    """Channel-align and fuse a list of multi-scale features with fast normalized weights."""

    def __init__(self, c1, c2, n_inputs=2, eps=1e-4):
        super().__init__()
        c1 = [int(c1)] * int(n_inputs) if isinstance(c1, int) else [int(c) for c in c1]
        if len(c1) != int(n_inputs) or int(n_inputs) < 2:
            raise ValueError(f"LiteBiFPNNode expected {n_inputs} input-channel entries, got {c1}.")
        if int(c2) <= 0:
            raise ValueError(f"LiteBiFPNNode output channels must be positive, got {c2}.")
        self.n_inputs = int(n_inputs)
        self.out_channels = int(c2)
        self.projections = nn.ModuleList(
            nn.Identity() if c == self.out_channels else Conv(c, self.out_channels, 1, 1) for c in c1
        )
        self.fuse = FastNormFusion(self.n_inputs, eps=eps)
        self.refine = nn.Sequential(
            DSConv(self.out_channels, self.out_channels, 3, 1),
            Conv(self.out_channels, self.out_channels, 1, 1),
        )

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != self.n_inputs:
            length = len(xs) if isinstance(xs, (list, tuple)) else "non-sequence"
            raise ValueError(f"LiteBiFPNNode expected {self.n_inputs} tensors, got {length}.")
        target_size = xs[0].shape[-2:]
        aligned = []
        for projection, x in zip(self.projections, xs):
            x = projection(x)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="nearest")
            aligned.append(x)
        return self.refine(self.fuse(aligned))


class UDCStem(Conv):
    """Underwater degradation calibration folded into the first backbone convolution."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, hidden=8):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.channel_scale = nn.Parameter(torch.ones(1, c1, 1, 1))
        self.channel_bias = nn.Parameter(torch.zeros(1, c1, 1, 1))
        hidden = max(int(hidden), c1)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 * 2, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.Conv2d(c1, c1, 1, bias=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        contrast = (x - mean).abs().mean(dim=(2, 3), keepdim=True)
        gate = 2.0 * self.context(torch.cat((mean, contrast), dim=1))
        calibrated = x * self.channel_scale * gate + self.channel_bias
        calibrated = calibrated + 0.05 * self.alpha.tanh() * self.local(calibrated)
        return super().forward(calibrated)

    def forward_fuse(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        contrast = (x - mean).abs().mean(dim=(2, 3), keepdim=True)
        gate = 2.0 * self.context(torch.cat((mean, contrast), dim=1))
        calibrated = x * self.channel_scale * gate + self.channel_bias
        calibrated = calibrated + 0.05 * self.alpha.tanh() * self.local(calibrated)
        return super().forward_fuse(calibrated)

class _UCRARefineBranch(nn.Module):
    """Lightweight residual branch for UCRA v10 refinement."""

    def __init__(self, hidden, k=3, dilation=1):
        super().__init__()
        self.conv = Conv(hidden, hidden, k, 1, p=autopad(k, None, dilation), d=dilation)
        self.out = nn.Conv2d(hidden, hidden, 1, bias=True)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.conv(x))


class _UCRABaseRefine(nn.Module):
    """UCRA v10 skip-guided upsampling refinement with scalar branch weighting."""

    def __init__(self, c_deep, c_skip, c_out, hidden=128, num_branches=2, k=3, aux_stride=8):
        super().__init__()
        hidden = max(int(hidden), 8)
        self.c_out = c_out
        self.aux_stride = aux_stride
        self.deep_proj = Conv(c_deep, hidden, 1, 1)
        self.skip_proj = Conv(c_skip, hidden, 1, 1)
        dilations = tuple(range(1, int(num_branches) + 1))
        self.branches = nn.ModuleList(_UCRARefineBranch(hidden, k=k, dilation=d) for d in dilations)
        self.branch_w = nn.Parameter(torch.ones(len(self.branches)) / len(self.branches))
        self.skip_attn = nn.Sequential(
            Conv(c_skip, hidden, 1, 1),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.skip_attn[1].weight)
        nn.init.constant_(self.skip_attn[1].bias, 4.0)
        self.out_proj = Conv(hidden, c_out, 1, 1)
        self.aux_head = nn.Conv2d(c_out, 1, 1, bias=True)
        nn.init.normal_(self.aux_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.aux_head.bias)

    def _branch_weights(self):
        return self.branch_w.softmax(0)

    def _cache_aux(self, out):
        if not (self.training and torch.is_grad_enabled()):
            return
        cache = getattr(_UCRABaseRefine, "_ucra_aux_forward_cache", None)
        if cache is None:
            cache = []
            setattr(_UCRABaseRefine, "_ucra_aux_forward_cache", cache)
        cache.append({"pred": self.aux_head(out), "stride": self.aux_stride})

    def forward(self, x):
        deep, skip = x
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="nearest")
        deep_embed = self.deep_proj(deep)
        refine = deep_embed + self.skip_proj(skip)
        weights = self._branch_weights()
        enhanced = sum(w * branch(refine) for w, branch in zip(weights, self.branches))
        enhanced = enhanced * self.skip_attn(skip)
        out = self.out_proj(deep_embed + enhanced)
        self._cache_aux(out)
        return out


class UCRA_SemUp(_UCRABaseRefine):
    """Semantic UCRA upsampling module with three skip-guided refinement branches."""

    def __init__(self, c_deep, c_skip, c_out, hidden=128, k=3):
        super().__init__(c_deep, c_skip, c_out, hidden=hidden, num_branches=3, k=k, aux_stride=16)


class UCRA_DetailUp(_UCRABaseRefine):
    """Detail UCRA upsampling module with two skip-guided refinement branches."""

    def __init__(self, c_deep, c_skip, c_out, hidden=128, k=3):
        super().__init__(c_deep, c_skip, c_out, hidden=hidden, num_branches=2, k=k, aux_stride=8)


class _SIRUCRARepBranch(nn.Module):
    """Small zero-start re-parameterized branch for the P3 UCRA detail path."""

    def __init__(self, hidden):
        super().__init__()
        self.rep = RepConv(hidden, hidden, 3, 1, 1, g=hidden, act=False, bn=True)
        self.mix = nn.Conv2d(hidden, hidden, 1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x):
        return x + 0.05 * self.alpha.tanh() * self.mix(self.rep(x))


class SIRUCRA_DetailUp(UCRA_DetailUp):
    """Stability-preserved re-parameterized UCRA detail module for P3 recovery."""

    def __init__(self, c_deep, c_skip, c_out, hidden=128, k=3):
        super().__init__(c_deep, c_skip, c_out, hidden=hidden, k=k)
        self.sir = _SIRUCRARepBranch(hidden)

    def forward(self, x):
        deep, skip = x
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="nearest")
        deep_embed = self.deep_proj(deep)
        refine = self.sir(deep_embed + self.skip_proj(skip))
        weights = self.branch_w.softmax(0)
        enhanced = sum(w * branch(refine) for w, branch in zip(weights, self.branches))
        enhanced = enhanced * self.skip_attn(skip)
        out = self.out_proj(deep_embed + enhanced)
        self._cache_aux(out)
        return out
