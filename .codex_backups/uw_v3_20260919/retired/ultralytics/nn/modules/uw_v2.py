from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .block import C3k2
from .conv import Conv


class FEMLite(nn.Module):
    """
    Underwater feature enhancement refinement block.

    Structure:
        1x1 reduce
          -> identity branch
          -> 3x3, dilation=1
          -> 3x3, dilation=2
          -> 3x3, dilation=5
        concat
        -> 1x1 fuse
        -> small-gain residual addition

    Notes:
        - Input/output spatial size is unchanged.
        - Default c1 == c2.
        - Only standard Conv/BN/SiLU/Concat/Add/Mul are used.
        - gamma is deliberately initialized small to preserve pretrained features.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        expansion: float = 0.25,
        gamma_init: float = 0.05,
    ) -> None:
        super().__init__()

        c2 = c1 if c2 is None else int(c2)

        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"c1/c2 must be positive, got {c1}/{c2}")
        if not (0.0 < expansion <= 1.0):
            raise ValueError(f"expansion must be in (0, 1], got {expansion}")
        if gamma_init < 0.0:
            raise ValueError(f"gamma_init must be >= 0, got {gamma_init}")

        # Round hidden channels to a multiple of 8.
        hidden = max(8, int(round(c2 * expansion / 8.0)) * 8)

        self.reduce = Conv(c1, hidden, k=1, s=1)

        self.branch_local = Conv(
            hidden,
            hidden,
            k=3,
            s=1,
            d=1,
        )

        self.branch_mid = Conv(
            hidden,
            hidden,
            k=3,
            s=1,
            d=2,
        )

        self.branch_wide = Conv(
            hidden,
            hidden,
            k=3,
            s=1,
            d=5,
        )

        self.fuse = Conv(
            hidden * 4,
            c2,
            k=1,
            s=1,
            act=False,
        )

        self.short = (
            nn.Identity()
            if c1 == c2
            else Conv(
                c1,
                c2,
                k=1,
                s=1,
                act=False,
            )
        )

        self.gamma = nn.Parameter(
            torch.full(
                (1, c2, 1, 1),
                float(gamma_init),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        z = self.reduce(x)

        y = torch.cat(
            (
                z,
                self.branch_local(z),
                self.branch_mid(z),
                self.branch_wide(z),
            ),
            dim=1,
        )

        y = self.fuse(y)

        gamma = self.gamma.to(dtype=y.dtype)

        return self.short(x) + gamma * y


class C3k2_UWFEM(C3k2):
    """
    C3k2 + residual underwater feature enhancement.

    Important:
        This subclasses C3k2 instead of wrapping it inside another `base`
        attribute. Therefore original C3k2 parameter names such as cv1/cv2/m
        remain unchanged, which maximizes transfer from yolo11n.pt when layer
        indices and channel shapes are unchanged.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        fem_expansion: float = 0.25,
        gamma_init: float = 0.05,
    ) -> None:

        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            g=g,
            shortcut=shortcut,
        )

        self.uw_fem = FEMLite(
            c1=c2,
            c2=c2,
            expansion=fem_expansion,
            gamma_init=gamma_init,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = super().forward(x)

        return self.uw_fem(x)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Keep refinement active when callers select the split implementation."""
        return self.uw_fem(super().forward_split(x))


class StableBiConcat2(nn.Module):
    """
    Two-input learnable weighted concatenation.

    Initial behavior is EXACTLY ordinary Concat:

        logits = [0, 0]
        softmax(logits) = [0.5, 0.5]
        weights = 2 * softmax = [1, 1]

        out = cat(x0 * 1, x1 * 1)

    Therefore pretrained YOLO11 Neck features are not rescaled at
    initialization. Fine-tuning can then learn the relative branch importance.

    Args:
        dimension:
            concat dimension, normally 1.
    """

    def __init__(
        self,
        dimension: int = 1,
    ) -> None:

        super().__init__()

        self.d = int(dimension)

        self.logits = nn.Parameter(
            torch.zeros(
                2,
                dtype=torch.float32,
            )
        )

    def normalized_weights(
        self,
        ref: torch.Tensor,
    ) -> torch.Tensor:

        weights = 2.0 * torch.softmax(
            self.logits,
            dim=0,
        )

        return weights.to(
            device=ref.device,
            dtype=ref.dtype,
        )

    def forward(
        self,
        x: Sequence[torch.Tensor],
    ) -> torch.Tensor:

        if (
            not isinstance(x, (list, tuple))
            or len(x) != 2
        ):
            raise ValueError(
                "StableBiConcat2 expects exactly two tensors"
            )

        a, b = x

        # Keep useful eager-mode diagnostics, but avoid data-dependent
        # Python shape checks while tracing/exporting.
        if not torch.jit.is_tracing():

            if a.ndim != b.ndim:
                raise ValueError(
                    f"rank mismatch: {a.ndim} vs {b.ndim}"
                )

            dim = (
                self.d
                if self.d >= 0
                else a.ndim + self.d
            )

            for i, (sa, sb) in enumerate(
                zip(a.shape, b.shape)
            ):
                if i != dim and sa != sb:
                    raise ValueError(
                        "shape mismatch outside "
                        f"concat dim {self.d}: "
                        f"{tuple(a.shape)} vs "
                        f"{tuple(b.shape)}"
                    )

        weights = self.normalized_weights(a)

        return torch.cat(
            (
                a * weights[0],
                b * weights[1],
            ),
            dim=self.d,
        )
