"""
RMSNorm — matches Gemma 4's normalization.

Gemma uses RMSNorm with learnable scale (weight), adding +1 offset internally
(the stored weights are offsets from 1.0).
"""

import torch
import torch.nn as nn
import gemma4_kernels


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid underflow/overflow with rsqrt
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass self.weight if with_scale is True, otherwise pass None
        weight = self.weight if self.with_scale else None
        normed = gemma4_kernels.rms_norm(x, self.eps, weight)
        return normed.type_as(x)
