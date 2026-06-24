"""
Rotary Position Embeddings for Gemma 4 E4B.

Two variants:
  1. Standard RoPE (sliding-window layers): theta=10000, full rotation
  2. Proportional RoPE (global layers): theta=1M, partial_rotary_factor=0.25
     Only the first 25% of head dimensions get rotary encoding.
"""

import torch
import torch.nn as nn
import gemma4_kernels


class RotaryEmbedding(nn.Module):
    """Precomputes and caches cos/sin tables for RoPE."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 131072,
        theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.partial_rotary_factor = partial_rotary_factor

        # How many dimensions actually get rotated
        self.rotary_dim = int(dim * partial_rotary_factor)
        # Must be even
        assert self.rotary_dim % 2 == 0, f"rotary_dim must be even, got {self.rotary_dim}"

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache up to seq_len."""
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [seq_len, rotary_dim // 2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, rotary_dim]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape [seq_len, rotary_dim]."""
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)

        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    partial_rotary_factor: float = 1.0,
) -> torch.Tensor:
    """
    Apply rotary embeddings to x.

    Args:
        x: [batch, num_heads, seq_len, head_dim]
        cos: [max_seq, rotary_dim]
        sin: [max_seq, rotary_dim]
        position_ids: [batch, seq_len]
        partial_rotary_factor: fraction of head_dim to rotate (1.0 = all)

    Returns:
        x with rotary embeddings applied, same shape as input.
    """
    return gemma4_kernels.apply_rope(
        x.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        position_ids.contiguous(),
        partial_rotary_factor,
    )
    head_dim = x.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)

    if rotary_dim == 0:
        return x

    # Gather cos/sin for the actual positions
    # cos/sin: [max_seq, rotary_dim] -> gather -> [batch, seq_len, rotary_dim]
    cos_pos = cos[position_ids]  # [batch, seq_len, rotary_dim]
    sin_pos = sin[position_ids]  # [batch, seq_len, rotary_dim]

    # Reshape for broadcasting: [batch, 1, seq_len, rotary_dim]
    cos_pos = cos_pos.unsqueeze(1)
    sin_pos = sin_pos.unsqueeze(1)

    if rotary_dim < head_dim:
        # Partial rotation: only rotate first rotary_dim dimensions
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = x_rot * cos_pos + _rotate_half(x_rot) * sin_pos
        return torch.cat([x_rot, x_pass], dim=-1)
    else:
        # Full rotation
        return x * cos_pos + _rotate_half(x) * sin_pos
