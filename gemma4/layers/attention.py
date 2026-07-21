"""
Gemma 4 E4B Attention — supports both sliding-window and global attention.

Key details:
  - GQA: 8 Q heads, 2 KV heads (ratio 4:1)
  - Sliding layers:  head_dim=256, standard RoPE (theta=10k), window=512
  - Global layers:   head_dim=512, proportional RoPE (theta=1M, 25% partial rotation)
  - QK norms: RMSNorm applied to Q and K after projection
  - All layers have their own Q/K/V projections
  - No attention bias
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma4.config import Gemma4Config
from gemma4.layers.norm import RMSNorm
from gemma4.layers.rope import apply_rotary_emb
import gemma4_kernels


class GemmaAttention(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = config.is_sliding(layer_idx)

        # KV Sharing parameters
        self.is_kv_shared_layer = layer_idx >= config.kv_sharing_start_layer

        # Find if we are the last non-shared layer of this type
        prev_layers = config.layer_types[: config.kv_sharing_start_layer]
        # Reverse find the last occurrence of self.layer_type in prev_layers
        if self.layer_type in prev_layers:
            last_idx = len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
            self.store_full_length_kv = (not self.is_kv_shared_layer) and (layer_idx == last_idx)
        else:
            self.store_full_length_kv = False

        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_gqa_groups = config.num_gqa_groups

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # Q/K/V normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # ── Q/K/V projections ───────────────────────────────────────
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        # Apply RoPE to Q
        partial = self.config.partial_rotary_factor if not self.is_sliding else 1.0
        q = apply_rotary_emb(q, cos, sin, position_ids, partial_rotary_factor=partial)

        if self.is_kv_shared_layer:
            # We reuse the KV from the last non-shared layer of the same type
            k, v = shared_kv[self.layer_type]
            # Slicing is not needed if the shared_kv holds the full length up to current sequence
        else:
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

            k = self.k_norm(k)
            k = k.transpose(1, 2)
            k = apply_rotary_emb(k, cos, sin, position_ids, partial_rotary_factor=partial)

            v = self.v_norm(v)
            v = v.transpose(1, 2)

        # ── KV cache update ─────────────────────────────────────────
        if kv_cache is not None and not self.is_kv_shared_layer:
            cache_key = f"layer_{self.layer_idx}"
            if cache_key in kv_cache:
                cached_k, cached_v = kv_cache[cache_key]
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            kv_cache[cache_key] = (k, v)

        if self.store_full_length_kv:
            shared_kv[self.layer_type] = (k, v)

        # ── GQA: expand KV heads ────────────────────────────────────
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_gqa_groups, dim=1)
            v = v.repeat_interleave(self.num_gqa_groups, dim=1)

        # ── Attention (using our custom FlashAttention Kernel) ──────────

        # We need everything as FP32 because our educational kernel is FP32 only.
        q_fp32 = q.to(torch.float32).contiguous()
        k_fp32 = k.to(torch.float32).contiguous()
        v_fp32 = v.to(torch.float32).contiguous()

        # Determine the sliding window size
        # If it's a sliding layer, use the config's sliding window size. Otherwise use -1 to indicate global attention.
        window_size = self.config.sliding_window if self.is_sliding else -1

        # Call the custom CUDA kernel
        attn_output = gemma4_kernels.flash_attn(q_fp32, k_fp32, v_fp32, window_size)

        # Convert back to the original datatype
        attn_output = attn_output.to(v.dtype)
        # [batch, num_heads, seq_len, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_output)
