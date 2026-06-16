"""
Gemma 4 E4B model configuration.

All values extracted from:
  https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Gemma4Config:
    """Text-only configuration for Gemma 4 E4B inference."""

    # ── Core dimensions ─────────────────────────────────────────────────
    hidden_size: int = 2560
    num_hidden_layers: int = 42
    num_attention_heads: int = 8          # Q heads
    num_key_value_heads: int = 2          # KV heads (GQA ratio = 4)
    head_dim: int = 256                   # sliding-window layers
    global_head_dim: int = 512            # global-attention layers
    intermediate_size: int = 10240        # MLP intermediate dim
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6

    # ── Per-Layer Embeddings (PLE) ──────────────────────────────────────
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262144  # same as main vocab

    # ── Attention ───────────────────────────────────────────────────────
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Layer types: pattern of 5 sliding + 1 global, repeated 7 times = 42 layers
    layer_types: list[str] = field(default_factory=lambda: [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
    ] * 7)

    sliding_window: int = 512
    max_position_embeddings: int = 131072

    # ── KV Sharing ──────────────────────────────────────────────────────
    num_kv_shared_layers: int = 18  # last 18 layers reuse KV from earlier ones

    # ── RoPE ────────────────────────────────────────────────────────────
    # Sliding-window layers: standard RoPE, theta=10000
    rope_theta_sliding: float = 10000.0

    # Global layers: proportional RoPE, theta=1M, partial_rotary_factor=0.25
    rope_theta_global: float = 1000000.0
    partial_rotary_factor: float = 0.25

    # ── Activation ──────────────────────────────────────────────────────
    hidden_activation: str = "gelu_pytorch_tanh"

    # ── Logit softcapping ───────────────────────────────────────────────
    final_logit_softcapping: float = 30.0

    # ── Embeddings ──────────────────────────────────────────────────────
    tie_word_embeddings: bool = True

    # ── Tokenizer ───────────────────────────────────────────────────────
    bos_token_id: int = 2
    eos_token_id: list[int] = field(default_factory=lambda: [1, 106])
    pad_token_id: int = 0

    # ── Misc ────────────────────────────────────────────────────────────
    use_cache: bool = True

    # ── Derived helpers ─────────────────────────────────────────────────
    @property
    def num_gqa_groups(self) -> int:
        """Number of Q heads per KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def num_ple_layers(self) -> int:
        """Number of layers that use Per-Layer Embeddings.
        Typically (num_hidden_layers - num_global_layers) but in practice
        every non-first layer uses PLE.  We derive from config: layers with
        hidden_size_per_layer_input > 0 use PLE."""
        return self.num_hidden_layers if self.hidden_size_per_layer_input > 0 else 0

    @property
    def kv_sharing_start_layer(self) -> int:
        """First layer index that shares KV with an earlier layer."""
        return self.num_hidden_layers - self.num_kv_shared_layers

    def head_dim_for_layer(self, layer_idx: int) -> int:
        """Return the appropriate head_dim for a given layer."""
        if self.layer_types[layer_idx] == "full_attention":
            return self.global_head_dim
        return self.head_dim

    def rope_theta_for_layer(self, layer_idx: int) -> float:
        """Return the appropriate RoPE theta for a given layer."""
        if self.layer_types[layer_idx] == "full_attention":
            return self.rope_theta_global
        return self.rope_theta_sliding

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    def is_global(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    @property
    def kv_sharing_start_layer(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    @classmethod
    def from_json(cls, path: str | Path) -> "Gemma4Config":
        """Load from a HuggingFace config.json, pulling values from text_config."""
        with open(path) as f:
            raw = json.load(f)

        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters", {})

        return cls(
            hidden_size=text.get("hidden_size", cls.hidden_size),
            num_hidden_layers=text.get("num_hidden_layers", cls.num_hidden_layers),
            num_attention_heads=text.get("num_attention_heads", cls.num_attention_heads),
            num_key_value_heads=text.get("num_key_value_heads", cls.num_key_value_heads),
            head_dim=text.get("head_dim", cls.head_dim),
            global_head_dim=text.get("global_head_dim", cls.global_head_dim),
            intermediate_size=text.get("intermediate_size", cls.intermediate_size),
            vocab_size=text.get("vocab_size", cls.vocab_size),
            rms_norm_eps=text.get("rms_norm_eps", cls.rms_norm_eps),
            hidden_size_per_layer_input=text.get("hidden_size_per_layer_input", cls.hidden_size_per_layer_input),
            vocab_size_per_layer_input=text.get("vocab_size_per_layer_input", cls.vocab_size_per_layer_input),
            attention_bias=text.get("attention_bias", cls.attention_bias),
            layer_types=text.get("layer_types", cls.__dataclass_fields__["layer_types"].default_factory()),
            sliding_window=text.get("sliding_window", cls.sliding_window),
            max_position_embeddings=text.get("max_position_embeddings", cls.max_position_embeddings),
            num_kv_shared_layers=text.get("num_kv_shared_layers", cls.num_kv_shared_layers),
            rope_theta_sliding=rope.get("sliding_attention", {}).get("rope_theta", cls.rope_theta_sliding),
            rope_theta_global=rope.get("full_attention", {}).get("rope_theta", cls.rope_theta_global),
            partial_rotary_factor=rope.get("full_attention", {}).get("partial_rotary_factor", cls.partial_rotary_factor),
            hidden_activation=text.get("hidden_activation", cls.hidden_activation),
            final_logit_softcapping=text.get("final_logit_softcapping", cls.final_logit_softcapping),
            tie_word_embeddings=text.get("tie_word_embeddings", raw.get("tie_word_embeddings", cls.tie_word_embeddings)),
            bos_token_id=text.get("bos_token_id", cls.bos_token_id),
            eos_token_id=raw.get("eos_token_id", cls.__dataclass_fields__["eos_token_id"].default_factory()),
            pad_token_id=text.get("pad_token_id", cls.pad_token_id),
        )
