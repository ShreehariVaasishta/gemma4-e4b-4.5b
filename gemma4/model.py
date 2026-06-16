"""
Gemma 4 E4B — Full model: embeddings → decoder stack → LM head.

Key architectural features:
  - Per-Layer Embeddings (PLE): each layer gets its own token embedding lookup,
    gated into the residual stream
  - Hybrid attention: 5 sliding-window + 1 global, repeated 7 times
  - Logit softcapping at 30.0
  - Tied word embeddings (embedding = LM head)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma4.config import Gemma4Config
from gemma4.layers.norm import RMSNorm
from gemma4.layers.mlp import GemmaMLP
from gemma4.layers.attention import GemmaAttention
from gemma4.layers.rope import RotaryEmbedding


# ────────────────────────────────────────────────────────────────────────────────
# Per-Layer Embedding (PLE) Module
# ────────────────────────────────────────────────────────────────────────────────


class PerLayerEmbedding(nn.Module):
    """
    Centralized PLE components.
    In Gemma 4 E4B, the token embeddings and main projection are centralized,
    but the gating and secondary projection happen inside each layer.
    """

    def __init__(self, config: Gemma4Config, num_layers: int):
        super().__init__()
        ple_dim = config.hidden_size_per_layer_input

        # Per-layer token embeddings: [vocab, num_layers * ple_dim]
        # Stored as one big embedding, sliced per layer
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            num_layers * ple_dim,
        )

        # Project main embeddings -> per-layer signals
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            num_layers * ple_dim,
            bias=False,
        )

        self.per_layer_projection_norm = RMSNorm(ple_dim, eps=config.rms_norm_eps)
        self.num_layers = num_layers
        self.ple_dim = ple_dim

    def get_per_layer_input(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        inputs_embeds: torch.Tensor,  # [batch, seq_len, hidden_size]
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute the pre-layer input signal for a given layer.
        This represents the token-identity lookup + context-aware projection
        BEFORE it gets gated by the layer's hidden states.

        Returns: [batch, seq_len, ple_dim]
        """
        # Token-identity lookup: [batch, seq_len, num_layers * ple_dim]
        per_layer_tok = self.embed_tokens_per_layer(input_ids)
        # Slice for this layer: [batch, seq_len, ple_dim]
        start = layer_idx * self.ple_dim
        end = start + self.ple_dim
        per_layer_tok = per_layer_tok[..., start:end]

        # Context-aware projection from main embeddings
        per_layer_proj = self.per_layer_model_projection(inputs_embeds)
        per_layer_proj = per_layer_proj[..., start:end]

        # Scale by 1/sqrt(2) and combine
        combined = (per_layer_tok + per_layer_proj) * (1.0 / math.sqrt(2.0))

        return self.per_layer_projection_norm(combined)


# ────────────────────────────────────────────────────────────────────────────────
# Decoder Layer
# ────────────────────────────────────────────────────────────────────────────────


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Layer scalar - learnable scaling factor [1]
        self.layer_scalar = nn.Parameter(torch.ones(1))

        # Per-layer PLE components
        ple_dim = config.hidden_size_per_layer_input
        if ple_dim > 0:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, ple_dim, bias=False)
            self.per_layer_projection = nn.Linear(ple_dim, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        per_layer_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns: hidden_states
        """
        residual = hidden_states

        # ── Self Attention ──────────────────────────────────────────
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            shared_kv=shared_kv,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ── MLP ─────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ── PLE Injection ───────────────────────────────────────────
        if per_layer_input is not None and self.config.hidden_size_per_layer_input > 0:
            residual = hidden_states
            gate = F.gelu(self.per_layer_input_gate(hidden_states), approximate="tanh")
            ple_signal = gate * per_layer_input
            ple_signal = self.per_layer_projection(ple_signal)
            ple_signal = self.post_per_layer_input_norm(ple_signal)
            hidden_states = residual + ple_signal

        # ── Final Layer Scaling ─────────────────────────────────────
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states


# ────────────────────────────────────────────────────────────────────────────────
# Full Model
# ────────────────────────────────────────────────────────────────────────────────


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config

        # Main token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Centralized Per-Layer Embeddings
        if config.hidden_size_per_layer_input > 0:
            self.ple = PerLayerEmbedding(config, config.num_hidden_layers)
        else:
            self.ple = None

        # Decoder layers
        self.layers = nn.ModuleList([Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE embeddings — one per attention type
        # Sliding-window layers: full rotation, theta=10k
        self.rope_sliding = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta_sliding,
            partial_rotary_factor=1.0,
        )
        # Global layers: partial rotation, theta=1M
        self.rope_global = RotaryEmbedding(
            dim=config.global_head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta_global,
            partial_rotary_factor=config.partial_rotary_factor,
        )

        # Normalizer for embeddings (Gemma multiplies by sqrt(hidden_size))
        self.embed_scale = config.hidden_size**0.5

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        position_ids: Optional[torch.Tensor] = None,  # [batch, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len] or 4D
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # ── Token embeddings ────────────────────────────────────────
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds * self.embed_scale

        # ── Precompute PLE per-layer inputs ─────────────────────────
        ple_inputs = {}
        if self.ple is not None:
            for i in range(self.config.num_hidden_layers):
                ple_inputs[i] = self.ple.get_per_layer_input(input_ids, inputs_embeds, i)

        # ── Build attention mask ────────────────────────────────────
        # Determine the full KV length (past cache + current sequence)
        kv_len = seq_len
        if kv_cache and "layer_0" in kv_cache:
            kv_len += kv_cache["layer_0"][0].shape[2]

        causal_mask_global = self._make_causal_mask(seq_len, kv_len, device, hidden_states.dtype)
        causal_mask_sliding = self._make_sliding_causal_mask(
            seq_len, kv_len, self.config.sliding_window, device, hidden_states.dtype
        )

        # ── Precompute RoPE ─────────────────────────────────────────
        # We must generate cos/sin tables up to the maximum position_id, which is kv_len
        cos_sliding, sin_sliding = self.rope_sliding(kv_len, device)
        cos_global, sin_global = self.rope_global(kv_len, device)

        # ── Decoder layers ──────────────────────────────────────────
        shared_kv = {}
        for i, layer in enumerate(self.layers):
            is_sliding = self.config.is_sliding(i)

            # Select appropriate RoPE and mask
            if is_sliding:
                cos, sin = cos_sliding, sin_sliding
                mask = causal_mask_sliding
            else:
                cos, sin = cos_global, sin_global
                mask = causal_mask_global

            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                shared_kv=shared_kv,
                attention_mask=mask,
                kv_cache=kv_cache,
                per_layer_input=ple_inputs.get(i),
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    def _make_causal_mask(seq_len: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Standard causal (lower-triangular) attention mask."""
        mask = torch.full((seq_len, kv_len), float("-inf"), device=device, dtype=dtype)
        # For decode (seq_len=1), offset is kv_len - 1, meaning we attend to all kv_len.
        # For prefill (seq_len=kv_len), offset is 0.
        offset = kv_len - seq_len
        mask = torch.triu(mask, diagonal=offset + 1)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]

    @staticmethod
    def _make_sliding_causal_mask(
        seq_len: int, kv_len: int, window: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Causal mask with sliding window — tokens can only attend to the
        previous `window` positions."""
        mask = torch.full((seq_len, kv_len), float("-inf"), device=device, dtype=dtype)
        offset = kv_len - seq_len

        for i in range(seq_len):
            # Absolute position of query i is (offset + i)
            abs_pos = offset + i
            start = max(0, abs_pos - window + 1)
            mask[i, start : abs_pos + 1] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)

        # LM head — tied with embed_tokens if tie_word_embeddings
        print(f"{config.tie_word_embeddings=}")
        if config.tie_word_embeddings:
            self.lm_head = None  # will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.final_logit_softcapping = config.final_logit_softcapping

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        # Logit softcapping: tanh(logits / cap) * cap
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation loop."""
        from gemma4.sampling import sample_top_k_top_p

        if eos_token_ids is None:
            eos_token_ids = self.config.eos_token_id

        kv_cache: dict = {}
        generated = input_ids.clone()

        for step in range(max_new_tokens):
            if step == 0:
                # Prefill: process all tokens
                ids = generated
                pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            else:
                # Decode: only the last token
                ids = generated[:, -1:]
                pos = torch.tensor([[generated.shape[1] - 1]], device=ids.device)

            logits = self(ids, position_ids=pos, kv_cache=kv_cache)
            next_logits = logits[:, -1, :]  # [batch, vocab]

            next_token = sample_top_k_top_p(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            # Check EOS
            if next_token.item() in eos_token_ids:
                break

        return generated
