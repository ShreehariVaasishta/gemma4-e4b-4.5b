"""
Sampling utilities for autoregressive text generation.
"""

import torch
import torch.nn.functional as F


def sample_top_k_top_p(
    logits: torch.Tensor,  # [batch, vocab]
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample from logits with temperature, top-k, and top-p (nucleus) filtering.

    Returns:
        next_token: [batch] — sampled token ids
    """
    if temperature <= 0:
        # Greedy
        return logits.argmax(dim=-1)

    logits = logits / temperature

    # ── Top-k filtering ─────────────────────────────────────────────
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = top_k_values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # ── Top-p (nucleus) filtering ───────────────────────────────────
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # Shift right so the first token above the threshold is kept
        sorted_mask = cumulative_probs - probs > top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # Scatter back to original ordering
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    # ── Sample ──────────────────────────────────────────────────────
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
