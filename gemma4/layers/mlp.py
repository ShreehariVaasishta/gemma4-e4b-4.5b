"""
Gemma 4 MLP — GeLU-gated feed-forward.

Architecture:
    hidden_states -> gate_proj -> gelu_tanh  ──┐
                                                ├─ * ──> down_proj -> out
    hidden_states -> up_proj ──────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gelu_pytorch_tanh = GELU with tanh approximation
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        return self.down_proj(gate * up)
