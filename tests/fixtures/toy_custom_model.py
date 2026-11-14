"""Toy custom PyTorch architecture with non-standard module names for E2 tests.

Used to verify that `build_model_map` identifies embedding/unembedding layers,
flags unfamiliar block submodules (`mix`, `mystery_gate`) in `needs_confirmation`,
leaves `confirmed_by_user=False`, and updates cleanly via `apply_confirmation`.
"""

from __future__ import annotations

import torch
from torch import nn


class OddBlock(nn.Module):
    """Transformer-like block using unfamiliar submodule names."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.ln_pre = nn.LayerNorm(d_model)
        # `mix` matches attention pattern; `mystery_gate` is unrecognized -> needs_confirmation
        self.mix = nn.Linear(d_model, d_model, bias=False)
        self.mystery_gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_pre(x)
        x = x + self.mix(h)
        x = x + self.mystery_gate(x)
        return x


class ToyCustomLanguageModel(nn.Module):
    """Minimal causal language model with custom attribute names (`tok_emb`, `blk`, `out_proj`)."""

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 32,
        n_layer: int = 2,
        n_head: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blk = nn.ModuleList([OddBlock(d_model) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for block in self.blk:
            x = block(x)
        x = self.final_ln(x)
        return self.out_proj(x)
