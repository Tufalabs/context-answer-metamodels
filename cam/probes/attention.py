"""Train a learned-query attention probe over full-prompt activation bins."""

from __future__ import annotations
import math
import torch


class AttentionProbe(torch.nn.Module):
    def __init__(self, hidden: int, width: int) -> None:
        super().__init__()
        self.query = torch.nn.Parameter(torch.zeros(hidden))
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, hidden),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = torch.nn.functional.layer_norm(sequence, (sequence.shape[-1],))
        scores = torch.einsum("btd,d->bt", normalized, self.query) / math.sqrt(sequence.shape[-1])
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, sequence)
        return self.predictor(pooled), weights
