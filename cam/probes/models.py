"""Learned-query attention readout used by the paper’s context probes."""

from __future__ import annotations
import math
import torch

HIDDEN = 4096


class PromptRisk(torch.nn.Module):
    def __init__(self, hidden: int = HIDDEN, width: int = 512, events: int = 2) -> None:
        super().__init__()
        self.query = torch.nn.Parameter(torch.zeros(hidden))
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, events),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.layer_norm(sequence, (sequence.shape[-1],))
        scores = torch.einsum("btd,d->bt", normalized, self.query) / math.sqrt(sequence.shape[-1])
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, sequence)
        return self.predictor(pooled)
