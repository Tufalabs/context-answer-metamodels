"""Conditional flow with learned-query attention over prompt activations."""

from __future__ import annotations
import math
import torch
from cam.models import flow as base

TARGET_HIDDEN = 4096


class AttentionConditionalFlow(base.ConditionalFlow):
    """Full-dimensional flow with learned-query pooling of sequence bins."""

    def __init__(self, sequence_bins: int, width: int, blocks: int) -> None:
        del sequence_bins
        super().__init__(TARGET_HIDDEN, width, blocks)
        self.attention_query = torch.nn.Parameter(torch.zeros(TARGET_HIDDEN))

    def encode_condition(self, value: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.layer_norm(value, (value.shape[-1],))
        scores = torch.einsum("btd,d->bt", normalized, self.attention_query) / math.sqrt(
            value.shape[-1]
        )
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, value)
        return self.condition(pooled)
