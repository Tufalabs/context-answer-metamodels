"""Exact-token conditioners for point and conditional-flow metamodels."""

from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from cam.models.flow import ConditionalFlow

HIDDEN = 4096


class ExactTokenQueryConditioner(torch.nn.Module):
    """Single learned-query attention over all exact prompt tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.normalization = torch.nn.LayerNorm(HIDDEN)
        self.query = torch.nn.Parameter(torch.zeros(HIDDEN))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(sequence)
        scores = torch.einsum("btd,d->bt", normalized, self.query) / math.sqrt(HIDDEN)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = scores.softmax(dim=-1)
        return torch.einsum("bt,btd->bd", weights, sequence)


@dataclass(frozen=True)
class Architecture:
    name: str
    display_name: str
    family: str
    learning_rate: float
    mlp_weight_decay: float
    flow_weight_decay: float
    training_batch_size: int


ARCHITECTURES = (
    Architecture(
        "token_query",
        "Exact-token learned-query attention",
        "exact_token_query",
        3e-4,
        1e-3,
        1e-4,
        256,
    ),
)


def architecture(name: str) -> Architecture:
    return next(value for value in ARCHITECTURES if value.name == name)


def build_conditioner(name: str) -> torch.nn.Module:
    if name != "token_query":
        raise ValueError(f"conditioner is outside the paper scope: {name}")
    return ExactTokenQueryConditioner()


class PointMetamodel(torch.nn.Module):
    def __init__(self, conditioner_name: str, width: int = 8192) -> None:
        super().__init__()
        self.conditioner = build_conditioner(conditioner_name)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, HIDDEN),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.conditioner(sequence, mask))


class ExactTokenConditionalFlow(ConditionalFlow):
    def __init__(self, conditioner_name: str, width: int = 4096, blocks: int = 8) -> None:
        super().__init__(HIDDEN, width, blocks)
        self.token_conditioner = build_conditioner(conditioner_name)

    def encode_tokens(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.condition(self.token_conditioner(sequence, mask))
