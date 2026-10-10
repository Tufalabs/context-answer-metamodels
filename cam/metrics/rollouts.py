"""Rollout seed partitions and point-prediction metrics."""

from __future__ import annotations
import math
from itertools import combinations
import numpy as np
import torch
from cam.training import scaling as base

HIDDEN = 4096
ALL_SEEDS = tuple(range(43, 59))
SEED_ORDERS = (
    (43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 51, 52, 53, 54),
    (47, 55, 44, 58, 49, 43, 56, 46, 50, 57, 45, 48, 52, 54, 51, 53),
    (58, 46, 50, 44, 56, 48, 43, 55, 45, 57, 47, 49, 54, 51, 53, 52),
)


def seed_positions(seeds: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor([ALL_SEEDS.index(seed) for seed in seeds], dtype=torch.long)


def point_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    target = target.float()
    target_mean = target.mean(1)
    center = target_mean - target_mean.mean(0)
    sse = (prediction - target_mean).double().square().sum(1).numpy()
    tss = center.double().square().sum(1).numpy()
    r2 = 1.0 - sse.sum() / tss.sum()
    distances = torch.linalg.vector_norm(prediction[:, None].float() - target, dim=2).mean(
        1
    ).double().numpy() / math.sqrt(HIDDEN)
    pairs = torch.stack(
        [
            torch.linalg.vector_norm(target[:, left] - target[:, right], dim=1)
            for left, right in combinations(range(target.shape[1]), 2)
        ],
        dim=1,
    )
    oracle = 0.5 * pairs.mean(1).double().numpy() / math.sqrt(HIDDEN)
    generator = np.random.default_rng(seed)
    r2_draws = np.empty(500)
    energy_draws = np.empty(500)
    for draw in range(500):
        indices = generator.integers(0, len(sse), size=len(sse))
        r2_draws[draw] = 1.0 - sse[indices].sum() / tss[indices].sum()
        energy_draws[draw] = distances[indices].mean()
    result: dict[str, float | int] = {
        "sample_mean_r2": float(r2),
        "sample_mean_r2_ci_low": float(np.quantile(r2_draws, 0.025)),
        "sample_mean_r2_ci_high": float(np.quantile(r2_draws, 0.975)),
        "sample_mean_cosine": float(
            torch.nn.functional.cosine_similarity(prediction.float(), target_mean, dim=1).mean()
        ),
        "raw_energy_score_per_sqrt_dimension": float(distances.mean()),
        "raw_energy_score_per_sqrt_dimension_ci_low": float(np.quantile(energy_draws, 0.025)),
        "raw_energy_score_per_sqrt_dimension_ci_high": float(np.quantile(energy_draws, 0.975)),
        "raw_oracle_energy_score_per_sqrt_dimension": float(oracle.mean()),
    }
    result.update(base.retrieval_metrics(prediction, target, device))
    return result
