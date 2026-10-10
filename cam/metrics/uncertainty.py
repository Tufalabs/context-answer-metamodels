"""Uncertainty rankings and bootstrap comparisons."""

from __future__ import annotations
import math
from typing import Any, Callable
import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Equivalent to scipy.stats.rankdata(method='average'), without SciPy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def top_indices(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(math.ceil(len(values) * fraction)))
    return np.argsort(np.asarray(values), kind="mergesort")[-count:]


def top_fraction_metrics(
    predicted: np.ndarray, observed: np.ndarray, fraction: float
) -> dict[str, float]:
    predicted_top = top_indices(predicted, fraction)
    observed_top = top_indices(observed, fraction)
    overlap = len(np.intersect1d(predicted_top, observed_top, assume_unique=True))
    expected_overlap = len(predicted_top) * len(observed_top) / len(predicted)
    precision = overlap / len(predicted_top)
    return {
        "top_fraction": fraction,
        "top_count": float(len(predicted_top)),
        "top_overlap": float(overlap),
        "top_precision": float(precision),
        "top_enrichment": float(overlap / expected_overlap),
    }


def ranking_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 1:
        raise ValueError(f"expected matching vectors, got {predicted.shape} and {observed.shape}")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ValueError("uncertainty arrays contain non-finite values")
    result = {
        "spearman": spearman(predicted, observed),
        "log1p_pearson": pearson(np.log1p(predicted), np.log1p(observed)),
        "predicted_variance_mean": float(predicted.mean()),
        "observed_variance_mean": float(observed.mean()),
        "variance_mean_ratio": float(predicted.mean() / observed.mean()),
    }
    result.update(top_fraction_metrics(predicted, observed, 0.10))
    return result


def bootstrap_ci(
    predicted: np.ndarray,
    observed: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    seed: int,
    draws: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = generator.integers(0, len(predicted), size=len(predicted))
        estimates[draw] = metric(predicted[selected], observed[selected])
    finite = estimates[np.isfinite(estimates)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))
