"""Small dependency-free binary probability metrics."""

from __future__ import annotations
import math
from typing import Any
import numpy as np


def average_precision(score: np.ndarray, target: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_target = target[order]
    boundaries = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    cumulative_positive = np.cumsum(sorted_target)[boundaries]
    recall = cumulative_positive / positives
    precision = cumulative_positive / (boundaries + 1)
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def auroc(score: np.ndarray, target: np.ndarray) -> float:
    positive = target == 1
    n_positive = int(positive.sum())
    n_negative = len(target) - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def reliability(score: np.ndarray, target: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    order = np.argsort(score)
    result = []
    for index, selected in enumerate(np.array_split(order, min(bins, len(order)))):
        result.append(
            {
                "bin": index,
                "examples": len(selected),
                "mean_probability": float(score[selected].mean()),
                "observed_rate": float(target[selected].mean()),
                "positive_examples": int(target[selected].sum()),
            }
        )
    return result


def binary_metrics(score: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(score) != len(target) or not len(score):
        raise ValueError("binary metrics need equally sized nonempty arrays")
    clipped = np.clip(score, 1e-9, 1 - 1e-9)
    base_rate = float(target.mean())
    order = np.argsort(-score)
    top_one = max(1, round(0.01 * len(score)))
    top_ten = max(1, round(0.10 * len(score)))
    rows = reliability(score, target)
    return {
        "examples": len(score),
        "positive_examples": int(target.sum()),
        "base_rate": base_rate,
        "brier": float(np.square(score - target).mean()),
        "log_loss": float(
            np.mean(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))
        ),
        "auprc": average_precision(score, target),
        "auroc": auroc(score, target),
        "top_1pct_rate": float(target[order[:top_one]].mean()),
        "top_1pct_lift": float(target[order[:top_one]].mean() / max(base_rate, 1e-12)),
        "top_10pct_rate": float(target[order[:top_ten]].mean()),
        "top_10pct_lift": float(target[order[:top_ten]].mean() / max(base_rate, 1e-12)),
        "ece_10_equal_count_bins": float(
            sum(
                row["examples"] / len(score) * abs(row["mean_probability"] - row["observed_rate"])
                for row in rows
            )
        ),
        "reliability": rows,
    }
