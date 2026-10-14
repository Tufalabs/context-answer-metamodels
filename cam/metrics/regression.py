"""Regression metrics and bootstrap intervals."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def pooled_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    residual_sum = torch.sum((target - prediction) ** 2)
    centered = target - target.mean(dim=0)
    total_sum = torch.sum(centered**2)
    r2 = float("nan") if float(total_sum) <= 1e-12 else float(1.0 - residual_sum / total_sum)
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=1, eps=1e-12)
    return r2, float(cosine.mean())


def bootstrap_ci(
    prediction: torch.Tensor,
    target: torch.Tensor,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    pred = prediction.double().numpy()
    true = target.double().numpy()
    n = len(true)
    row_residual = np.sum((true - pred) ** 2, axis=1)
    row_target_squared = np.sum(true**2, axis=1)
    row_cosine = np.sum(pred * true, axis=1) / (
        (np.linalg.norm(pred, axis=1) + 1e-12) * (np.linalg.norm(true, axis=1) + 1e-12)
    )
    counts = np.empty((n_bootstrap, n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for row in range(n_bootstrap):
        counts[row] = np.bincount(rng.integers(0, n, size=n), minlength=n)

    sum_target = counts @ true
    total_sum = counts @ row_target_squared - np.sum(sum_target**2, axis=1) / n
    residual_sum = counts @ row_residual
    r2_samples = 1.0 - residual_sum / total_sum
    cosine_samples = (counts @ row_cosine) / n
    point_r2, point_cosine = pooled_metrics(prediction.double(), target.double())

    def interval(point: float, samples: np.ndarray) -> dict[str, float]:
        return {
            "point": point,
            "low": float(np.quantile(samples, 0.025)),
            "high": float(np.quantile(samples, 0.975)),
        }

    return {
        "r2": interval(point_r2, r2_samples),
        "mean_cosine": interval(point_cosine, cosine_samples),
    }


def bootstrap_fields(
    prediction: torch.Tensor, target: torch.Tensor, n_bootstrap: int, seed: int
) -> dict[str, float]:
    interval = bootstrap_ci(prediction, target, n_bootstrap, seed)
    return {
        "test_r2_ci_low": interval["r2"]["low"],
        "test_r2_ci_high": interval["r2"]["high"],
        "test_mean_cosine_ci_low": interval["mean_cosine"]["low"],
        "test_mean_cosine_ci_high": interval["mean_cosine"]["high"],
    }
