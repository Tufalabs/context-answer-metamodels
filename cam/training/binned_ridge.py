"""Fit ridge readouts over binned context activations."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F

HIDDEN = 4096
SEQUENCE_BINS = 32


@dataclass
class PointData:
    x: torch.Tensor
    y: torch.Tensor


def r2_from_sums(sse: torch.Tensor, target: torch.Tensor) -> float:
    centered = target.double() - target.double().mean(0)
    denominator = centered.square().sum().clamp_min(1e-30)
    return float(1 - sse.double() / denominator)


@torch.inference_mode()
def point_r2(
    weight: torch.Tensor,
    data: PointData,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    y_mean: torch.Tensor,
    representation: str,
    device: torch.device,
    batch_size: int,
) -> float:
    weight = weight.to(device=device, dtype=torch.float32)
    x_mean = x_mean.to(device=device, dtype=torch.float32)
    x_scale = x_scale.to(device=device, dtype=torch.float32)
    y_mean = y_mean.to(device=device, dtype=torch.float32)
    sse = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, len(data.x), batch_size):
        end = min(start + batch_size, len(data.x))
        x = data.x[start:end].to(device=device, dtype=torch.float32)
        if representation == "binned":
            x = ((x - x_mean) / x_scale).flatten(1)
        else:
            x = (x - x_mean) / x_scale
        prediction = F.linear(x, weight) + y_mean
        target = data.y[start:end].to(device=device, dtype=torch.float32)
        sse += (prediction.double() - target.double()).square().sum()
    return r2_from_sums(sse.cpu(), data.y)


def binned_moments(
    data: PointData, device: torch.device, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (SEQUENCE_BINS, HIDDEN)
    x_sum = torch.zeros(shape, device=device, dtype=torch.float64)
    x_square = torch.zeros_like(x_sum)
    y_sum = torch.zeros(HIDDEN, device=device, dtype=torch.float64)
    for start in range(0, len(data.x), chunk_size):
        end = min(start + chunk_size, len(data.x))
        x = data.x[start:end].to(device=device, dtype=torch.float32)
        y = data.y[start:end].to(device=device, dtype=torch.float32)
        x_sum += x.sum(0, dtype=torch.float64)
        x_square += x.square().sum(0, dtype=torch.float64)
        y_sum += y.sum(0, dtype=torch.float64)
    x_mean = x_sum / len(data.x)
    x_scale = (x_square / len(data.x) - x_mean.square()).clamp_min(1e-12).sqrt()
    y_mean = y_sum / len(data.x)
    return x_mean.float(), x_scale.float(), y_mean.float()


def train_binned_candidate(
    train: PointData,
    validation: PointData,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    y_mean: torch.Tensor,
    ridge_lambda: float,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> tuple[torch.Tensor, float, list[dict[str, float]]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    weight = torch.nn.Parameter(
        torch.zeros((HIDDEN, SEQUENCE_BINS * HIDDEN), device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([weight], lr=learning_rate)
    x_mean_device = x_mean.to(device)
    x_scale_device = x_scale.to(device)
    y_mean_device = y_mean.to(device)
    best_weight: torch.Tensor | None = None
    best_score = -math.inf
    stale = 0
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train.x), generator=generator)
        epoch_data_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            x = train.x[indices].to(device=device, dtype=torch.float32)
            y = train.y[indices].to(device=device, dtype=torch.float32)
            x = ((x - x_mean_device) / x_scale_device).flatten(1)
            y = y - y_mean_device
            optimizer.zero_grad(set_to_none=True)
            prediction = F.linear(x, weight)
            data_loss = (prediction - y).square().sum(1).mean()
            penalty = (ridge_lambda / len(train.x)) * weight.square().sum()
            loss = data_loss + penalty
            loss.backward()
            optimizer.step()
            epoch_data_loss += float(data_loss.detach()) * len(indices)
            seen += len(indices)
        validation_score = point_r2(
            weight,
            validation,
            x_mean,
            x_scale,
            y_mean,
            "binned",
            device,
            batch_size,
        )
        row = {
            "epoch": float(epoch),
            "training_sse_per_context": epoch_data_loss / seen,
            "validation_mean_r2": validation_score,
        }
        trace.append(row)
        print(
            f"lambda={ridge_lambda:g} epoch={epoch}/{epochs} "
            f"train_sse/context={row['training_sse_per_context']:.6g} "
            f"validation_r2={validation_score:.6f}",
            flush=True,
        )
        if not math.isfinite(validation_score):
            raise RuntimeError("non-finite validation R2 during binned linear training")
        if validation_score > best_score:
            best_score = validation_score
            best_weight = weight.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_weight is not None
    del optimizer, weight
    torch.cuda.empty_cache()
    return best_weight, best_score, trace


def fit_binned_linear(
    train: PointData,
    validation: PointData,
    ridge_lambdas: list[float],
    device: torch.device,
    batch_size: int,
    moment_chunk_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    x_mean, x_scale, y_mean = binned_moments(train, device, moment_chunk_size)
    best: tuple[float, float, torch.Tensor] | None = None
    candidates = []
    for index, ridge_lambda in enumerate(ridge_lambdas):
        weight, score, trace = train_binned_candidate(
            train,
            validation,
            x_mean,
            x_scale,
            y_mean,
            ridge_lambda,
            device,
            batch_size,
            epochs,
            learning_rate,
            patience,
            seed + index * 10_000,
        )
        candidates.append(
            {
                "lambda": ridge_lambda,
                "best_validation_mean_r2": score,
                "epoch_trace": trace,
            }
        )
        if best is None or score > best[0]:
            best = (score, ridge_lambda, weight)
        else:
            del weight
    assert best is not None
    normalization = {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean}
    training = {
        "solver": "minibatch Adam on the convex full 32-bin linear MSE plus L2 objective",
        "coefficient_count": HIDDEN * SEQUENCE_BINS * HIDDEN,
        "selected_lambda": best[1],
        "best_validation_mean_r2": best[0],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "early_stopping_patience": patience,
        "candidates": candidates,
    }
    return best[2], normalization, training
