"""Fit a prompt-conditioned predictor of held-out Qwen3.5 rollout activations."""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from cam.models import flow as flow_base
from cam.metrics import auxiliary
from cam.training import scaling as matrix
from cam.metrics import rollouts as seed_scaling
from cam.metrics.uncertainty import bootstrap_ci
from cam.metrics.uncertainty import ranking_metrics
from cam.metrics.uncertainty import spearman
from cam.metrics.uncertainty import top_fraction_metrics

TARGET_HIDDEN = 4_096
TRAIN_ROLLOUTS = 8
VALIDATION_INTERVAL = 500
VALIDATION_PATIENCE = 12


@dataclass
class DomainData:
    x: torch.Tensor
    y: torch.Tensor


class RectangularAttentionProbe(torch.nn.Module):
    """The established attention MLP with a source-width input and 4096-D target."""

    def __init__(self, condition_hidden: int, width: int = 8_192) -> None:
        super().__init__()
        self.query = torch.nn.Parameter(torch.zeros(condition_hidden))
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(condition_hidden, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, TARGET_HIDDEN),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = torch.nn.functional.layer_norm(sequence, (sequence.shape[-1],))
        scores = torch.einsum("btd,d->bt", normalized, self.query) / math.sqrt(sequence.shape[-1])
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, sequence)
        return self.predictor(pooled), weights


class RectangularAttentionConditionalFlow(flow_base.ConditionalFlow):
    """The established 4096-D flow conditioned on arbitrary-width prompt bins."""

    def __init__(self, condition_hidden: int, width: int = 4_096, blocks: int = 8) -> None:
        super().__init__(TARGET_HIDDEN, width, blocks)
        self.condition_hidden = condition_hidden
        self.attention_query = torch.nn.Parameter(torch.zeros(condition_hidden))
        # This is identical to the established flow when condition_hidden=4096.
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(condition_hidden, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )

    def encode_condition(self, value: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.layer_norm(value, (value.shape[-1],))
        scores = torch.einsum("btd,d->bt", normalized, self.attention_query) / math.sqrt(
            value.shape[-1]
        )
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, value)
        return self.condition(pooled)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_npz_atomic(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def selected_rollouts(data: DomainData, positions: torch.Tensor) -> DomainData:
    return DomainData(x=data.x, y=data.y.index_select(1, positions))


def feature_moments(x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x.shape[-1]
    total = torch.zeros(hidden, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    count = 0
    for chunk in x.split(256):
        value = chunk.to(device=device, dtype=torch.float32)
        total += value.double().sum((0, 1))
        square += value.double().square().sum((0, 1))
        count += value.shape[0] * value.shape[1]
    mean = total / count
    scale = (square / count - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(), scale.float()


def target_moments(
    y: torch.Tensor, device: torch.device, point: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(TARGET_HIDDEN, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    count = 0
    for chunk in y.split(256):
        value = chunk.to(device=device, dtype=torch.float32)
        if point:
            value = value.mean(1)
            total += value.double().sum(0)
            square += value.double().square().sum(0)
            count += len(value)
        else:
            total += value.double().sum((0, 1))
            square += value.double().square().sum((0, 1))
            count += value.shape[0] * value.shape[1]
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-12)
    scale = variance.mean().sqrt() if point else variance.sqrt()
    return mean.float(), scale.float()


def normalization(train: DomainData, family: str, device: torch.device) -> dict[str, torch.Tensor]:
    x_mean, x_scale = feature_moments(train.x, device)
    y_mean, y_scale = target_moments(train.y, device, point=family == "mlp")
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale}


def cpu_batch(
    data: DomainData,
    batch_size: int,
    generator: torch.Generator,
    family: str,
    values: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.randint(len(data.x), (batch_size,), generator=generator)
    x = data.x.index_select(0, selected).to(device=device, dtype=torch.float32)
    y = data.y.index_select(0, selected).to(device=device, dtype=torch.float32)
    x = (x - values["x_mean"]) / values["x_scale"]
    if family == "mlp":
        y = (y.mean(1) - values["y_mean"]) / values["y_scale"]
    else:
        y = (y - values["y_mean"]) / values["y_scale"]
    return x, y


@torch.inference_mode()
def mlp_predict(
    model: RectangularAttentionProbe,
    x: torch.Tensor,
    values: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    parts = []
    model.eval()
    for chunk in x.split(128):
        value = chunk.to(device=device, dtype=torch.float32)
        value = (value - values["x_mean"]) / values["x_scale"]
        prediction, _ = model(value)
        parts.append((prediction * values["y_scale"] + values["y_mean"]).cpu())
    return torch.cat(parts)


def fit_mlp(
    train: DomainData,
    validation: DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    max_steps: int,
) -> tuple[RectangularAttentionProbe, dict[str, Any]]:
    torch.manual_seed(seed)
    model = RectangularAttentionProbe(train.x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    stale = 0
    curve = []
    for step in range(1, max_steps + 1):
        x, target = cpu_batch(train, 256, generator, "mlp", values, device)
        model.train()
        prediction, _ = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % VALIDATION_INTERVAL and step != max_steps:
            continue
        raw_prediction = mlp_predict(model, validation.x, values, device)
        score = matrix.r2_value(validation.y.float().mean(1), raw_prediction)
        curve.append({"step": step, "training_mse": float(loss.detach()), "validation_r2": score})
        print(
            f"MLP step={step} loss={float(loss.detach()):.6f} val_r2={score:.6f}",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_step = step
            best_state = matrix.clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= VALIDATION_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("MLP produced no validation checkpoint")
    if stale < VALIDATION_PATIENCE and curve[-1]["step"] == max_steps:
        raise RuntimeError(f"MLP did not plateau within {max_steps:,} steps")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_r2": best_score,
        "steps_run": curve[-1]["step"],
        "stop_reason": "validation_plateau",
        "curve": curve,
    }


def flow_validation(
    model: RectangularAttentionConditionalFlow,
    validation: DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> float:
    x = (validation.x.to(device=device, dtype=torch.float32) - values["x_mean"]) / values["x_scale"]
    y = (validation.y.to(device=device, dtype=torch.float32) - values["y_mean"]) / values["y_scale"]
    return flow_base.validation_energy(model, x, y, 16, 16, 16, seed)


def fit_flow(
    train: DomainData,
    validation: DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    max_steps: int,
) -> tuple[RectangularAttentionConditionalFlow, dict[str, Any]]:
    torch.manual_seed(seed)
    model = RectangularAttentionConditionalFlow(train.x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    cpu_generator = torch.Generator().manual_seed(seed + 1)
    cuda_generator = torch.Generator(device=device).manual_seed(seed + 2)
    best_energy = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    stale = 0
    curve = []
    for step in range(1, max_steps + 1):
        condition, targets = cpu_batch(train, 256, cpu_generator, "flow", values, device)
        rollout = torch.randint(
            targets.shape[1], (len(condition),), generator=cuda_generator, device=device
        )
        target = targets[torch.arange(len(condition), device=device), rollout]
        noise = torch.randn(target.shape, generator=cuda_generator, device=device)
        time_value = torch.rand((len(target),), generator=cuda_generator, device=device)
        interpolated = (1 - time_value[:, None]) * noise + time_value[:, None] * target
        velocity = target - noise
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(interpolated, time_value, model.encode_condition(condition))
        loss = torch.nn.functional.mse_loss(prediction, velocity)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % VALIDATION_INTERVAL and step != max_steps:
            continue
        model.eval()
        energy = flow_validation(model, validation, values, device, seed + 50_000_000)
        curve.append(
            {
                "step": step,
                "training_flow_mse": float(loss.detach()),
                "gradient_norm": float(gradient),
                "validation_standardized_energy_per_sqrt_dimension": energy,
            }
        )
        print(
            f"flow step={step} loss={float(loss.detach()):.6f} val_energy={energy:.6f}",
            flush=True,
        )
        if energy < best_energy - 1e-5:
            best_energy = energy
            best_step = step
            best_state = matrix.clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= VALIDATION_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("flow produced no validation checkpoint")
    if stale < VALIDATION_PATIENCE and curve[-1]["step"] == max_steps:
        raise RuntimeError(f"flow did not plateau within {max_steps:,} steps")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_standardized_energy_per_sqrt_dimension": best_energy,
        "steps_run": curve[-1]["step"],
        "stop_reason": "validation_plateau",
        "curve": curve,
    }


def uncertainty_metrics(
    arrays: dict[str, np.ndarray], seed: int, bootstrap_draws: int = 500
) -> dict[str, float]:
    predicted = arrays["raw_generated_variance_trace"]
    observed = arrays["raw_real_variance_trace"]
    result = ranking_metrics(predicted, observed)
    result["spearman_ci_low"], result["spearman_ci_high"] = bootstrap_ci(
        predicted, observed, spearman, seed=seed, draws=bootstrap_draws
    )
    result["top_enrichment_ci_low"], result["top_enrichment_ci_high"] = bootstrap_ci(
        predicted,
        observed,
        lambda left, right: top_fraction_metrics(left, right, 0.10)["top_enrichment"],
        seed=seed + 1,
        draws=bootstrap_draws,
    )
    return result


@torch.inference_mode()
def evaluate_flow(
    model: RectangularAttentionConditionalFlow,
    data: DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]:
    x = (data.x.to(device=device, dtype=torch.float32) - values["x_mean"]) / values["x_scale"]
    y_raw = data.y.to(device=device, dtype=torch.float32)
    y = (y_raw - values["y_mean"]) / values["y_scale"]
    metrics, arrays, prediction = flow_base.evaluate_flow(
        model,
        x,
        y,
        y_raw,
        values["y_mean"],
        values["y_scale"],
        n_samples=64,
        n_steps=32,
        context_batch=16,
        n_bootstrap=500,
        seed=seed,
    )
    metrics.update(seed_scaling.point_metrics(prediction, data.y, seed + 10, device))
    metrics["uncertainty"] = uncertainty_metrics(arrays, seed + 20)
    return metrics, arrays, prediction
