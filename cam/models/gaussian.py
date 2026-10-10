"""Gaussian residual forecasts and validation-selected covariance models."""

from __future__ import annotations
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from cam.models import flow as flow_base
from cam.models.attention_flow import AttentionConditionalFlow
from cam.training import scaling as matrix
from cam.probes.attention import AttentionProbe

HIDDEN = 4096
SAMPLE_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


class ScaleHead(torch.nn.Module):
    """Small heteroscedastic variance head over a frozen pooled prompt state."""

    def __init__(self, hidden: int = HIDDEN, width: int = 256) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(hidden, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_mlp(
    checkpoint: Path, device: torch.device
) -> tuple[AttentionProbe, dict[str, torch.Tensor]]:
    state = load_file(checkpoint, device="cpu")
    values = {
        key.removeprefix("normalization."): value.to(device)
        for key, value in state.items()
        if key.startswith("normalization.")
    }
    model_state = {
        key: value for key, value in state.items() if not key.startswith("normalization.")
    }
    model = AttentionProbe(HIDDEN, 8192).to(device)
    model.load_state_dict(model_state)
    model.eval()
    if set(values) != {"x_mean", "x_scale", "y_mean", "y_scale"}:
        raise RuntimeError(f"incomplete MLP normalization in {checkpoint}: {sorted(values)}")
    return model, values


def load_flow(
    checkpoint: Path, normalization_path: Path, device: torch.device
) -> tuple[AttentionConditionalFlow, dict[str, torch.Tensor]]:
    model = AttentionConditionalFlow(32, 4096, 8).to(device)
    model.load_state_dict(load_file(checkpoint, device="cpu"))
    model.eval()
    values = {key: value.to(device) for key, value in load_file(normalization_path).items()}
    if set(values) != {"x_mean", "x_scale", "y_mean", "y_scale"}:
        raise RuntimeError(f"incomplete flow normalization in {normalization_path}")
    return model, values


@torch.inference_mode()
def mlp_outputs(
    model: AttentionProbe,
    values: dict[str, torch.Tensor],
    x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    pooled_features: list[torch.Tensor] = []
    for chunk in x.split(128):
        standardized = (chunk.to(device=device, dtype=torch.float32) - values["x_mean"]) / values[
            "x_scale"
        ]
        prediction, weights = model(standardized)
        pooled = torch.einsum("bt,btd->bd", weights, standardized)
        predictions.append((prediction * values["y_scale"] + values["y_mean"]).float().cpu())
        pooled_features.append(pooled.to(dtype=torch.bfloat16).cpu())
    return torch.cat(predictions), torch.cat(pooled_features)


def context_log_variance(
    y: torch.Tensor,
    positions: torch.Tensor,
    reference_spread: float | None = None,
) -> tuple[torch.Tensor, float]:
    selected = y.index_select(1, positions).float()
    centered = selected - selected.mean(1, keepdim=True)
    spread = centered.square().sum((1, 2)) / ((selected.shape[1] - 1) * HIDDEN)
    global_spread = float(spread.mean()) if reference_spread is None else reference_spread
    target = torch.log((spread / global_spread).clamp(1e-4, 1e4)).clamp(-6.0, 6.0)
    return target, global_spread


def feature_moments(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(features.shape[1], dtype=torch.float64)
    square = torch.zeros_like(total)
    for chunk in features.split(2048):
        value = chunk.double()
        total += value.sum(0)
        square += value.square().sum(0)
    mean = total / len(features)
    scale = (square / len(features) - mean.square()).clamp_min(1e-8).sqrt()
    return mean.float(), scale.float()


def fit_scale_head(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    validation_features: torch.Tensor,
    validation_target: torch.Tensor,
    device: torch.device,
    seed: int,
    max_steps: int,
) -> tuple[ScaleHead, dict[str, torch.Tensor], dict[str, Any]]:
    feature_mean, feature_scale = feature_moments(train_features)
    feature_mean = feature_mean.to(device)
    feature_scale = feature_scale.to(device)
    torch.manual_seed(seed)
    model = ScaleHead(hidden=train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    stale = 0
    curve = []
    evaluation_interval = 100
    patience = 10
    for step in range(1, max_steps + 1):
        index = torch.randint(len(train_features), (1024,), generator=generator)
        x = train_features.index_select(0, index).to(device=device, dtype=torch.float32)
        target = train_target.index_select(0, index).to(device)
        prediction = model((x - feature_mean) / feature_scale)
        loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % evaluation_interval and step != max_steps:
            continue
        model.eval()
        validation_predictions = []
        with torch.inference_mode():
            for chunk in validation_features.split(1024):
                value = chunk.to(device=device, dtype=torch.float32)
                validation_predictions.append(model((value - feature_mean) / feature_scale).cpu())
        validation_prediction = torch.cat(validation_predictions)
        validation_loss = float(
            torch.nn.functional.smooth_l1_loss(validation_prediction, validation_target)
        )
        curve.append({"step": step, "validation_smooth_l1": validation_loss})
        print(f"scale-head step={step} validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_step = step
            best_state = matrix.clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        model.train()
    if best_state is None:
        raise RuntimeError("scale head produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        train_log_parts = []
        for chunk in train_features.split(1024):
            value = chunk.to(device=device, dtype=torch.float32)
            train_log_parts.append(model((value - feature_mean) / feature_scale).cpu())
    train_log = torch.cat(train_log_parts).clamp(-6.0, 6.0)
    rms_normalizer = torch.exp(train_log).mean().sqrt().clamp_min(1e-6)
    values = {
        "feature_mean": feature_mean.cpu(),
        "feature_scale": feature_scale.cpu(),
        "rms_normalizer": rms_normalizer.cpu(),
    }
    training = {
        "architecture": "frozen-prompt-pool -> 256 SiLU -> scalar log variance",
        "best_step": best_step,
        "best_validation_smooth_l1": best_loss,
        "steps_run": curve[-1]["step"],
        "curve": curve,
    }
    return model, values, training


@torch.inference_mode()
def predict_context_scale(
    model: ScaleHead,
    values: dict[str, torch.Tensor],
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    result = []
    feature_mean = values["feature_mean"].to(device)
    feature_scale = values["feature_scale"].to(device)
    for chunk in features.split(1024):
        value = chunk.to(device=device, dtype=torch.float32)
        log_ratio = model((value - feature_mean) / feature_scale).clamp(-6.0, 6.0)
        result.append((torch.exp(0.5 * log_ratio) / values["rms_normalizer"].to(device)).cpu())
    return torch.cat(result).clamp(0.10, 10.0)


def fit_residual_model(
    y: torch.Tensor,
    positions: torch.Tensor,
    device: torch.device,
    seed: int,
    pca_contexts: int,
    max_rank: int,
) -> dict[str, torch.Tensor]:
    rollout_count = len(positions)
    correction = math.sqrt(rollout_count / (rollout_count - 1))
    diagonal_sum = torch.zeros(HIDDEN, dtype=torch.float64)
    row_count = 0
    for chunk in y.split(256):
        selected = chunk.index_select(1, positions).float()
        residual = (selected - selected.mean(1, keepdim=True)) * correction
        diagonal_sum += residual.double().square().sum((0, 1))
        row_count += residual.shape[0] * residual.shape[1]
    diagonal_variance = (diagonal_sum / row_count).float()
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(len(y), generator=generator)[: min(pca_contexts, len(y))]
    selected = y.index_select(0, chosen).index_select(1, positions).float()
    residual = ((selected - selected.mean(1, keepdim=True)) * correction).reshape(-1, HIDDEN)
    residual -= residual.mean(0, keepdim=True)
    bootstrap_bank = residual.to(dtype=torch.bfloat16).contiguous()
    pca_input = residual.to(device)
    print(
        f"fitting randomized PCA rows={len(pca_input):,} dimensions={HIDDEN} rank={max_rank}",
        flush=True,
    )
    _, singular, basis = torch.pca_lowrank(pca_input, q=max_rank, center=False, niter=2)
    eigenvalues = singular.square() / max(1, len(pca_input) - 1)
    return {
        "diagonal_variance": diagonal_variance,
        "basis": basis.cpu(),
        "eigenvalues": eigenvalues.cpu(),
        "bootstrap_bank": bootstrap_bank,
    }


def projection_matrix(seed: int, projections: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    value = torch.randn((projections, HIDDEN), generator=generator, device=device)
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True)


def sample_arrays(
    samples: torch.Tensor,
    target: torch.Tensor,
    projections: torch.Tensor,
) -> tuple[dict[str, np.ndarray], torch.Tensor]:
    components = flow_base.energy_components(samples, target)
    lower = torch.quantile(samples, 0.05, dim=1)
    upper = torch.quantile(samples, 0.95, dim=1)
    coverage = ((target >= lower[:, None]) & (target <= upper[:, None])).float().mean((1, 2))
    sample_projection = torch.einsum("bsd,pd->bsp", samples, projections)
    target_projection = torch.einsum("brd,pd->brp", target, projections)
    projection_cross = torch.abs(sample_projection[:, :, None] - target_projection[:, None]).mean(
        (1, 2, 3)
    )
    projection_pair = torch.abs(sample_projection - sample_projection.roll(1, dims=1)).mean((1, 2))
    arrays = {name: value.detach().cpu().double().numpy() for name, value in components.items()}
    arrays["marginal_90pct_coverage"] = coverage.cpu().double().numpy()
    arrays["projected_crps"] = (projection_cross - 0.5 * projection_pair).cpu().double().numpy()
    return arrays, samples.mean(1).cpu()


def combine_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        index = generator.integers(0, len(values), size=len(values))
        estimates[draw] = values[index].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize_arrays(
    arrays: dict[str, np.ndarray],
    prediction: torch.Tensor,
    target: torch.Tensor,
    seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    target_mean = target.float().mean(1)
    centered = target_mean - target_mean.mean(0)
    sse = (prediction.float() - target_mean).double().square().sum(1).numpy()
    tss = centered.double().square().sum(1).numpy()
    r2 = 1.0 - sse.sum() / tss.sum()
    energy = arrays["energy_score"] / math.sqrt(HIDDEN)
    energy_ci = bootstrap_mean_ci(energy, seed, bootstrap_draws)
    spread = arrays["real_variance_trace"]
    boundaries = np.quantile(spread, (1 / 3, 2 / 3))
    bins = np.digitize(spread, boundaries)
    result: dict[str, Any] = {
        "sample_mean_r2": float(r2),
        "raw_energy_score_per_sqrt_dimension": float(energy.mean()),
        "raw_energy_score_per_sqrt_dimension_ci_low": energy_ci[0],
        "raw_energy_score_per_sqrt_dimension_ci_high": energy_ci[1],
        "raw_oracle_energy_score_per_sqrt_dimension": float(
            arrays["oracle_energy_score"].mean() / math.sqrt(HIDDEN)
        ),
        "raw_distribution_energy_skill_over_collapsed_mean": float(
            1.0
            - arrays["energy_score"].mean() / arrays["collapsed_sample_mean_energy_score"].mean()
        ),
        "raw_generated_real_pair_distance_ratio": float(
            arrays["generated_pair_distance"].mean() / arrays["real_pair_distance"].mean()
        ),
        "raw_generated_real_variance_trace_ratio": float(
            arrays["generated_variance_trace"].mean() / arrays["real_variance_trace"].mean()
        ),
        "marginal_90pct_coverage": float(arrays["marginal_90pct_coverage"].mean()),
        "projected_crps": float(arrays["projected_crps"].mean()),
    }
    for index, label in enumerate(("low", "medium", "high")):
        result[f"raw_energy_{label}_spread_tercile"] = float(energy[bins == index].mean())
    return result


def gaussian_sampler(
    residual_model: dict[str, torch.Tensor],
    kind: str,
    rank: int,
    device: torch.device,
) -> Callable[[int, int, int], torch.Tensor]:
    diagonal = residual_model["diagonal_variance"].to(device)
    if kind == "isotropic":
        standard_deviation = diagonal.mean().sqrt()

        def sample(count: int, n_samples: int, seed: int) -> torch.Tensor:
            generator = torch.Generator(device=device).manual_seed(seed)
            return (
                torch.randn((count, n_samples, HIDDEN), generator=generator, device=device)
                * standard_deviation
            )

        return sample
    if kind == "diagonal":
        standard_deviation = diagonal.clamp_min(0).sqrt()

        def sample(count: int, n_samples: int, seed: int) -> torch.Tensor:
            generator = torch.Generator(device=device).manual_seed(seed)
            return (
                torch.randn((count, n_samples, HIDDEN), generator=generator, device=device)
                * standard_deviation
            )

        return sample
    if kind != "lowrank_diagonal" or rank <= 0:
        raise ValueError(f"invalid Gaussian sampler: kind={kind} rank={rank}")
    basis = residual_model["basis"][:, :rank].to(device)
    eigenvalues = residual_model["eigenvalues"][:rank].to(device)
    factor = basis * eigenvalues.sqrt()[None]
    explained_diagonal = (basis.square() * eigenvalues[None]).sum(1)
    remainder = (diagonal - explained_diagonal).clamp_min(0).sqrt()

    def sample(count: int, n_samples: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(seed)
        low = torch.randn((count * n_samples, rank), generator=generator, device=device) @ factor.T
        diagonal_noise = (
            torch.randn((count * n_samples, HIDDEN), generator=generator, device=device) * remainder
        )
        return (low + diagonal_noise).reshape(count, n_samples, HIDDEN)

    return sample


def evaluate_grid(
    sampler: Callable[[int, int, int], torch.Tensor],
    prediction: torch.Tensor,
    target: torch.Tensor,
    context_scales: dict[str, torch.Tensor],
    sample_scales: tuple[float, ...],
    projections: torch.Tensor,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> dict[tuple[str, float], tuple[dict[str, np.ndarray], torch.Tensor]]:
    parts: dict[tuple[str, float], list[dict[str, np.ndarray]]] = {
        (mode, scale): [] for mode in context_scales for scale in sample_scales
    }
    predictions: dict[tuple[str, float], list[torch.Tensor]] = {key: [] for key in parts}
    for start in range(0, len(target), 16):
        end = min(start + 16, len(target))
        base_noise = sampler(end - start, n_samples, seed + start)
        mean = prediction[start:end].to(device=device, dtype=torch.float32)
        target_chunk = target[start:end].to(device=device, dtype=torch.float32)
        for mode, context_scale in context_scales.items():
            local_scale = context_scale[start:end].to(device)[:, None, None]
            for scale in sample_scales:
                samples = mean[:, None] + scale * local_scale * base_noise
                arrays, sample_mean = sample_arrays(samples, target_chunk, projections)
                key = (mode, scale)
                parts[key].append(arrays)
                predictions[key].append(sample_mean)
    return {
        key: (combine_parts(value), torch.cat(predictions[key])) for key, value in parts.items()
    }


@torch.inference_mode()
def evaluate_flow(
    model: AttentionConditionalFlow,
    values: dict[str, torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    projections: torch.Tensor,
    device: torch.device,
    n_samples: int,
    n_steps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], torch.Tensor]:
    parts = []
    predictions = []
    for start in range(0, len(target), 8):
        end = min(start + 8, len(target))
        condition = (
            x[start:end].to(device=device, dtype=torch.float32) - values["x_mean"]
        ) / values["x_scale"]
        standardized = flow_base.sample_flow(model, condition, n_samples, n_steps, seed + start)
        samples = standardized * values["y_scale"][None, None] + values["y_mean"][None, None]
        arrays, sample_mean = sample_arrays(
            samples, target[start:end].to(device=device, dtype=torch.float32), projections
        )
        parts.append(arrays)
        predictions.append(sample_mean)
    return combine_parts(parts), torch.cat(predictions)
