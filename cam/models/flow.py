"""Conditional flow model, sampling, and energy-score evaluation."""

from __future__ import annotations
import math
from itertools import combinations
from typing import Any
import numpy as np
import torch


class FlowBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.normalization = torch.nn.LayerNorm(width)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return state + self.network(self.normalization(state) + condition)


class ConditionalFlow(torch.nn.Module):
    """Conditional rectified-flow vector field in the full activation space."""

    def __init__(self, hidden: int, width: int, blocks: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.width = width
        self.state = torch.nn.Linear(hidden, width)
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(hidden, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.time = torch.nn.Sequential(
            torch.nn.Linear(64, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.blocks = torch.nn.ModuleList(FlowBlock(width) for _ in range(blocks))
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, hidden),
        )
        self.register_buffer("time_frequencies", torch.exp(torch.linspace(0, math.log(1_000), 32)))

    def time_embedding(self, value: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * value[:, None] * self.time_frequencies[None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)

    def encode_condition(self, value: torch.Tensor) -> torch.Tensor:
        return self.condition(value)

    def forward(
        self, state: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        value = self.state(state) + self.time(self.time_embedding(time_value))
        for block in self.blocks:
            value = block(value, condition)
        return self.output(value)


@torch.inference_mode()
def sample_flow(
    model: ConditionalFlow,
    condition: torch.Tensor,
    n_samples: int,
    n_steps: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    contexts = len(condition)
    state = torch.randn(
        (contexts, n_samples, model.hidden),
        generator=generator,
        device=condition.device,
        dtype=condition.dtype,
    )
    encoded = model.encode_condition(condition)
    encoded = encoded[:, None, :].expand(-1, n_samples, -1).reshape(contexts * n_samples, -1)
    state = state.reshape(contexts * n_samples, model.hidden)
    step_size = 1.0 / n_steps
    for step in range(n_steps):
        time_value = torch.full(
            (len(state),),
            step * step_size,
            device=state.device,
            dtype=state.dtype,
        )
        state.add_(step_size * model(state, time_value, encoded))
    return state.reshape(contexts, n_samples, model.hidden)


def energy_components(samples: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-context energy components; generated-pair term uses an unbiased pairing."""
    cross = torch.stack(
        [
            torch.linalg.vector_norm(samples - target[:, rollout : rollout + 1], dim=2).mean(1)
            for rollout in range(target.shape[1])
        ],
        dim=1,
    ).mean(1)
    generated_pair = torch.linalg.vector_norm(samples - samples.roll(1, dims=1), dim=2).mean(1)
    real_pair = torch.stack(
        [
            torch.linalg.vector_norm(target[:, left] - target[:, right], dim=1)
            for left, right in combinations(range(target.shape[1]), 2)
        ],
        dim=1,
    ).mean(1)
    sample_mean = samples.mean(1)
    point = torch.stack(
        [
            torch.linalg.vector_norm(sample_mean - target[:, rollout], dim=1)
            for rollout in range(target.shape[1])
        ],
        dim=1,
    ).mean(1)
    return {
        "cross_distance": cross,
        "generated_pair_distance": generated_pair,
        "real_pair_distance": real_pair,
        "energy_score": cross - 0.5 * generated_pair,
        "oracle_energy_score": 0.5 * real_pair,
        "energy_distance": 2 * cross - generated_pair - real_pair,
        "collapsed_sample_mean_energy_score": point,
        "generated_variance_trace": samples.var(dim=1, correction=1).sum(1),
        "real_variance_trace": target.var(dim=1, correction=1).sum(1),
    }


@torch.inference_mode()
def validation_energy(
    model: ConditionalFlow,
    x: torch.Tensor,
    y: torch.Tensor,
    n_samples: int,
    n_steps: int,
    context_batch: int,
    seed: int,
) -> float:
    total = 0.0
    for start in range(0, len(x), context_batch):
        end = min(start + context_batch, len(x))
        samples = sample_flow(model, x[start:end], n_samples, n_steps, seed + start)
        total += float(energy_components(samples, y[start:end])["energy_score"].sum())
    return total / (len(x) * math.sqrt(y.shape[-1]))


def mean_ci(values: np.ndarray, generator: np.random.Generator, draws: int) -> tuple[float, float]:
    n = len(values)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        estimates[draw] = values[generator.integers(0, n, size=n)].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    generator: np.random.Generator,
    draws: int,
) -> tuple[float, float]:
    n = len(numerator)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        index = generator.integers(0, n, size=n)
        estimates[draw] = numerator[index].mean() / denominator[index].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


@torch.inference_mode()
def evaluate_flow(
    model: ConditionalFlow,
    x: torch.Tensor,
    y_standardized: torch.Tensor,
    y_raw: torch.Tensor,
    y_mean: torch.Tensor,
    y_scale: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    context_batch: int,
    n_bootstrap: int,
    seed: int,
    sample_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]:
    arrays: dict[str, list[np.ndarray]] = {}
    prediction_parts: list[torch.Tensor] = []
    coverage_numerator = 0.0
    coverage_denominator = 0
    for start in range(0, len(x), context_batch):
        end = min(start + context_batch, len(x))
        samples_standardized = sample_flow(model, x[start:end], n_samples, n_steps, seed + start)
        if sample_scale != 1.0:
            sample_mean = samples_standardized.mean(1, keepdim=True)
            samples_standardized = sample_mean + sample_scale * (samples_standardized - sample_mean)
        target_standardized = y_standardized[start:end]
        standardized = energy_components(samples_standardized, target_standardized)
        samples_raw = samples_standardized * y_scale[None, None, :] + y_mean[None, None, :]
        target_raw = y_raw[start:end]
        raw = energy_components(samples_raw, target_raw)
        prediction_parts.append(samples_raw.mean(1).cpu())
        lower = torch.quantile(samples_standardized, 0.05, dim=1)
        upper = torch.quantile(samples_standardized, 0.95, dim=1)
        coverage_numerator += float(
            (
                (target_standardized >= lower[:, None]) & (target_standardized <= upper[:, None])
            ).sum()
        )
        coverage_denominator += target_standardized.numel()
        for prefix, values in (("standardized", standardized), ("raw", raw)):
            for name, value in values.items():
                arrays.setdefault(f"{prefix}_{name}", []).append(value.cpu().double().numpy())
    joined = {name: np.concatenate(parts) for name, parts in arrays.items()}
    prediction = torch.cat(prediction_parts)
    target_mean = y_raw.mean(1).cpu()
    target_center = target_mean.mean(0)
    context_sse = (prediction - target_mean).square().sum(1).double().numpy()
    context_tss = (target_mean - target_center).square().sum(1).double().numpy()
    mean_r2 = 1.0 - context_sse.sum() / context_tss.sum()
    mean_cosine = float(
        torch.nn.functional.cosine_similarity(prediction, target_mean, dim=1).mean()
    )
    generator = np.random.default_rng(seed + 1_000_000)
    r2_draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        index = generator.integers(0, len(context_sse), size=len(context_sse))
        r2_draws[draw] = 1.0 - context_sse[index].sum() / context_tss[index].sum()
    metrics: dict[str, Any] = {
        "test_flow_sample_mean_r2": float(mean_r2),
        "test_flow_sample_mean_r2_ci_low": float(np.quantile(r2_draws, 0.025)),
        "test_flow_sample_mean_r2_ci_high": float(np.quantile(r2_draws, 0.975)),
        "test_flow_sample_mean_cosine": mean_cosine,
        "test_marginal_90pct_coverage": coverage_numerator / coverage_denominator,
    }
    dimension_root = math.sqrt(y_raw.shape[-1])
    for prefix in ("standardized", "raw"):
        energy = joined[f"{prefix}_energy_score"]
        oracle = joined[f"{prefix}_oracle_energy_score"]
        energy_distance = joined[f"{prefix}_energy_distance"]
        point = joined[f"{prefix}_collapsed_sample_mean_energy_score"]
        generated_pair = joined[f"{prefix}_generated_pair_distance"]
        real_pair = joined[f"{prefix}_real_pair_distance"]
        generated_variance = joined[f"{prefix}_generated_variance_trace"]
        real_variance = joined[f"{prefix}_real_variance_trace"]
        metrics.update(
            {
                f"test_{prefix}_energy_score": float(energy.mean()),
                f"test_{prefix}_energy_score_per_sqrt_dimension": float(
                    energy.mean() / dimension_root
                ),
                f"test_{prefix}_oracle_energy_score": float(oracle.mean()),
                f"test_{prefix}_energy_distance": float(energy_distance.mean()),
                f"test_{prefix}_energy_distance_over_real_pair_distance": float(
                    energy_distance.mean() / real_pair.mean()
                ),
                f"test_{prefix}_collapsed_sample_mean_energy_score": float(point.mean()),
                f"test_{prefix}_distribution_energy_gain_over_collapsed_mean": float(
                    point.mean() - energy.mean()
                ),
                f"test_{prefix}_distribution_energy_skill_over_collapsed_mean": float(
                    1.0 - energy.mean() / point.mean()
                ),
                f"test_{prefix}_generated_real_pair_distance_ratio": float(
                    generated_pair.mean() / real_pair.mean()
                ),
                f"test_{prefix}_generated_real_variance_trace_ratio": float(
                    generated_variance.mean() / real_variance.mean()
                ),
            }
        )
        energy_low, energy_high = mean_ci(energy, generator, n_bootstrap)
        pair_low, pair_high = ratio_ci(generated_pair, real_pair, generator, n_bootstrap)
        variance_low, variance_high = ratio_ci(
            generated_variance, real_variance, generator, n_bootstrap
        )
        metrics.update(
            {
                f"test_{prefix}_energy_score_ci_low": energy_low,
                f"test_{prefix}_energy_score_ci_high": energy_high,
                f"test_{prefix}_generated_real_pair_distance_ratio_ci_low": pair_low,
                f"test_{prefix}_generated_real_pair_distance_ratio_ci_high": pair_high,
                f"test_{prefix}_generated_real_variance_trace_ratio_ci_low": variance_low,
                f"test_{prefix}_generated_real_variance_trace_ratio_ci_high": variance_high,
            }
        )
    return metrics, joined, prediction
