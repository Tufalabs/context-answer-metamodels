"""Fit matched linear and Gaussian baselines for cross-model scaling."""

from __future__ import annotations
import math
from collections.abc import Callable
from typing import Any
import numpy as np
import torch
from cam.metrics import auxiliary
from cam.training import cross_model as base
from cam.metrics import rollouts as seed_scaling
from cam.models import gaussian as stochastic

GAUSSIAN_FAMILIES = ("isotropic", "diagonal", "lowrank_diagonal")
GAUSSIAN_RANKS = (64, 256, 512)
SAMPLE_SCALES = stochastic.SAMPLE_SCALES
N_SAMPLES = 64
SCALE_HEAD_STEPS = 5_000


def with_raw_prefix(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"raw_{key}": value for key, value in arrays.items()}


def distribution_metrics(
    arrays: dict[str, np.ndarray],
    prediction: torch.Tensor,
    target: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    summary = stochastic.summarize_arrays(
        arrays, prediction, target, seed=seed, bootstrap_draws=500
    )
    point = seed_scaling.point_metrics(prediction, target, seed + 1, device)
    raw_energy = arrays["energy_score"].astype(np.float64)
    raw_oracle = arrays["oracle_energy_score"].astype(np.float64)
    raw_collapsed = arrays["collapsed_sample_mean_energy_score"].astype(np.float64)
    generated = arrays["generated_variance_trace"].astype(np.float64)
    observed = arrays["real_variance_trace"].astype(np.float64)
    uncertainty = base.uncertainty_metrics(with_raw_prefix(arrays), seed + 2)
    point.update(
        {
            "test_raw_energy_score": float(raw_energy.mean()),
            "test_raw_energy_score_per_sqrt_dimension": float(
                raw_energy.mean() / math.sqrt(base.TARGET_HIDDEN)
            ),
            "test_raw_oracle_energy_score": float(raw_oracle.mean()),
            "test_raw_distribution_energy_skill_over_collapsed_mean": float(
                1.0 - raw_energy.mean() / raw_collapsed.mean()
            ),
            "test_raw_generated_real_variance_trace_ratio": float(
                generated.mean() / observed.mean()
            ),
            "uncertainty": uncertainty,
            "gaussian_summary": summary,
        }
    )
    return point


@torch.inference_mode()
def mlp_outputs(
    model: base.RectangularAttentionProbe,
    values: dict[str, torch.Tensor],
    x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    pooled_features: list[torch.Tensor] = []
    model.eval()
    for chunk in x.split(128):
        standardized = (chunk.to(device=device, dtype=torch.float32) - values["x_mean"]) / values[
            "x_scale"
        ]
        prediction, weights = model(standardized)
        pooled = torch.einsum("bt,btd->bd", weights, standardized)
        predictions.append((prediction * values["y_scale"] + values["y_mean"]).float().cpu())
        pooled_features.append(pooled.to(dtype=torch.bfloat16).cpu())
    return torch.cat(predictions), torch.cat(pooled_features)


def gaussian_specs(
    residual_model: dict[str, torch.Tensor], device: torch.device
) -> list[tuple[str, int, Callable[[int, int, int], torch.Tensor]]]:
    specs = [
        ("isotropic", 0, stochastic.gaussian_sampler(residual_model, "isotropic", 0, device)),
        ("diagonal", 0, stochastic.gaussian_sampler(residual_model, "diagonal", 0, device)),
    ]
    specs.extend(
        (
            "lowrank_diagonal",
            rank,
            stochastic.gaussian_sampler(residual_model, "lowrank_diagonal", rank, device),
        )
        for rank in GAUSSIAN_RANKS
    )
    return specs


def fit_gaussian(
    train: base.DomainData,
    validation: base.DomainData,
    test: base.DomainData,
    shuffled: base.DomainData,
    device: torch.device,
    seed: int,
    max_steps: int,
    auxiliary_reference: dict[str, torch.Tensor] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    int,
]:
    values = base.normalization(train, "mlp", device)
    model, mean_training = base.fit_mlp(train, validation, values, device, seed, max_steps)
    _, train_features = mlp_outputs(model, values, train.x, device)
    validation_prediction, validation_features = mlp_outputs(model, values, validation.x, device)
    rollout_positions = torch.arange(base.TRAIN_ROLLOUTS)
    train_scale_target, global_spread = stochastic.context_log_variance(train.y, rollout_positions)
    validation_scale_target, _ = stochastic.context_log_variance(
        validation.y, rollout_positions, global_spread
    )
    scale_head, scale_values, scale_training = stochastic.fit_scale_head(
        train_features,
        train_scale_target,
        validation_features,
        validation_scale_target,
        device,
        seed=seed + 30_000_000,
        max_steps=SCALE_HEAD_STEPS,
    )
    validation_context_scale = stochastic.predict_context_scale(
        scale_head, scale_values, validation_features, device
    )
    residual_model = stochastic.fit_residual_model(
        train.y,
        rollout_positions,
        device,
        seed=seed + 40_000_000,
        pca_contexts=min(8_192, len(train.y)),
        max_rank=max(GAUSSIAN_RANKS),
    )
    projections = stochastic.projection_matrix(seed + 50_000_000, 16, device)
    validation_scales = {
        "global": torch.ones(len(validation.y)),
        "heteroscedastic": validation_context_scale,
    }
    candidate_rows: list[dict[str, Any]] = []
    specs = gaussian_specs(residual_model, device)
    for family_index, (family, rank, sampler) in enumerate(specs):
        grid = stochastic.evaluate_grid(
            sampler,
            validation_prediction,
            validation.y,
            validation_scales,
            SAMPLE_SCALES,
            projections,
            device,
            N_SAMPLES,
            seed=seed + 60_000_000 + family_index * 10_000,
        )
        for (scale_mode, sample_scale), (arrays, prediction) in grid.items():
            metrics = stochastic.summarize_arrays(
                arrays,
                prediction,
                validation.y,
                seed=seed + 61_000_000 + family_index,
                bootstrap_draws=500,
            )
            candidate_rows.append(
                {
                    "family": family,
                    "rank": rank,
                    "scale_mode": scale_mode,
                    "sample_scale": sample_scale,
                    **metrics,
                }
            )
    selected_by_family_mode = []
    for family in GAUSSIAN_FAMILIES:
        for scale_mode in ("global", "heteroscedastic"):
            choices = [
                row
                for row in candidate_rows
                if row["family"] == family and row["scale_mode"] == scale_mode
            ]
            selected_by_family_mode.append(
                min(choices, key=lambda row: row["raw_energy_score_per_sqrt_dimension"])
            )
    selected = min(
        selected_by_family_mode,
        key=lambda row: row["raw_energy_score_per_sqrt_dimension"],
    )
    family = str(selected["family"])
    rank = int(selected["rank"])
    sampler = stochastic.gaussian_sampler(residual_model, family, rank, device)

    def evaluate(
        data: base.DomainData,
        prediction: torch.Tensor,
        features: torch.Tensor,
        eval_seed: int,
    ):
        context_scale = (
            torch.ones(len(data.y))
            if selected["scale_mode"] == "global"
            else stochastic.predict_context_scale(scale_head, scale_values, features, device)
        )
        grid = stochastic.evaluate_grid(
            sampler,
            prediction,
            data.y,
            {str(selected["scale_mode"]): context_scale},
            (float(selected["sample_scale"]),),
            projections,
            device,
            N_SAMPLES,
            seed=eval_seed,
        )
        arrays, sample_prediction = next(iter(grid.values()))
        metrics = distribution_metrics(arrays, sample_prediction, data.y, eval_seed + 1, device)
        return metrics, arrays

    test_prediction, test_features = mlp_outputs(model, values, test.x, device)
    shuffled_prediction, shuffled_features = mlp_outputs(model, values, shuffled.x, device)
    test_metrics, test_arrays = evaluate(test, test_prediction, test_features, seed + 70_000_000)
    if auxiliary_reference is not None:
        auxiliary_context_scale = (
            torch.ones(len(test.y))
            if selected["scale_mode"] == "global"
            else stochastic.predict_context_scale(scale_head, scale_values, test_features, device)
        )
        test_metrics["auxiliary"] = auxiliary.gaussian_metrics(
            sampler,
            test_prediction,
            auxiliary_context_scale,
            float(selected["sample_scale"]),
            residual_model,
            family,
            rank,
            test.y,
            auxiliary_reference,
            seed=seed + 75_000_000,
            device=device,
        )
    shuffled_metrics, _ = evaluate(
        shuffled, shuffled_prediction, shuffled_features, seed + 80_000_000
    )
    training = {
        **mean_training,
        "scale_head": scale_training,
        "selected_gaussian": selected,
        "selected_by_family_mode": selected_by_family_mode,
        "global_training_residual_variance_per_coordinate": global_spread,
    }
    architecture = {
        "mean": "32-bin learned-query attention + width-8192 MLP",
        "residual_family": family,
        "residual_rank": rank,
        "scale_mode": selected["scale_mode"],
        "sample_scale": selected["sample_scale"],
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters()) + sum(
        parameter.numel() for parameter in scale_head.parameters()
    )
    return training, architecture, test_metrics, shuffled_metrics, test_arrays, parameter_count
