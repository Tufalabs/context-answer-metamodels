"""Matched NLL, predictive MMD, and PCA Wasserstein diagnostics."""

from __future__ import annotations
import math
from collections.abc import Callable
from typing import Any
import torch
from cam.models import flow as flow_base
from cam.metrics import distribution as geometry

HIDDEN = 4096


LOG_2PI = math.log(2.0 * math.pi)


def _projected_real(
    target_raw: torch.Tensor, reference: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_common = (target_raw.float() - reference["common_y_mean"][None, None]) / reference[
        "common_y_scale"
    ][None, None]
    answer = (target_common - reference["answer_center"][None, None]) @ reference["answer_basis"]
    residual = target_common - target_common.mean(1, keepdim=True)
    residual = (residual - reference["residual_center"][None, None]) @ reference["residual_basis"]
    return target_common, answer, residual


@torch.inference_mode()
def _sample_and_project(
    sample_chunk: Callable[[int, int], torch.Tensor],
    target_raw: torch.Tensor,
    reference: dict[str, torch.Tensor],
    *,
    context_batch: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    target_common, answer_real, residual_real = _projected_real(target_raw, reference)
    answer_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    for start in range(0, len(target_raw), context_batch):
        end = min(start + context_batch, len(target_raw))
        generated_raw = sample_chunk(start, end).to(device=device, dtype=torch.float32)
        generated_common = (
            generated_raw - reference["common_y_mean"].to(device)[None, None]
        ) / reference["common_y_scale"].to(device)[None, None]
        answer_parts.append(
            (
                (generated_common - reference["answer_center"].to(device)[None, None])
                @ reference["answer_basis"].to(device)
            ).cpu()
        )
        generated_residual = generated_common - generated_common.mean(1, keepdim=True)
        residual_parts.append(
            (
                (generated_residual - reference["residual_center"].to(device)[None, None])
                @ reference["residual_basis"].to(device)
            ).cpu()
        )
    return {
        "answer_real": answer_real,
        "answer_generated": torch.cat(answer_parts),
        "residual_real": residual_real,
        "residual_generated": torch.cat(residual_parts),
        "target_common": target_common,
    }


def _distance_summary(
    projected: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, float]:
    pieces: dict[str, list[torch.Tensor]] = {}
    device = torch.device("cuda")
    for start in range(0, len(projected["answer_real"]), 32):
        end = min(start + 32, len(projected["answer_real"]))
        values = geometry.score_projected_distributions(
            projected["answer_real"][start:end].to(device),
            projected["answer_generated"][start:end].to(device),
            projected["residual_real"][start:end].to(device),
            projected["residual_generated"][start:end].to(device),
            reference["answer_eigenvalues"].to(device),
            reference["residual_eigenvalues"].to(device),
            seed + start,
        )
        for key, value in values.items():
            pieces.setdefault(key, []).append(value.cpu())
    means = {key: float(torch.cat(value).double().mean()) for key, value in pieces.items()}
    return {
        "predictive_mmd2": means["predictive_mmd2"],
        "predictive_sliced_w2": means["predictive_sliced_w2"],
        "real_split_predictive_mmd2": means["real_split_predictive_mmd2"],
        "real_split_predictive_sliced_w2": means["real_split_predictive_sliced_w2"],
    }


def flow_metrics(
    model: flow_base.ConditionalFlow,
    values: dict[str, torch.Tensor],
    condition_raw: torch.Tensor,
    target_raw: torch.Tensor,
    reference: dict[str, torch.Tensor],
    *,
    seed: int,
    n_samples: int = 64,
    flow_steps: int = 32,
    likelihood_contexts: int = 64,
    likelihood_steps: int = 32,
    hutchinson_probes: int = 2,
    context_batch: int = 8,
) -> dict[str, float]:
    """Evaluate a fixed flow in raw target and shared PCA coordinates."""
    device = next(model.parameters()).device
    condition = (condition_raw.float() - values["x_mean"].cpu()) / values["x_scale"].cpu()

    def sample_chunk(start: int, end: int) -> torch.Tensor:
        standardized = flow_base.sample_flow(
            model,
            condition[start:end].to(device),
            n_samples,
            flow_steps,
            seed + 10_000 + start,
        )
        return standardized * values["y_scale"][None, None] + values["y_mean"][None, None]

    projected = _sample_and_project(
        sample_chunk,
        target_raw,
        reference,
        context_batch=context_batch,
        device=device,
    )
    result = _distance_summary(projected, reference, seed=seed + 20_000)
    count = min(likelihood_contexts, len(target_raw))
    index = torch.linspace(0, len(target_raw) - 1, count, dtype=torch.long).unique()
    target_model = (target_raw.float() - values["y_mean"].cpu()[None, None]) / values[
        "y_scale"
    ].cpu()[None, None]
    likelihood = geometry.approximate_log_likelihood(
        model,
        condition.index_select(0, index).to(device),
        target_model.index_select(0, index)[:, 0].to(device),
        steps=likelihood_steps,
        probes=hutchinson_probes,
        batch_size=2,
        seed=seed + 30_000,
    )
    result["nll_raw"] = float(
        likelihood["flow_nll_nats_per_dimension"].double().mean()
        + values["y_scale"].double().log().mean()
    )
    return result


def _gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    context_scale: torch.Tensor,
    sample_scale: float,
    residual_model: dict[str, torch.Tensor],
    family: str,
    rank: int,
    device: torch.device,
) -> float:
    diagonal = residual_model["diagonal_variance"].float().clamp_min(1e-8)
    multiplier = (context_scale.float() * sample_scale).square().clamp_min(1e-8)
    if family == "isotropic":
        diagonal = diagonal.mean().expand_as(diagonal)
    if family in {"isotropic", "diagonal"}:
        variance = multiplier[:, None] * diagonal[None]
        value = 0.5 * (
            (target.float() - mean.float()).square() / variance + variance.log() + LOG_2PI
        ).mean(1)
        return float(value.double().mean())
    if family != "lowrank_diagonal":
        raise ValueError(f"unsupported Gaussian family for NLL: {family}")
    basis = residual_model["basis"][:, :rank].double().to(device)
    eigenvalues = residual_model["eigenvalues"][:rank].double().clamp_min(0).to(device)
    factor = basis * eigenvalues.sqrt()[None]
    diagonal = diagonal.double().to(device)
    jitter = diagonal.mean() * 1e-6
    remainder = (diagonal - (basis.square() * eigenvalues[None]).sum(1)).clamp_min(jitter)
    # Evaluate the scalar-scaled low-rank covariance with the determinant lemma
    # and Woodbury identity.  Float64 avoids the false non-positive-definite
    # failures seen in the generic batched float32 distribution constructor.
    inverse_diagonal_factor = factor / remainder[:, None]
    capacitance = torch.eye(rank, dtype=torch.float64, device=device)
    capacitance += factor.T @ inverse_diagonal_factor
    capacitance = 0.5 * (capacitance + capacitance.T)
    cholesky = torch.linalg.cholesky(capacitance)
    base_log_determinant = remainder.log().sum()
    base_log_determinant += 2.0 * cholesky.diagonal().log().sum()
    values = []
    for start in range(0, len(target), 4):
        end = min(start + 4, len(target))
        scale = multiplier[start:end].double().sqrt().to(device)
        residual = target[start:end].to(device=device, dtype=torch.float64) - mean[start:end].to(
            device=device, dtype=torch.float64
        )
        standardized = residual / scale[:, None]
        diagonal_quadratic = (standardized.square() / remainder[None]).sum(1)
        right = standardized @ inverse_diagonal_factor
        solved = torch.cholesky_solve(right.T, cholesky).T
        correction = (right * solved).sum(1)
        log_determinant = base_log_determinant + 2.0 * HIDDEN * scale.log()
        nll = 0.5 * (HIDDEN * LOG_2PI + log_determinant + diagonal_quadratic - correction)
        values.append((nll / HIDDEN).cpu())
    return float(torch.cat(values).mean())


def gaussian_metrics(
    sampler: Callable[[int, int, int], torch.Tensor],
    prediction: torch.Tensor,
    context_scale: torch.Tensor,
    sample_scale: float,
    residual_model: dict[str, torch.Tensor],
    family: str,
    rank: int,
    target_raw: torch.Tensor,
    reference: dict[str, torch.Tensor],
    *,
    seed: int,
    n_samples: int = 64,
    context_batch: int = 8,
    likelihood_contexts: int = 64,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a selected Gaussian residual model with the flow diagnostics."""

    def sample_chunk(start: int, end: int) -> torch.Tensor:
        noise = sampler(end - start, n_samples, seed + 10_000 + start)
        scale = context_scale[start:end].to(device)[:, None, None] * sample_scale
        return prediction[start:end].to(device)[:, None] + scale * noise

    projected = _sample_and_project(
        sample_chunk,
        target_raw,
        reference,
        context_batch=context_batch,
        device=device,
    )
    result = _distance_summary(projected, reference, seed=seed + 20_000)
    count = min(likelihood_contexts, len(target_raw))
    index = torch.linspace(0, len(target_raw) - 1, count, dtype=torch.long).unique()
    result["nll_raw"] = _gaussian_nll(
        target_raw.index_select(0, index)[:, 0],
        prediction.index_select(0, index),
        context_scale.index_select(0, index),
        sample_scale,
        residual_model,
        family,
        rank,
        device,
    )
    return result


def load_reference(path: Any) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return {key: value.float() for key, value in load_file(path, device="cpu").items()}
