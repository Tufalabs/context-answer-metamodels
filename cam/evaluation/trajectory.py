"""Snapshot one fixed fine-grid flow trajectory from Gaussian base to data."""

from __future__ import annotations
import numpy as np
import torch
from cam.models import flow as flow_base

DEFAULT_TOTAL_STEPS = 1_000
DEFAULT_SNAPSHOT_STEPS = (0, 50, 200, 500, 800, 1_000)
DEFAULT_METRIC_STEPS = (0, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000)


def record_snapshot(
    state: torch.Tensor,
    target_standardized: torch.Tensor,
    target_raw: torch.Tensor,
    y_mean: torch.Tensor,
    y_scale: torch.Tensor,
    projection_center: torch.Tensor,
    projection_basis: torch.Tensor,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    samples_raw = state * y_scale[None, None, :] + y_mean[None, None, :]
    components = flow_base.energy_components(samples_raw, target_raw)
    lower = torch.quantile(state, 0.05, dim=1)
    upper = torch.quantile(state, 0.95, dim=1)
    coverage = (
        ((target_standardized >= lower[:, None]) & (target_standardized <= upper[:, None]))
        .float()
        .mean(dim=(1, 2))
    )
    projected = (samples_raw - projection_center[None, None, :]) @ projection_basis
    arrays = {
        "energy": components["energy_score"].cpu().double().numpy(),
        "generated_variance": components["generated_variance_trace"].cpu().double().numpy(),
        "real_variance": components["real_variance_trace"].cpu().double().numpy(),
        "coverage": coverage.cpu().double().numpy(),
    }
    return arrays, projected.cpu().float().numpy()
