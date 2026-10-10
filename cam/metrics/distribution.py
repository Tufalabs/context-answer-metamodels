"""PCA, likelihood, and projected distribution distances."""

from __future__ import annotations
import math
import torch
from cam.models import flow as flow_base

HIDDEN = 4096


LOG_2PI = math.log(2 * math.pi)


def fit_pca(
    values: torch.Tensor,
    rank: int,
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if values.ndim != 2 or len(values) <= rank:
        raise ValueError(f"PCA needs a matrix with rows > rank, received {tuple(values.shape)}")
    torch.manual_seed(seed)
    value = values.to(device=device, dtype=torch.float32)
    center = value.mean(0)
    centered = value - center
    _, singular, basis = torch.pca_lowrank(centered, q=rank, center=False, niter=3)
    eigenvalues = singular.square() / (len(value) - 1)
    total_variance = centered.square().sum() / (len(value) - 1)
    retained = float(eigenvalues.sum() / total_variance.clamp_min(1e-30))
    return center.cpu(), basis.cpu(), eigenvalues.cpu(), retained


def kernel_mmd_squared(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-context nonnegative multiscale RBF MMD^2 V-statistic."""
    rank = left.shape[-1]
    xx = torch.cdist(left, left).square()
    yy = torch.cdist(right, right).square()
    xy = torch.cdist(left, right).square()
    result = torch.zeros(len(left), device=left.device)
    for multiplier in (0.5, 1.0, 2.0):
        bandwidth_squared = rank * multiplier**2
        kernel_xx = torch.exp(-xx / (2 * bandwidth_squared))
        kernel_yy = torch.exp(-yy / (2 * bandwidth_squared))
        kernel_xy = torch.exp(-xy / (2 * bandwidth_squared))
        result += kernel_xx.mean((1, 2)) + kernel_yy.mean((1, 2))
        result -= 2 * kernel_xy.mean((1, 2))
    return result / 3


def sliced_wasserstein(
    left: torch.Tensor,
    right: torch.Tensor,
    directions: torch.Tensor,
    quantiles: int = 64,
) -> torch.Tensor:
    """Per-context sliced W2 in whitened PCA coordinates."""
    left_1d = left @ directions
    right_1d = right @ directions
    grid = torch.linspace(0, 1, quantiles, device=left.device)
    left_quantile = torch.quantile(left_1d, grid, dim=1)
    right_quantile = torch.quantile(right_1d, grid, dim=1)
    return (left_quantile - right_quantile).square().mean((0, 2)).sqrt()


def distribution_distances(
    left: torch.Tensor,
    right: torch.Tensor,
    eigenvalues: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = eigenvalues.sqrt().clamp_min(1e-6).to(left.device)
    left_white = left / scale
    right_white = right / scale
    return (
        kernel_mmd_squared(left_white, right_white),
        sliced_wasserstein(left_white, right_white, directions),
    )


def score_projected_distributions(
    answer_real: torch.Tensor,
    answer_flow: torch.Tensor,
    residual_real: torch.Tensor,
    residual_flow: torch.Tensor,
    answer_eigenvalues: torch.Tensor,
    residual_eigenvalues: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=answer_real.device).manual_seed(seed)
    rank = answer_real.shape[-1]
    directions = torch.randn((rank, 64), generator=generator, device=answer_real.device)
    directions /= torch.linalg.vector_norm(directions, dim=0, keepdim=True)
    half = answer_real.shape[1] // 2
    rollout_count = answer_real.shape[1]
    flow_groups = answer_flow.shape[1] // rollout_count
    # Equalize sample counts for every distance. The two real folds are compared
    # with each same-sized fold from every disjoint group of generated samples.
    predictive_mmd_folds = []
    predictive_sw_folds = []
    shape_mmd_folds = []
    shape_sw_folds = []
    flow_floor_mmd_folds = []
    flow_floor_sw_folds = []
    for group in range(flow_groups):
        group_start = group * rollout_count
        for fold in range(2):
            real_start, real_end = fold * half, (fold + 1) * half
            flow_start = group_start + fold * half
            flow_end = group_start + (fold + 1) * half
            predictive_mmd, predictive_sw = distribution_distances(
                answer_real[:, real_start:real_end],
                answer_flow[:, flow_start:flow_end],
                answer_eigenvalues,
                directions,
            )
            predictive_mmd_folds.append(predictive_mmd)
            predictive_sw_folds.append(predictive_sw)
            residual_real_fold = residual_real[:, real_start:real_end]
            residual_flow_fold = residual_flow[:, flow_start:flow_end]
            residual_real_fold = residual_real_fold - residual_real_fold.mean(1, keepdim=True)
            residual_flow_fold = residual_flow_fold - residual_flow_fold.mean(1, keepdim=True)
            shape_mmd, shape_sw = distribution_distances(
                residual_real_fold,
                residual_flow_fold,
                residual_eigenvalues,
                directions,
            )
            shape_mmd_folds.append(shape_mmd)
            shape_sw_folds.append(shape_sw)
        floor_mmd, floor_sw = distribution_distances(
            answer_flow[:, group_start : group_start + half],
            answer_flow[:, group_start + half : group_start + rollout_count],
            answer_eigenvalues,
            directions,
        )
        flow_floor_mmd_folds.append(floor_mmd)
        flow_floor_sw_folds.append(floor_sw)
    predictive_mmd = torch.stack(predictive_mmd_folds).mean(0)
    predictive_sw = torch.stack(predictive_sw_folds).mean(0)
    shape_mmd = torch.stack(shape_mmd_folds).mean(0)
    shape_sw = torch.stack(shape_sw_folds).mean(0)
    oracle_predictive_mmd, oracle_predictive_sw = distribution_distances(
        answer_real[:, :half], answer_real[:, half : 2 * half], answer_eigenvalues, directions
    )
    oracle_shape_left = residual_real[:, :half]
    oracle_shape_right = residual_real[:, half : 2 * half]
    oracle_shape_left = oracle_shape_left - oracle_shape_left.mean(1, keepdim=True)
    oracle_shape_right = oracle_shape_right - oracle_shape_right.mean(1, keepdim=True)
    oracle_shape_mmd, oracle_shape_sw = distribution_distances(
        oracle_shape_left, oracle_shape_right, residual_eigenvalues, directions
    )
    flow_floor_mmd = torch.stack(flow_floor_mmd_folds).mean(0)
    flow_floor_sw = torch.stack(flow_floor_sw_folds).mean(0)
    return {
        "predictive_mmd2": predictive_mmd,
        "predictive_sliced_w2": predictive_sw,
        "shape_mmd2": shape_mmd,
        "shape_sliced_w2": shape_sw,
        "real_split_predictive_mmd2": oracle_predictive_mmd,
        "real_split_predictive_sliced_w2": oracle_predictive_sw,
        "real_split_shape_mmd2": oracle_shape_mmd,
        "real_split_shape_sliced_w2": oracle_shape_sw,
        "flow_split_predictive_mmd2": flow_floor_mmd,
        "flow_split_predictive_sliced_w2": flow_floor_sw,
    }


def approximate_log_likelihood(
    model: flow_base.ConditionalFlow,
    condition: torch.Tensor,
    target: torch.Tensor,
    *,
    steps: int,
    probes: int,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Euler CNF likelihood using a Hutchinson trace of the learned vector field."""
    device = condition.device
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoded_all = model.encode_condition(condition).detach()
    generator = torch.Generator(device=device).manual_seed(seed)
    flow_nll, base_nll, latent_norm, inversion = [], [], [], []
    step_size = -1.0 / steps
    for start in range(0, len(target), batch_size):
        end = min(start + batch_size, len(target))
        state = target[start:end].detach().clone()
        encoded = encoded_all[start:end]
        integral_divergence = torch.zeros(len(state), device=device)
        for step in range(steps):
            state.requires_grad_(True)
            time_value = torch.full(
                (len(state),), 1.0 - step / steps, device=device, dtype=state.dtype
            )
            velocity = model(state, time_value, encoded)
            divergence = torch.zeros(len(state), device=device)
            for probe in range(probes):
                epsilon = (
                    torch.empty_like(state).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                )
                product = torch.autograd.grad(
                    velocity,
                    state,
                    grad_outputs=epsilon,
                    retain_graph=probe + 1 < probes,
                )[0]
                divergence += (product * epsilon).sum(1) / probes
            state = (state + step_size * velocity).detach()
            integral_divergence += step_size * divergence.detach()
        log_base = -0.5 * (state.square() + LOG_2PI).sum(1)
        log_flow = log_base + integral_divergence
        flow_nll.append(-log_flow / HIDDEN)
        selected_target = target[start:end]
        base_nll.append(0.5 * (selected_target.square() + LOG_2PI).mean(1))
        latent_norm.append(state.square().mean(1))
        with torch.inference_mode():
            reconstructed = state
            positive_step = 1.0 / steps
            for step in range(steps):
                time_value = torch.full(
                    (len(state),), step / steps, device=device, dtype=state.dtype
                )
                reconstructed = reconstructed + positive_step * model(
                    reconstructed, time_value, encoded
                )
            inversion.append((reconstructed - selected_target).square().mean(1).sqrt())
        print(f"likelihood {end}/{len(target)}", flush=True)
    return {
        "flow_nll_nats_per_dimension": torch.cat(flow_nll),
        "standard_normal_nll_nats_per_dimension": torch.cat(base_nll),
        "backward_latent_second_moment": torch.cat(latent_norm),
        "roundtrip_rmse_standardized": torch.cat(inversion),
    }
