"""Run grouped cells in the Qwen3.5 3x3 LMSYS/WeirdChat scaling matrix."""

from __future__ import annotations
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from cam.models import flow as flow_base
from cam.models.attention_flow import AttentionConditionalFlow
from cam.probes.attention import AttentionProbe

HIDDEN = 4096
SEQUENCE_BINS = 32
ROLLOUTS = 4
FAMILIES = ("linear", "mlp", "flow")
REGIMES = ("lmsys", "combined")
RIDGE_LAMBDAS = np.logspace(-2, 7, 19)
VALIDATION_INTERVAL = 500
VALIDATION_PATIENCE = 12
MAX_TRAINING_STEPS = 80_000


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_tensors_atomic(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, temporary)
    os.replace(temporary, path)


def file_rows(path: Path) -> int:
    stem = path.stem
    start, end = (int(value) for value in stem.rsplit("_", 2)[-2:])
    return end - start + 1


@dataclass
class DomainData:
    x: torch.Tensor
    y: torch.Tensor


def load_partition(
    root: Path, domain: str, partition: str, family: str, limit: int | None
) -> DomainData:
    paths = sorted((root / domain / partition).glob(f"{partition}_*.safetensors"))
    available = sum(file_rows(path) for path in paths)
    count = available if limit is None else min(limit, available)
    if count <= 0 or (limit is not None and count != limit):
        raise RuntimeError(f"{domain}/{partition}: requested {limit}, available {available}")
    x_shape = (count, HIDDEN) if family == "linear" else (count, SEQUENCE_BINS, HIDDEN)
    x_dtype = torch.float32 if family == "linear" else torch.bfloat16
    x = torch.empty(x_shape, dtype=x_dtype)
    y = torch.empty((count, ROLLOUTS, HIDDEN), dtype=torch.bfloat16)
    key = "x_last" if family == "linear" else "prompt_bins"
    loaded = 0
    for path in paths:
        if loaded == count:
            break
        take = min(file_rows(path), count - loaded)
        with safe_open(path, framework="pt", device="cpu") as handle:
            x[loaded : loaded + take].copy_(handle.get_tensor(key)[:take])
            y[loaded : loaded + take].copy_(handle.get_tensor("y_rollouts")[:take])
        loaded += take
    if loaded != count:
        raise RuntimeError(f"loaded {loaded} rows, expected {count}")
    print(f"loaded {domain}/{partition}: {count:,} rows", flush=True)
    return DomainData(x=x, y=y)


def r2_value(target: torch.Tensor, prediction: torch.Tensor) -> float:
    residual = (target.double() - prediction.double()).square().sum()
    centered = target.double() - target.double().mean(0)
    return float(1 - residual / centered.square().sum().clamp_min(1e-30))


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def domain_feature_moments(
    x: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(HIDDEN, device=device)
    square = torch.zeros(HIDDEN, device=device)
    count = 0
    for chunk in x.split(256):
        value = chunk.to(device=device, dtype=torch.float32)
        axes = (0, 1) if value.ndim == 3 else (0,)
        total += value.sum(axes)
        square += value.square().sum(axes)
        count += value.numel() // HIDDEN
    mean = total / count
    second = square / count
    return mean, second


def domain_target_moments(
    y: torch.Tensor, device: torch.device, point: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(HIDDEN, device=device)
    square = torch.zeros(HIDDEN, device=device)
    count = 0
    for chunk in y.split(512):
        value = chunk.to(device=device, dtype=torch.float32)
        if point:
            value = value.mean(1)
            total += value.sum(0)
            square += value.square().sum(0)
            count += len(value)
        else:
            total += value.sum((0, 1))
            square += value.square().sum((0, 1))
            count += len(value) * value.shape[1]
    return total / count, square / count


def normalization(
    lmsys: DomainData,
    weird: DomainData | None,
    family: str,
    device: torch.device,
    secondary_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    x_mean_l, x_second_l = domain_feature_moments(lmsys.x, device)
    y_mean_l, y_second_l = domain_target_moments(lmsys.y, device, family != "flow")
    if weird is None:
        x_mean, x_second = x_mean_l, x_second_l
        y_mean, y_second = y_mean_l, y_second_l
    else:
        if not 0 < secondary_weight < 1:
            raise ValueError(f"secondary_weight must be in (0,1), got {secondary_weight}")
        x_mean_w, x_second_w = domain_feature_moments(weird.x, device)
        y_mean_w, y_second_w = domain_target_moments(weird.y, device, family != "flow")
        primary_weight = 1 - secondary_weight
        x_mean = primary_weight * x_mean_l + secondary_weight * x_mean_w
        x_second = primary_weight * x_second_l + secondary_weight * x_second_w
        y_mean = primary_weight * y_mean_l + secondary_weight * y_mean_w
        y_second = primary_weight * y_second_l + secondary_weight * y_second_w
    x_scale = (x_second - x_mean.square()).clamp_min(1e-12).sqrt()
    if family == "mlp":
        y_scale = (y_second - y_mean.square()).mean().clamp_min(1e-12).sqrt()
    else:
        y_scale = (y_second - y_mean.square()).clamp_min(1e-12).sqrt()
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale}


def cpu_batch(
    data: DomainData,
    count: int,
    generator: torch.Generator,
    family: str,
    normalization_values: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.randint(len(data.x), (count,), generator=generator)
    x = data.x.index_select(0, index).to(device=device, dtype=torch.float32)
    y = data.y.index_select(0, index).to(device=device, dtype=torch.float32)
    x = (x - normalization_values["x_mean"]) / normalization_values["x_scale"]
    if family == "mlp":
        y = (y.mean(1) - normalization_values["y_mean"]) / normalization_values["y_scale"]
    else:
        y = (y - normalization_values["y_mean"]) / normalization_values["y_scale"]
    return x, y


def mixed_batch(
    lmsys: DomainData,
    weird: DomainData | None,
    batch_size: int,
    generator: torch.Generator,
    family: str,
    values: dict[str, torch.Tensor],
    device: torch.device,
    secondary_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weird is None:
        return cpu_batch(lmsys, batch_size, generator, family, values, device)
    if not 0 < secondary_weight < 1:
        raise ValueError(f"secondary_weight must be in (0,1), got {secondary_weight}")
    weird_count = max(1, min(batch_size - 1, round(batch_size * secondary_weight)))
    lmsys_count = batch_size - weird_count
    left = cpu_batch(lmsys, lmsys_count, generator, family, values, device)
    right = cpu_batch(weird, weird_count, generator, family, values, device)
    permutation = torch.randperm(batch_size, generator=generator)
    return (
        torch.cat((left[0], right[0])).index_select(0, permutation.to(device)),
        torch.cat((left[1], right[1])).index_select(0, permutation.to(device)),
    )


@torch.inference_mode()
def mlp_predict(
    model: AttentionProbe,
    data: DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    parts = []
    model.eval()
    for chunk in data.x.split(128):
        x = chunk.to(device=device, dtype=torch.float32)
        x = (x - values["x_mean"]) / values["x_scale"]
        prediction, _ = model(x)
        parts.append((prediction * values["y_scale"] + values["y_mean"]).cpu())
    return torch.cat(parts)


def validation_mlp(
    model: AttentionProbe,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    scores = {
        domain: r2_value(data.y.float().mean(1), mlp_predict(model, data, values, device))
        for domain, data in validations.items()
    }
    return float(np.mean(list(scores.values()))), scores


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 500) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        estimates[draw] = values[generator.integers(0, len(values), size=len(values))].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


@torch.inference_mode()
def retrieval_metrics(
    prediction: torch.Tensor,
    y: torch.Tensor,
    device: torch.device | None = None,
    query_batch: int = 128,
) -> dict[str, float | int]:
    """Retrieve each prompt's empirical answer mean from its evaluation-set bank."""
    if prediction.ndim != 2 or y.ndim != 3 or prediction.shape != (len(y), y.shape[-1]):
        raise ValueError(
            f"retrieval shapes must be prediction=[N,D], y=[N,R,D]; got "
            f"{tuple(prediction.shape)} and {tuple(y.shape)}"
        )
    if not len(prediction) or query_batch <= 0:
        raise ValueError("retrieval needs at least one query and a positive query batch")
    selected_device = device if device is not None else prediction.device
    candidates = y.float().mean(1).to(selected_device)
    candidate_norm = candidates.square().sum(1)
    correct = 0
    for start in range(0, len(prediction), query_batch):
        end = min(start + query_batch, len(prediction))
        query = prediction[start:end].to(device=selected_device, dtype=torch.float32)
        squared_distance = (
            query.square().sum(1, keepdim=True) + candidate_norm[None, :] - 2 * query @ candidates.T
        )
        retrieved = squared_distance.argmin(1)
        expected = torch.arange(start, end, device=selected_device)
        correct += int((retrieved == expected).sum())
    count = len(prediction)
    accuracy = correct / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (accuracy + z * z / (2 * count)) / denominator
    half_width = (
        z * math.sqrt(accuracy * (1 - accuracy) / count + z * z / (4 * count * count)) / denominator
    )
    return {
        "retrieval_top1_accuracy": accuracy,
        "retrieval_top1_accuracy_ci_low": max(0.0, center - half_width),
        "retrieval_top1_accuracy_ci_high": min(1.0, center + half_width),
        "n_retrieval_top1_correct": correct,
        "n_retrieval_candidates": count,
    }


def point_metrics(
    prediction: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    device: torch.device | None = None,
) -> dict[str, float | int]:
    target = y.float()
    target_mean = target.mean(1)
    context_center = target_mean - target_mean.mean(0)
    sse = (prediction - target_mean).double().square().sum(1).numpy()
    tss = context_center.double().square().sum(1).numpy()
    r2 = 1.0 - sse.sum() / tss.sum()
    distances = torch.linalg.vector_norm(prediction[:, None].float() - target, dim=2).mean(
        1
    ).double().numpy() / math.sqrt(HIDDEN)
    oracle_pairs = []
    for left in range(ROLLOUTS):
        for right in range(left + 1, ROLLOUTS):
            oracle_pairs.append(torch.linalg.vector_norm(target[:, left] - target[:, right], dim=1))
    oracle = 0.5 * torch.stack(oracle_pairs, 1).mean(1).double().numpy() / math.sqrt(HIDDEN)
    generator = np.random.default_rng(seed)
    r2_draws = np.empty(500)
    for draw in range(len(r2_draws)):
        index = generator.integers(0, len(sse), size=len(sse))
        r2_draws[draw] = 1.0 - sse[index].sum() / tss[index].sum()
    energy_low, energy_high = bootstrap_interval(distances, seed + 1)
    result: dict[str, float | int] = {
        "sample_mean_r2": float(r2),
        "sample_mean_r2_ci_low": float(np.quantile(r2_draws, 0.025)),
        "sample_mean_r2_ci_high": float(np.quantile(r2_draws, 0.975)),
        "sample_mean_cosine": float(
            torch.nn.functional.cosine_similarity(prediction.float(), target_mean, dim=1).mean()
        ),
        "raw_energy_score_per_sqrt_dimension": float(distances.mean()),
        "raw_energy_score_per_sqrt_dimension_ci_low": energy_low,
        "raw_energy_score_per_sqrt_dimension_ci_high": energy_high,
        "raw_oracle_energy_score_per_sqrt_dimension": float(oracle.mean()),
    }
    result.update(retrieval_metrics(prediction, y, device))
    return result


def evaluation_sets(
    lmsys_train: DomainData,
    weird_train: DomainData | None,
    lmsys_test: DomainData,
    weird_test: DomainData,
) -> list[tuple[str, str, DomainData]]:
    result = [
        ("train", "lmsys", DomainData(lmsys_train.x[:1000], lmsys_train.y[:1000])),
        ("test", "lmsys", lmsys_test),
        ("test", "weirdchat", weird_test),
    ]
    if weird_train is not None:
        result.insert(
            1,
            (
                "train",
                "weirdchat",
                DomainData(weird_train.x[:1000], weird_train.y[:1000]),
            ),
        )
    return result


def fit_mlp(
    lmsys: DomainData,
    weird: DomainData | None,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    secondary_weight: float = 0.5,
) -> tuple[AttentionProbe, dict[str, Any]]:
    torch.manual_seed(seed)
    model = AttentionProbe(HIDDEN, 8192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    best_score = -math.inf
    best_state = None
    best_step = 0
    stale = 0
    curve = []
    plateaued = False
    stopping_step = 0
    for step in range(1, MAX_TRAINING_STEPS + 1):
        x, target = mixed_batch(
            lmsys, weird, 256, generator, "mlp", values, device, secondary_weight
        )
        model.train()
        prediction, _ = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % VALIDATION_INTERVAL:
            continue
        score, by_domain = validation_mlp(model, validations, values, device)
        curve.append(
            {
                "step": step,
                "train_mse": float(loss.detach()),
                "validation_macro_r2": score,
                "validation_r2_by_domain": by_domain,
            }
        )
        print(f"MLP step={step} loss={float(loss):.6f} val_macro_r2={score:.6f}", flush=True)
        if score > best_score + 1e-6:
            best_score, best_step, best_state, stale = score, step, clone_state(model), 0
        else:
            stale += 1
            if stale >= VALIDATION_PATIENCE:
                plateaued = True
                stopping_step = step
                break
    if best_state is None:
        raise RuntimeError("MLP did not produce a checkpoint")
    if not plateaued:
        raise RuntimeError(f"MLP validation did not plateau within {MAX_TRAINING_STEPS:,} steps")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_macro_r2": best_score,
        "curve": curve,
        "stop_reason": "validation_plateau",
        "stopping_step": stopping_step,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_patience": VALIDATION_PATIENCE,
        "maximum_steps_guard": MAX_TRAINING_STEPS,
    }


def linear_design(
    lmsys: DomainData,
    weird: DomainData | None,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x_parts = [lmsys.x.to(device=device, dtype=torch.float64)]
    y_parts = [lmsys.y.float().mean(1).to(device=device, dtype=torch.float64)]
    weights = [torch.ones(len(lmsys.x), device=device, dtype=torch.float64)]
    if weird is not None:
        total = len(lmsys.x) + len(weird.x)
        weights[0].fill_(total / (2 * len(lmsys.x)))
        x_parts.append(weird.x.to(device=device, dtype=torch.float64))
        y_parts.append(weird.y.float().mean(1).to(device=device, dtype=torch.float64))
        weights.append(
            torch.full(
                (len(weird.x),), total / (2 * len(weird.x)), device=device, dtype=torch.float64
            )
        )
    x = torch.cat(x_parts)
    y = torch.cat(y_parts)
    weight = torch.cat(weights)
    denominator = weight.sum()
    x_mean = (weight[:, None] * x).sum(0) / denominator
    x_scale = torch.sqrt((weight[:, None] * (x - x_mean).square()).sum(0) / denominator).clamp_min(
        1e-9
    )
    y_mean = (weight[:, None] * y).sum(0) / denominator
    x = (x - x_mean) / x_scale
    y = y - y_mean
    return x, y, weight, x_mean, x_scale, y_mean


@torch.inference_mode()
def fit_linear(
    lmsys: DomainData,
    weird: DomainData | None,
    validations: dict[str, DomainData],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    x, y, weight, x_mean, x_scale, y_mean = linear_design(lmsys, weird, device)
    square_root = weight.sqrt()[:, None]
    x_weighted, y_weighted = x * square_root, y * square_root
    validation_tensors = {
        domain: (
            (data.x.to(device=device, dtype=torch.float64) - x_mean) / x_scale,
            data.y.float().mean(1).to(device=device, dtype=torch.float64),
        )
        for domain, data in validations.items()
    }
    if len(x) <= HIDDEN:
        space = "dual"
        gram = x_weighted @ x_weighted.T
        eigenvalues, basis = torch.linalg.eigh(gram)
        projected = basis.T @ y_weighted

        def weights_for(ridge_lambda: float) -> torch.Tensor:
            coefficients = basis @ (projected / (eigenvalues[:, None].clamp_min(0) + ridge_lambda))
            return x_weighted.T @ coefficients

    else:
        space = "primal"
        gram = x_weighted.T @ x_weighted
        eigenvalues, basis = torch.linalg.eigh(gram)
        projected = basis.T @ (x_weighted.T @ y_weighted)

        def weights_for(ridge_lambda: float) -> torch.Tensor:
            return basis @ (projected / (eigenvalues[:, None].clamp_min(0) + ridge_lambda))

    trace = []
    best = None
    for ridge_lambda in RIDGE_LAMBDAS:
        model_weight = weights_for(float(ridge_lambda))
        scores = {
            domain: r2_value(target, design @ model_weight + y_mean)
            for domain, (design, target) in validation_tensors.items()
        }
        macro = float(np.mean(list(scores.values())))
        trace.append({"lambda": float(ridge_lambda), "macro_r2": macro, "r2_by_domain": scores})
        if best is None or macro > best[0]:
            best = (macro, float(ridge_lambda), model_weight)
    assert best is not None
    state = {
        "weight": best[2].float().cpu(),
        "x_mean": x_mean.float().cpu(),
        "x_scale": x_scale.float().cpu(),
        "y_mean": y_mean.float().cpu(),
    }
    return state, {
        "solution_space": space,
        "selected_lambda": best[1],
        "best_validation_macro_r2": best[0],
        "lambda_trace": trace,
        "stop_reason": "closed_form_validation_selection",
    }


@torch.inference_mode()
def linear_predict(
    state: dict[str, torch.Tensor], data: DomainData, device: torch.device
) -> torch.Tensor:
    weight = state["weight"].to(device)
    x_mean, x_scale, y_mean = (state[key].to(device) for key in ("x_mean", "x_scale", "y_mean"))
    parts = []
    for chunk in data.x.split(1024):
        x = chunk.to(device)
        parts.append((((x - x_mean) / x_scale) @ weight + y_mean).cpu())
    return torch.cat(parts)


def flow_validation(
    model: AttentionConditionalFlow,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[float, dict[str, float]]:
    scores = {}
    model.eval()
    for offset, (domain, data) in enumerate(validations.items()):
        x = (data.x.to(device=device, dtype=torch.float32) - values["x_mean"]) / values["x_scale"]
        y = (data.y.to(device=device, dtype=torch.float32) - values["y_mean"]) / values["y_scale"]
        scores[domain] = flow_base.validation_energy(
            model, x, y, 16, 16, 16, seed + 10_000 * offset
        )
    return float(np.mean(list(scores.values()))), scores


def fit_flow(
    lmsys: DomainData,
    weird: DomainData | None,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    secondary_weight: float = 0.5,
) -> tuple[AttentionConditionalFlow, dict[str, Any]]:
    torch.manual_seed(seed)
    model = AttentionConditionalFlow(SEQUENCE_BINS, 4096, 8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    cpu_generator = torch.Generator().manual_seed(seed + 1)
    cuda_generator = torch.Generator(device=device).manual_seed(seed + 2)
    best_energy = math.inf
    best_state = None
    best_step = 0
    stale = 0
    curve = []
    plateaued = False
    stopping_step = 0
    for step in range(1, MAX_TRAINING_STEPS + 1):
        condition, targets = mixed_batch(
            lmsys,
            weird,
            256,
            cpu_generator,
            "flow",
            values,
            device,
            secondary_weight,
        )
        rollout = torch.randint(
            ROLLOUTS, (len(condition),), generator=cuda_generator, device=device
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
        if step % VALIDATION_INTERVAL:
            continue
        # Common validation noise across checkpoints makes the plateau test
        # reflect model changes rather than a fresh Monte Carlo draw each time.
        energy, by_domain = flow_validation(model, validations, values, device, seed + 50_000_000)
        curve.append(
            {
                "step": step,
                "train_flow_mse": float(loss.detach()),
                "gradient_norm": float(gradient),
                "validation_macro_energy": energy,
                "validation_energy_by_domain": by_domain,
            }
        )
        print(f"flow step={step} loss={float(loss):.6f} val_macro_energy={energy:.6f}", flush=True)
        if energy < best_energy - 1e-5:
            best_energy, best_step, best_state, stale = energy, step, clone_state(model), 0
        else:
            stale += 1
            if stale >= VALIDATION_PATIENCE:
                plateaued = True
                stopping_step = step
                break
    if best_state is None:
        raise RuntimeError("flow did not produce a checkpoint")
    if not plateaued:
        raise RuntimeError(f"flow validation did not plateau within {MAX_TRAINING_STEPS:,} steps")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_macro_energy": best_energy,
        "curve": curve,
        "stop_reason": "validation_plateau",
        "stopping_step": stopping_step,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_patience": VALIDATION_PATIENCE,
        "maximum_steps_guard": MAX_TRAINING_STEPS,
    }


@torch.inference_mode()
def flow_metrics(
    model: AttentionConditionalFlow,
    data: DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    x = (data.x.to(device=device, dtype=torch.float32) - values["x_mean"]) / values["x_scale"]
    y_raw = data.y.to(device=device, dtype=torch.float32)
    y = (y_raw - values["y_mean"]) / values["y_scale"]
    metrics, _, prediction = flow_base.evaluate_flow(
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
    result = {key.removeprefix("test_"): value for key, value in metrics.items()}
    for suffix in ("r2", "r2_ci_low", "r2_ci_high", "cosine"):
        source = f"flow_sample_mean_{suffix}"
        if source in result:
            result[f"sample_mean_{suffix}"] = result.pop(source)
    result.update(retrieval_metrics(prediction, data.y, device))
    return result


def run_size(
    family: str,
    regime: str,
    n_train: int,
    loaded_train: DomainData,
    weird_train: DomainData | None,
    validations_all: dict[str, DomainData],
    tests: dict[str, DomainData],
    output: Path,
    seed: int,
    secondary_weight: float = 0.5,
) -> None:
    cell = output / family / regime / f"train_{n_train:06d}"
    result_path = cell / "results.json"
    if result_path.exists() and read_json(result_path).get("status") == "complete":
        print(f"skipping completed {family}/{regime}/{n_train}", flush=True)
        return
    started = time.monotonic()
    device = torch.device("cuda")
    lmsys = DomainData(loaded_train.x[:n_train], loaded_train.y[:n_train])
    combined_weird = (
        weird_train
        if regime == "combined" and weird_train is not None and len(weird_train.x) > 0
        else None
    )
    validations = validations_all if regime == "combined" else {"lmsys": validations_all["lmsys"]}
    evaluations = evaluation_sets(lmsys, combined_weird, tests["lmsys"], tests["weirdchat"])
    values = (
        normalization(lmsys, combined_weird, family, device, secondary_weight)
        if family != "linear"
        else {}
    )
    cell.mkdir(parents=True, exist_ok=True)
    if family == "linear":
        state, training = fit_linear(lmsys, combined_weird, validations, device)
        save_tensors_atomic(cell / "model.safetensors", state)
        evaluation_rows = []
        for index, (split, domain, data) in enumerate(evaluations):
            prediction = linear_predict(state, data, device)
            evaluation_rows.append(
                {
                    "split": split,
                    "domain": domain,
                    **point_metrics(prediction, data.y, seed + index, device),
                }
            )
        architecture = {
            "type": "closed-form adaptive ridge",
            "input": "exact last prompt token",
            "lambda_grid": RIDGE_LAMBDAS.tolist(),
        }
    elif family == "mlp":
        model, training = fit_mlp(
            lmsys,
            combined_weird,
            validations,
            values,
            device,
            seed,
            secondary_weight,
        )
        save_tensors_atomic(
            cell / "model.safetensors",
            {
                **model.state_dict(),
                **{f"normalization.{key}": value for key, value in values.items()},
            },
        )
        evaluation_rows = []
        for index, (split, domain, data) in enumerate(evaluations):
            prediction = mlp_predict(model, data, values, device)
            evaluation_rows.append(
                {
                    "split": split,
                    "domain": domain,
                    **point_metrics(prediction, data.y, seed + index, device),
                }
            )
        architecture = {
            "type": "learned-query attention probe + two-hidden-layer MLP",
            "input": "32 full-prompt activation bins",
            "width": 8192,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
        }
    else:
        model, training = fit_flow(
            lmsys,
            combined_weird,
            validations,
            values,
            device,
            seed,
            secondary_weight,
        )
        save_tensors_atomic(cell / "flow.safetensors", model.state_dict())
        save_tensors_atomic(cell / "normalization.safetensors", values)
        evaluation_rows = []
        for index, (split, domain, data) in enumerate(evaluations):
            evaluation_rows.append(
                {
                    "split": split,
                    "domain": domain,
                    **flow_metrics(model, data, values, device, seed + 100_000 * (index + 1)),
                }
            )
        architecture = {
            "type": "full-dimensional conditional rectified flow",
            "conditioner": "learned-query attention over 32 full-prompt activation bins",
            "width": 4096,
            "blocks": 8,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "test_samples": 64,
            "euler_steps": 32,
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "family": family,
        "training_regime": regime,
        "n_lmsys_train": n_train,
        "n_weirdchat_train": len(combined_weird.x) if combined_weird is not None else 0,
        "n_total_unique_train": n_train
        + (len(combined_weird.x) if combined_weird is not None else 0),
        "combined_domain_weighting": (
            f"{100 * (1 - secondary_weight):g}% LMSYS / {100 * secondary_weight:g}% WeirdChat"
            if combined_weird is not None
            else (
                "LMSYS only; proportional WeirdChat count rounded to zero"
                if regime == "combined"
                else None
            )
        ),
        "layer": 18,
        "rollout_seeds": [43, 44, 45, 46],
        "architecture": architecture,
        "training": training,
        "evaluation_protocols": {
            "retrieval_top1": {
                "version": 1,
                "query": "predicted answer activation mean",
                "candidate_bank": (
                    "four-rollout empirical answer means in the same split and domain"
                ),
                "distance": "raw activation-space squared Euclidean distance",
                "correctness": "nearest candidate has the same prompt index as the query",
            }
        },
        "evaluations": evaluation_rows,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    write_json_atomic(result_path, result)
    print(
        f"completed {family}/{regime} n_lmsys={n_train:,} elapsed={result['elapsed_seconds']:.1f}s",
        flush=True,
    )
    del values
    torch.cuda.empty_cache()
