"""MLP predictors and validation-selected training recipes."""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from cam.metrics import regression as common  # noqa: E402


class Predictor(torch.nn.Module):
    def __init__(self, hidden: int, width: int, architecture: str) -> None:
        super().__init__()
        if architecture == "linear":
            self.network = torch.nn.Linear(hidden, hidden)
        elif architecture == "mlp1":
            self.network = torch.nn.Sequential(
                torch.nn.Linear(hidden, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, hidden),
            )
        elif architecture == "mlp2":
            self.network = torch.nn.Sequential(
                torch.nn.Linear(hidden, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, hidden),
            )
        else:
            raise ValueError(f"unknown architecture: {architecture}")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass(frozen=True)
class Recipe:
    architecture: str
    width: int
    learning_rate: float
    weight_decay: float

    @property
    def name(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("+", "")
        wd = f"{self.weight_decay:.0e}".replace("+", "")
        return f"{self.architecture}_w{self.width}_lr{lr}_wd{wd}"


def index_rows(value: torch.Tensor, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return value.index_select(0, torch.from_numpy(indices)).to(device, dtype=torch.float32)


def train_predictor(
    x: torch.Tensor,
    target: torch.Tensor,
    train: np.ndarray,
    validation: np.ndarray,
    evaluation: np.ndarray | None,
    recipe: Recipe,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    normalize_target_rms: bool = False,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:
    started = time.monotonic()
    x_train = index_rows(x, train, device)
    y_train = index_rows(target, train, device)
    x_validation = index_rows(x, validation, device)
    x_evaluation = index_rows(x, evaluation, device) if evaluation is not None else None
    x_mean = x_train.mean(0)
    x_scale = x_train.std(0, correction=0) + 1e-6
    y_mean = y_train.mean(0)
    x_train = (x_train - x_mean) / x_scale
    x_validation = (x_validation - x_mean) / x_scale
    if x_evaluation is not None:
        x_evaluation = (x_evaluation - x_mean) / x_scale
    y_train = y_train - y_mean
    y_scale = (
        torch.sqrt(torch.mean(y_train.square())).clamp_min(1e-6)
        if normalize_target_rms
        else torch.ones((), device=device)
    )
    y_train = y_train / y_scale

    permutation = np.random.default_rng(seed).permutation(len(train))
    n_internal_val = max(1, round(0.1 * len(train)))
    early = torch.from_numpy(permutation[:n_internal_val]).to(device)
    fit = torch.from_numpy(permutation[n_internal_val:]).to(device)
    x_fit, y_fit = x_train.index_select(0, fit), y_train.index_select(0, fit)
    x_early, y_early = x_train.index_select(0, early), y_train.index_select(0, early)

    torch.manual_seed(seed)
    model = Predictor(x.shape[1], recipe.width, recipe.architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state = None
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fit_prediction = model(x_fit)
        fit_loss = torch.nn.functional.mse_loss(fit_prediction, y_fit)
        fit_loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            early_loss = torch.nn.functional.mse_loss(model(x_early), y_early)
        fit_value, early_value = float(fit_loss.detach()), float(early_loss)
        curve.append(
            {
                "epoch": epoch,
                "training_mse": fit_value,
                "internal_validation_mse": early_value,
            }
        )
        if early_value < best_loss - 1e-6:
            best_loss = early_value
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_prediction = (model(x_validation) * y_scale + y_mean).float().cpu()
        evaluation_prediction = (
            (model(x_evaluation) * y_scale + y_mean).float().cpu()
            if x_evaluation is not None
            else None
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "architecture": recipe.architecture,
        "width": recipe.width,
        "learning_rate": recipe.learning_rate,
        "weight_decay": recipe.weight_decay,
        "target_rms_normalized": normalize_target_rms,
        "target_rms_scale": float(y_scale),
        "parameter_count": parameter_count,
        "epochs_run": len(curve),
        "best_epoch": best_epoch,
        "best_internal_validation_mse": best_loss,
        "elapsed_seconds": time.monotonic() - started,
        "loss_curve": curve,
    }
    return metadata, validation_prediction, evaluation_prediction


def candidate_path(
    output_dir: Path,
    input_name: str,
    layer: int,
    method: str,
    recipe: Recipe,
) -> Path:
    return (
        output_dir
        / "candidates"
        / input_name
        / f"layer_{layer:03d}"
        / method
        / (f"{recipe.name}.json")
    )


def metric(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    r2, cosine = common.pooled_metrics(prediction.double(), target.double())
    return {"r2": r2, "mean_cosine": cosine}


def choose_alpha(
    ridge_prediction: torch.Tensor,
    residual_prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    best_alpha = 0.0
    best_metric = metric(ridge_prediction, target)
    for alpha in np.linspace(0.0, 1.0, 11):
        candidate = ridge_prediction + float(alpha) * residual_prediction
        candidate_metric = metric(candidate, target)
        if candidate_metric["r2"] > best_metric["r2"]:
            best_alpha, best_metric = float(alpha), candidate_metric
    return best_alpha, best_metric


def run_grid(
    *,
    output_dir: Path,
    input_name: str,
    layer: int,
    method: str,
    grid: list[Recipe],
    x: torch.Tensor,
    training_target: torch.Tensor,
    validation_target: torch.Tensor,
    train: np.ndarray,
    validation: np.ndarray,
    ridge_validation: torch.Tensor | None,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    normalize_target_rms: bool = False,
) -> tuple[dict[str, Any], Recipe | None]:
    candidates = []
    for recipe in grid:
        path = candidate_path(output_dir, input_name, layer, method, recipe)
        if path.exists():
            result = common.read_json(path)
        else:
            print(
                f"input={input_name} layer={layer} method={method} recipe={recipe.name}",
                flush=True,
            )
            train_metadata, validation_prediction, _ = train_predictor(
                x,
                training_target,
                train,
                validation,
                None,
                recipe,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
                normalize_target_rms=normalize_target_rms,
            )
            if method == "residual":
                assert ridge_validation is not None
                alpha, validation_metric = choose_alpha(
                    ridge_validation,
                    validation_prediction,
                    validation_target,
                )
            else:
                alpha = 1.0
                validation_metric = metric(validation_prediction, validation_target)
            result = {
                **train_metadata,
                "input_measurement": input_name,
                "layer": layer,
                "method": method,
                "selected_residual_alpha": alpha,
                "validation_r2": validation_metric["r2"],
                "validation_mean_cosine": validation_metric["mean_cosine"],
            }
            common.write_json_atomic(path, result)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        candidates.append(result)
    selected = max(candidates, key=lambda row: row["validation_r2"])
    if method == "residual" and selected["selected_residual_alpha"] == 0.0:
        return selected, None
    selected_recipe = Recipe(
        selected["architecture"],
        selected["width"],
        selected["learning_rate"],
        selected["weight_decay"],
    )
    return selected, selected_recipe
