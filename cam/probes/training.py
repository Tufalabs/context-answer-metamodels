"""Shared answer/context probe fitting with validation-log-loss selection."""

from __future__ import annotations
import math
from typing import Any
import torch
from cam.probes.models import PromptRisk

HIDDEN = 4096
ROLLOUTS = 4


class AnswerMLP(torch.nn.Module):
    def __init__(self, hidden: int = HIDDEN, width: int = 256) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value).squeeze(-1)


def prefix_state(model: torch.nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{name}": value for name, value in model.state_dict().items()}


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.inference_mode()
def answer_logits(
    model: torch.nn.Module,
    values: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for chunk in values.split(512):
        standardized = (chunk.to(device=device, dtype=torch.float32) - mean) / scale
        output = model(standardized.reshape(-1, HIDDEN))
        parts.append(output.reshape(len(chunk), ROLLOUTS).cpu())
    return torch.cat(parts)


@torch.inference_mode()
def prompt_logits(
    model: PromptRisk,
    values: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for chunk in values.split(128):
        standardized = (chunk.to(device=device, dtype=torch.float32) - mean) / scale
        parts.append(model(standardized).squeeze(-1).cpu())
    return torch.cat(parts)


def fit_answer(
    model: torch.nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model.to(device)
    prevalence = train_y.float().mean().clamp(1e-5, 1 - 1e-5)
    final = model if isinstance(model, torch.nn.Linear) else model.layers[-1]
    with torch.no_grad():
        final.bias.fill_(float(torch.logit(prevalence)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_step, stale = math.inf, None, 0, 0
    curve = []
    flat_x = train_x.reshape(-1, HIDDEN)
    flat_y = train_y.reshape(-1)
    for step in range(1, 4_001):
        selected = torch.randint(len(flat_x), (512,), generator=generator)
        value = flat_x.index_select(0, selected).to(device=device, dtype=torch.float32)
        value = (value - mean) / scale
        target = flat_y.index_select(0, selected).to(device=device, dtype=torch.float32)
        logits = model(value).reshape(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100:
            continue
        validation_logits = answer_logits(model, val_x, mean, scale, device)
        validation_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(validation_logits, val_y.float())
        )
        curve.append(
            {"step": step, "train_log_loss": float(loss), "validation_log_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-5:
            best_loss, best_state, best_step, stale = (
                validation_loss,
                clone_state(model),
                step,
                0,
            )
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("answer probe produced no checkpoint")
    return best_state, {
        "best_step": best_step,
        "best_validation_log_loss": best_loss,
        "stopping_step": curve[-1]["step"],
        "stop_reason": "validation_plateau" if stale >= 10 else "maximum_steps",
        "curve": curve,
    }


def fit_prompt(
    model: PromptRisk,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model.to(device)
    prevalence = train_y.float().mean().clamp(1e-5, 1 - 1e-5)
    with torch.no_grad():
        model.predictor[-1].bias.fill_(float(torch.logit(prevalence)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    target = train_y.float().mean(1)
    val_target = val_y.float().mean(1)
    best_loss, best_state, best_step, stale = math.inf, None, 0, 0
    curve = []
    for step in range(1, 6_001):
        selected = torch.randint(len(train_x), (min(256, len(train_x)),), generator=generator)
        value = train_x.index_select(0, selected).to(device=device, dtype=torch.float32)
        value = (value - mean) / scale
        selected_target = target.index_select(0, selected).to(device)
        logits = model(value).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, selected_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100:
            continue
        validation_logits = prompt_logits(model, val_x, mean, scale, device)
        validation_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(validation_logits, val_target)
        )
        curve.append(
            {"step": step, "train_log_loss": float(loss), "validation_log_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-5:
            best_loss, best_state, best_step, stale = (
                validation_loss,
                clone_state(model),
                step,
                0,
            )
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("prompt probe produced no checkpoint")
    return best_state, {
        "best_step": best_step,
        "best_validation_log_loss": best_loss,
        "stopping_step": curve[-1]["step"],
        "stop_reason": "validation_plateau" if stale >= 10 else "maximum_steps",
        "curve": curve,
    }
