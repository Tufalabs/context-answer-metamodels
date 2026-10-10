"""Train exact-token MLP and flow metamodels on the established scaling matrix."""

from __future__ import annotations
import bisect
import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors import safe_open
from cam.models import flow as flow_base
from cam.models.token_pooling import HIDDEN
from cam.models.token_pooling import ExactTokenConditionalFlow
from cam.models.token_pooling import PointMetamodel
from cam.models.token_pooling import architecture
from cam.training import scaling as metrics_base

ROLLOUTS = 4
VALIDATION_INTERVAL = int(os.environ.get("CAM_VALIDATION_INTERVAL", "500"))
VALIDATION_PATIENCE = int(os.environ.get("CAM_VALIDATION_PATIENCE", "12"))


VALIDATION_R2_MIN_IMPROVEMENT = float(os.environ.get("CAM_VALIDATION_R2_MIN_IMPROVEMENT", "1e-6"))
VALIDATION_ENERGY_MIN_IMPROVEMENT = float(
    os.environ.get("CAM_VALIDATION_ENERGY_MIN_IMPROVEMENT", "1e-5")
)
MAX_TRAINING_STEPS = int(os.environ.get("CAM_MAX_TRAINING_STEPS", "80000"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass
class TokenPart:
    start: int
    end_exclusive: int
    tokens: torch.Tensor
    offsets: torch.Tensor


class RaggedTokenStore:
    def __init__(self, root: Path, domain: str) -> None:
        self.parts: list[TokenPart] = []
        self.starts: list[int] = []
        paths = sorted((root / domain / "parts").glob("tokens_*.safetensors"))
        for path in paths:
            marker = read_json(path.with_suffix(".json"))
            start, end = int(marker["start"]), int(marker["end_exclusive"])
            handle = safe_open(path, framework="pt", device="cpu")
            tokens = handle.get_tensor("tokens")
            offsets = handle.get_tensor("offsets")
            if tokens.dtype != torch.bfloat16 or tokens.shape[1] != HIDDEN:
                raise RuntimeError(f"invalid token tensor in {path}")
            if len(offsets) != end - start + 1:
                raise RuntimeError(f"invalid offsets in {path}")
            self.parts.append(TokenPart(start, end, tokens, offsets))
            self.starts.append(start)
        if not self.parts:
            raise RuntimeError(f"no staged token parts for {domain}")
        print(
            f"mapped {domain} exact-token parts: {len(self.parts)} parts, "
            f"global coverage {self.parts[0].start:,}:{self.parts[-1].end_exclusive:,}",
            flush=True,
        )

    def _part(self, global_index: int) -> TokenPart:
        position = bisect.bisect_right(self.starts, global_index) - 1
        if position < 0:
            raise IndexError(f"global index {global_index} precedes staged coverage")
        part = self.parts[position]
        if not part.start <= global_index < part.end_exclusive:
            raise IndexError(f"global index {global_index} is absent from staged token parts")
        return part

    def length(self, global_index: int) -> int:
        part = self._part(global_index)
        local = global_index - part.start
        return int(part.offsets[local + 1] - part.offsets[local])

    def sequence(self, global_index: int) -> torch.Tensor:
        part = self._part(global_index)
        local = global_index - part.start
        start, end = int(part.offsets[local]), int(part.offsets[local + 1])
        return part.tokens[start:end]


@dataclass
class DomainData:
    store: RaggedTokenStore
    global_indices: torch.Tensor
    y: torch.Tensor
    lengths: torch.Tensor

    def prefix(self, count: int) -> DomainData:
        return DomainData(
            self.store,
            self.global_indices[:count],
            self.y[:count],
            self.lengths[:count],
        )

    def __len__(self) -> int:
        return len(self.global_indices)


class BucketSampler:
    def __init__(self, data: DomainData, seed: int) -> None:
        self.data = data
        self.order = torch.argsort(data.lengths)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, count: int) -> torch.Tensor:
        requested = count
        count = min(requested, len(self.order))
        anchor = int(torch.randint(len(self.order), (1,), generator=self.generator))
        start = max(0, min(anchor - count // 2, len(self.order) - count))
        window = self.order[start : start + count]
        if requested > len(window):
            draws = torch.randint(len(window), (requested,), generator=self.generator)
            return window.index_select(0, draws)
        permutation = torch.randperm(len(window), generator=self.generator)
        return window.index_select(0, permutation)


def collate(
    items: list[tuple[DomainData, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    references: list[tuple[DomainData, int]] = []
    for data, positions in items:
        references.extend((data, int(position)) for position in positions)
    if not references:
        raise ValueError("cannot collate an empty batch")
    lengths = [int(data.lengths[position]) for data, position in references]
    maximum = max(lengths)
    sequence = torch.zeros((len(references), maximum, HIDDEN), dtype=torch.bfloat16)
    mask = torch.zeros((len(references), maximum), dtype=torch.bool)
    target = torch.empty((len(references), ROLLOUTS, HIDDEN), dtype=torch.bfloat16)
    for row, ((data, position), length) in enumerate(zip(references, lengths, strict=True)):
        value = data.store.sequence(int(data.global_indices[position]))
        if len(value) != length:
            raise RuntimeError("stored sequence length changed after indexing")
        sequence[row, :length].copy_(value)
        mask[row, :length] = True
        target[row].copy_(data.y[position])
    return sequence, mask, target


def mixed_batch(
    lmsys_sampler: BucketSampler,
    weird_sampler: BucketSampler | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weird_sampler is None:
        selected = lmsys_sampler.sample(batch_size)
        return collate([(lmsys_sampler.data, selected)])
    left_count = batch_size // 2
    left = lmsys_sampler.sample(left_count)
    right = weird_sampler.sample(batch_size - left_count)
    return collate([(lmsys_sampler.data, left), (weird_sampler.data, right)])


def evaluation_batches(data: DomainData, batch_size: int) -> Iterator[torch.Tensor]:
    order = torch.argsort(data.lengths)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def target_normalization(
    lmsys: DomainData,
    weird: DomainData | None,
    family: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    def moments(data: DomainData) -> tuple[torch.Tensor, torch.Tensor]:
        total = torch.zeros(HIDDEN, device=device)
        square = torch.zeros(HIDDEN, device=device)
        count = 0
        for chunk in data.y.split(512):
            value = chunk.to(device=device, dtype=torch.float32)
            if family == "mlp":
                value = value.mean(1)
                total += value.sum(0)
                square += value.square().sum(0)
                count += len(value)
            else:
                total += value.sum((0, 1))
                square += value.square().sum((0, 1))
                count += len(value) * ROLLOUTS
        return total / count, square / count

    mean_l, second_l = moments(lmsys)
    if weird is None:
        mean, second = mean_l, second_l
    else:
        mean_w, second_w = moments(weird)
        mean, second = 0.5 * (mean_l + mean_w), 0.5 * (second_l + second_w)
    variance = (second - mean.square()).clamp_min(1e-12)
    scale = variance.mean().sqrt() if family == "mlp" else variance.sqrt()
    return {"y_mean": mean, "y_scale": scale}


def conditioner_output(
    model: PointMetamodel | ExactTokenConditionalFlow,
    sequence: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if isinstance(model, PointMetamodel):
            value = model.conditioner(sequence, mask)
        else:
            value = model.token_conditioner(sequence, mask)
    return value.float()


@torch.inference_mode()
def point_predict(
    model: PointMetamodel,
    data: DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    result = torch.empty((len(data), HIDDEN), dtype=torch.float32)
    model.eval()
    for positions in evaluation_batches(data, batch_size):
        sequence, mask, _ = collate([(data, positions)])
        sequence = sequence.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pooled = conditioner_output(model, sequence, mask)
        prediction = model.predictor(pooled)
        prediction = prediction * values["y_scale"] + values["y_mean"]
        result.index_copy_(0, positions, prediction.cpu())
    return result


def validation_point(
    model: PointMetamodel,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    scores = {
        domain: metrics_base.r2_value(
            data.y.float().mean(1), point_predict(model, data, values, device, batch_size)
        )
        for domain, data in validations.items()
    }
    return float(np.mean(list(scores.values()))), scores


def fit_point(
    conditioner_name: str,
    lmsys: DomainData,
    weird: DomainData | None,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[PointMetamodel, dict[str, Any]]:
    config = architecture(conditioner_name)
    torch.manual_seed(seed)
    model = PointMetamodel(conditioner_name, 8192).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.mlp_weight_decay
    )
    lmsys_sampler = BucketSampler(lmsys, seed + 1)
    weird_sampler = BucketSampler(weird, seed + 2) if weird is not None else None
    best_score, best_state, best_step, stale = -math.inf, None, 0, 0
    curve = []
    plateaued = False
    for step in range(1, MAX_TRAINING_STEPS + 1):
        sequence, mask, target = mixed_batch(
            lmsys_sampler, weird_sampler, config.training_batch_size
        )
        sequence = sequence.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32).mean(1)
        target = (target - values["y_mean"]) / values["y_scale"]
        model.train()
        pooled = conditioner_output(model, sequence, mask)
        prediction = model.predictor(pooled)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % VALIDATION_INTERVAL:
            continue
        score, by_domain = validation_point(
            model, validations, values, device, min(128, config.training_batch_size)
        )
        row = {
            "step": step,
            "train_mse": float(loss.detach()),
            "gradient_norm": float(gradient),
            "validation_macro_r2": score,
            "validation_r2_by_domain": by_domain,
        }
        curve.append(row)
        print(f"point {conditioner_name} {row}", flush=True)
        if score > best_score + VALIDATION_R2_MIN_IMPROVEMENT:
            best_score, best_step = score, step
            best_state = metrics_base.clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= VALIDATION_PATIENCE:
                plateaued = True
                break
    if best_state is None:
        raise RuntimeError("point metamodel produced no checkpoint")
    if not plateaued:
        raise RuntimeError(
            f"point metamodel did not validation-plateau within {MAX_TRAINING_STEPS:,} steps"
        )
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_macro_r2": best_score,
        "stop_reason": "validation_plateau",
        "stopping_step": curve[-1]["step"],
        "validation_interval": VALIDATION_INTERVAL,
        "validation_patience": VALIDATION_PATIENCE,
        "validation_minimum_improvement": VALIDATION_R2_MIN_IMPROVEMENT,
        "maximum_steps_guard": MAX_TRAINING_STEPS,
        "curve": curve,
    }


class EncodedFlowView(torch.nn.Module):
    """Expose an already encoded token condition to the standard flow evaluators."""

    def __init__(self, model: ExactTokenConditionalFlow) -> None:
        super().__init__()
        self.model = model
        self.hidden = model.hidden

    def encode_condition(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def forward(
        self, state: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.model(state, time_value, condition)


@torch.inference_mode()
def encode_dataset(
    model: ExactTokenConditionalFlow,
    data: DomainData,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    result = torch.empty((len(data), model.width), device=device)
    model.eval()
    for positions in evaluation_batches(data, batch_size):
        sequence, mask, _ = collate([(data, positions)])
        sequence = sequence.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pooled = conditioner_output(model, sequence, mask)
        encoded = model.condition(pooled)
        result.index_copy_(0, positions.to(device), encoded)
    return result


def validation_flow(
    model: ExactTokenConditionalFlow,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    scores = {}
    view = EncodedFlowView(model)
    for offset, (domain, data) in enumerate(validations.items()):
        encoded = encode_dataset(model, data, device, batch_size)
        target = data.y.to(device=device, dtype=torch.float32)
        target = (target - values["y_mean"]) / values["y_scale"]
        scores[domain] = flow_base.validation_energy(
            view, encoded, target, 16, 16, 16, seed + 10_000 * offset
        )
    return float(np.mean(list(scores.values()))), scores


def fit_flow(
    conditioner_name: str,
    lmsys: DomainData,
    weird: DomainData | None,
    validations: dict[str, DomainData],
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[ExactTokenConditionalFlow, dict[str, Any]]:
    config = architecture(conditioner_name)
    torch.manual_seed(seed)
    model = ExactTokenConditionalFlow(conditioner_name, 4096, 8).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.flow_weight_decay
    )
    lmsys_sampler = BucketSampler(lmsys, seed + 1)
    weird_sampler = BucketSampler(weird, seed + 2) if weird is not None else None
    cuda_generator = torch.Generator(device=device).manual_seed(seed + 3)
    best_energy, best_state, best_step, stale = math.inf, None, 0, 0
    curve = []
    plateaued = False
    for step in range(1, MAX_TRAINING_STEPS + 1):
        sequence, mask, target_all = mixed_batch(
            lmsys_sampler, weird_sampler, config.training_batch_size
        )
        sequence = sequence.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_all = target_all.to(device=device, dtype=torch.float32)
        target_all = (target_all - values["y_mean"]) / values["y_scale"]
        rollout = torch.randint(
            ROLLOUTS,
            (len(sequence),),
            generator=cuda_generator,
            device=device,
        )
        target = target_all[torch.arange(len(sequence), device=device), rollout]
        noise = torch.randn(target.shape, generator=cuda_generator, device=device)
        time_value = torch.rand((len(target),), generator=cuda_generator, device=device)
        interpolated = (1 - time_value[:, None]) * noise + time_value[:, None] * target
        velocity = target - noise
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pooled = conditioner_output(model, sequence, mask)
        encoded = model.condition(pooled)
        prediction = model(interpolated, time_value, encoded)
        loss = torch.nn.functional.mse_loss(prediction, velocity)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % VALIDATION_INTERVAL:
            continue
        energy, by_domain = validation_flow(
            model,
            validations,
            values,
            device,
            min(128, config.training_batch_size),
            seed + 50_000_000,
        )
        row = {
            "step": step,
            "train_flow_mse": float(loss.detach()),
            "gradient_norm": float(gradient),
            "validation_macro_energy": energy,
            "validation_energy_by_domain": by_domain,
        }
        curve.append(row)
        print(f"flow {conditioner_name} {row}", flush=True)
        if energy < best_energy - VALIDATION_ENERGY_MIN_IMPROVEMENT:
            best_energy, best_step = energy, step
            best_state = metrics_base.clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= VALIDATION_PATIENCE:
                plateaued = True
                break
    if best_state is None:
        raise RuntimeError("flow metamodel produced no checkpoint")
    if not plateaued:
        raise RuntimeError(
            f"flow metamodel did not validation-plateau within {MAX_TRAINING_STEPS:,} steps"
        )
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_macro_energy": best_energy,
        "stop_reason": "validation_plateau",
        "stopping_step": curve[-1]["step"],
        "validation_interval": VALIDATION_INTERVAL,
        "validation_patience": VALIDATION_PATIENCE,
        "validation_minimum_improvement": VALIDATION_ENERGY_MIN_IMPROVEMENT,
        "maximum_steps_guard": MAX_TRAINING_STEPS,
        "curve": curve,
    }
