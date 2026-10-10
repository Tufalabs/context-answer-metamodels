"""Compare training rollout counts using a fixed sixteen-rollout bank."""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors import safe_open
from cam.models.attention_flow import AttentionConditionalFlow
from cam.training import scaling as base

HIDDEN = 4096
SEQUENCE_BINS = 32
ALL_SEEDS = tuple(range(43, 59))
SEED_COUNTS = (2, 4, 8, 12, 16)
SEED_ORDERS = (
    (43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 51, 52, 53, 54),
    (47, 55, 44, 58, 49, 43, 56, 46, 50, 57, 45, 48, 52, 54, 51, 53),
    (58, 46, 50, 44, 56, 48, 43, 55, 45, 57, 47, 49, 54, 51, 53, 52),
)
SIZE_GROUPS = (
    (100, 250, 500),
    (1_000, 2_500, 5_000),
    (10_000, 20_000, 28_600),
    (50_000, 75_000, 100_000),
)
PROMPT_SIZES = tuple(size for group in SIZE_GROUPS for size in group)


@dataclass
class LoadedData:
    x: torch.Tensor
    y: torch.Tensor


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_rows(path: Path) -> int:
    start, end = (int(value) for value in path.stem.rsplit("_", 2)[-2:])
    return end - start + 1


def file_start(path: Path) -> int:
    return int(path.stem.rsplit("_", 2)[-2])


def load_partition(
    input_dir: Path,
    target_dir: Path,
    partition: str,
    family: str,
    limit: int | None,
) -> LoadedData:
    input_paths = sorted((input_dir / "lmsys" / partition).glob(f"{partition}_*.safetensors"))
    target_paths = sorted((target_dir / "lmsys" / partition).glob(f"{partition}_*.safetensors"))
    available = sum(file_rows(path) for path in input_paths)
    target_available = sum(file_rows(path) for path in target_paths)
    jointly_available = min(available, target_available)
    count = jointly_available if limit is None else limit
    if count <= 0 or count > jointly_available:
        raise RuntimeError(
            f"{partition}: requested {limit}, jointly available "
            f"{jointly_available} ({available} inputs/{target_available} targets)"
        )
    x_shape = (count, HIDDEN) if family == "linear" else (count, SEQUENCE_BINS, HIDDEN)
    x_dtype = torch.float32 if family == "linear" else torch.bfloat16
    x = torch.empty(x_shape, dtype=x_dtype)
    y = torch.empty((count, len(ALL_SEEDS), HIDDEN), dtype=torch.bfloat16)
    key = "x_last" if family == "linear" else "prompt_bins"
    loaded = 0
    for input_path, target_path in zip(input_paths, target_paths, strict=False):
        if file_start(input_path) != file_start(target_path):
            raise RuntimeError(f"unaligned shard starts: {input_path} and {target_path}")
        rows = min(file_rows(input_path), file_rows(target_path), count - loaded)
        if rows <= 0:
            break
        with safe_open(input_path, framework="pt", device="cpu") as handle:
            x[loaded : loaded + rows].copy_(handle.get_tensor(key)[:rows])
        with safe_open(target_path, framework="pt", device="cpu") as handle:
            y[loaded : loaded + rows].copy_(handle.get_tensor("y_rollouts")[:rows])
        loaded += rows
    if loaded != count:
        raise RuntimeError(f"{partition}: loaded {loaded}, expected {count}")
    print(
        f"loaded {partition}: {count:,} rows with {len(ALL_SEEDS)} rollout targets",
        flush=True,
    )
    return LoadedData(x=x, y=y)


def seed_positions(seeds: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor([ALL_SEEDS.index(seed) for seed in seeds], dtype=torch.long)


def selected_data(
    data: LoadedData, positions: torch.Tensor, count: int | None = None
) -> base.DomainData:
    x = data.x if count is None else data.x[:count]
    y = data.y if count is None else data.y[:count]
    return base.DomainData(x=x, y=y.index_select(1, positions))


def fixed_data(data: LoadedData, count: int | None = None) -> base.DomainData:
    x = data.x if count is None else data.x[:count]
    y = data.y if count is None else data.y[:count]
    return base.DomainData(x=x, y=y)


def point_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    target = target.float()
    target_mean = target.mean(1)
    center = target_mean - target_mean.mean(0)
    sse = (prediction - target_mean).double().square().sum(1).numpy()
    tss = center.double().square().sum(1).numpy()
    r2 = 1.0 - sse.sum() / tss.sum()
    distances = torch.linalg.vector_norm(prediction[:, None].float() - target, dim=2).mean(
        1
    ).double().numpy() / math.sqrt(HIDDEN)
    pairs = torch.stack(
        [
            torch.linalg.vector_norm(target[:, left] - target[:, right], dim=1)
            for left, right in combinations(range(target.shape[1]), 2)
        ],
        dim=1,
    )
    oracle = 0.5 * pairs.mean(1).double().numpy() / math.sqrt(HIDDEN)
    generator = np.random.default_rng(seed)
    r2_draws = np.empty(500)
    energy_draws = np.empty(500)
    for draw in range(500):
        indices = generator.integers(0, len(sse), size=len(sse))
        r2_draws[draw] = 1.0 - sse[indices].sum() / tss[indices].sum()
        energy_draws[draw] = distances[indices].mean()
    result: dict[str, float | int] = {
        "sample_mean_r2": float(r2),
        "sample_mean_r2_ci_low": float(np.quantile(r2_draws, 0.025)),
        "sample_mean_r2_ci_high": float(np.quantile(r2_draws, 0.975)),
        "sample_mean_cosine": float(
            torch.nn.functional.cosine_similarity(prediction.float(), target_mean, dim=1).mean()
        ),
        "raw_energy_score_per_sqrt_dimension": float(distances.mean()),
        "raw_energy_score_per_sqrt_dimension_ci_low": float(np.quantile(energy_draws, 0.025)),
        "raw_energy_score_per_sqrt_dimension_ci_high": float(np.quantile(energy_draws, 0.975)),
        "raw_oracle_energy_score_per_sqrt_dimension": float(oracle.mean()),
    }
    result.update(base.retrieval_metrics(prediction, target, device))
    return result


def fit_flow(
    train: base.DomainData,
    validation: base.DomainData,
    values: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[AttentionConditionalFlow, dict[str, Any]]:
    torch.manual_seed(seed)
    model = AttentionConditionalFlow(SEQUENCE_BINS, 4096, 8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    cpu_generator = torch.Generator().manual_seed(seed + 1)
    cuda_generator = torch.Generator(device=device).manual_seed(seed + 2)
    best_energy, best_state, best_step, stale = math.inf, None, 0, 0
    curve = []
    plateaued = False
    stopping_step = 0
    for step in range(1, base.MAX_TRAINING_STEPS + 1):
        condition, targets = base.cpu_batch(train, 256, cpu_generator, "flow", values, device)
        rollout = torch.randint(
            targets.shape[1], (len(condition),), generator=cuda_generator, device=device
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
        if step % base.VALIDATION_INTERVAL:
            continue
        energy, by_domain = base.flow_validation(
            model, {"lmsys": validation}, values, device, seed + 50_000_000
        )
        curve.append(
            {
                "step": step,
                "train_flow_mse": float(loss.detach()),
                "gradient_norm": float(gradient),
                "validation_energy": energy,
                "validation_energy_by_domain": by_domain,
            }
        )
        print(f"flow step={step} loss={float(loss):.6f} val_energy={energy:.6f}", flush=True)
        if energy < best_energy - 1e-5:
            best_energy = energy
            best_step = step
            best_state = base.clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= base.VALIDATION_PATIENCE:
                plateaued = True
                stopping_step = step
                break
    if best_state is None:
        raise RuntimeError("flow did not produce a checkpoint")
    if not plateaued:
        raise RuntimeError(
            f"flow validation did not plateau within {base.MAX_TRAINING_STEPS:,} steps"
        )
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_energy": best_energy,
        "curve": curve,
        "stop_reason": "validation_plateau",
        "stopping_step": stopping_step,
        "validation_interval": base.VALIDATION_INTERVAL,
        "validation_patience": base.VALIDATION_PATIENCE,
        "maximum_steps_guard": base.MAX_TRAINING_STEPS,
    }


def run_size(
    family: str,
    replicate: int,
    seeds: tuple[int, ...],
    n_train: int,
    loaded_train: LoadedData,
    validation: LoadedData,
    test: LoadedData,
    output_dir: Path,
    seed: int,
) -> None:
    cell = (
        output_dir
        / family
        / f"replicate_{replicate}"
        / f"seeds_{len(seeds):02d}"
        / f"train_{n_train:06d}"
    )
    result_path = cell / "results.json"
    if result_path.exists() and read_json(result_path).get("status") == "complete":
        print(f"skipping completed {family}/rep={replicate}/seeds={len(seeds)}/{n_train}")
        return
    started = time.monotonic()
    device = torch.device("cuda")
    positions = seed_positions(seeds)
    train_selected = selected_data(loaded_train, positions, n_train)
    validation_fixed = fixed_data(validation)
    train_fixed = fixed_data(loaded_train, min(1_000, n_train))
    test_fixed = fixed_data(test)
    values = base.normalization(train_selected, None, family, device) if family != "linear" else {}
    cell.mkdir(parents=True, exist_ok=True)
    checkpoint_saved = True
    if family == "linear":
        state, training = base.fit_linear(train_selected, None, {"lmsys": validation_fixed}, device)
        if checkpoint_saved:
            base.save_tensors_atomic(cell / "model.safetensors", state)
        evaluations = []
        for index, (split, data) in enumerate((("train", train_fixed), ("test", test_fixed))):
            prediction = base.linear_predict(state, data, device)
            evaluations.append(
                {
                    "split": split,
                    **point_metrics(prediction, data.y, seed + index, device),
                }
            )
        architecture = "closed-form adaptive ridge on exact last prompt token"
    elif family == "mlp":
        model, training = base.fit_mlp(
            train_selected, None, {"lmsys": validation_fixed}, values, device, seed
        )
        if checkpoint_saved:
            base.save_tensors_atomic(
                cell / "model.safetensors",
                {
                    **model.state_dict(),
                    **{f"normalization.{key}": value for key, value in values.items()},
                },
            )
        evaluations = []
        for index, (split, data) in enumerate((("train", train_fixed), ("test", test_fixed))):
            prediction = base.mlp_predict(model, data, values, device)
            evaluations.append(
                {
                    "split": split,
                    **point_metrics(prediction, data.y, seed + index, device),
                }
            )
        architecture = "32-bin learned-query attention + width-8192 two-hidden-layer MLP"
    else:
        model, training = fit_flow(train_selected, validation_fixed, values, device, seed)
        if checkpoint_saved:
            base.save_tensors_atomic(cell / "flow.safetensors", model.state_dict())
            base.save_tensors_atomic(cell / "normalization.safetensors", values)
        evaluations = []
        for index, (split, data) in enumerate((("train", train_fixed), ("test", test_fixed))):
            evaluations.append(
                {
                    "split": split,
                    **base.flow_metrics(model, data, values, device, seed + 100_000 * (index + 1)),
                }
            )
        architecture = "32-bin attention-conditioned rectified flow, width4096, 8 blocks"
    result = {
        "schema_version": 1,
        "status": "complete",
        "family": family,
        "replicate": replicate,
        "n_train_contexts": n_train,
        "n_validation_contexts": len(validation_fixed.x),
        "n_test_contexts": len(test_fixed.x),
        "n_train_rollout_seeds": len(seeds),
        "n_training_context_rollout_pairs": n_train * len(seeds),
        "training_rollout_seeds": list(seeds),
        "evaluation_rollout_seeds": list(ALL_SEEDS),
        "evaluation_target_policy": "fixed 16-seed bank for every cell",
        "layer": 18,
        "architecture": architecture,
        "training": training,
        "evaluations": evaluations,
        "checkpoint_saved": checkpoint_saved,
        "elapsed_seconds": time.monotonic() - started,
        "fit_seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    write_json_atomic(result_path, result)
    print(
        f"completed {family} rep={replicate} seeds={len(seeds)} n={n_train:,} "
        f"elapsed={result['elapsed_seconds']:.1f}s",
        flush=True,
    )
    del values
    torch.cuda.empty_cache()


def task_specification(
    family: str, task_index: int
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if family not in {"linear", "mlp", "flow"}:
        raise ValueError(f"unknown family: {family}")
    tasks = [
        (replicate, order[:seed_count], sizes)
        for replicate, order in enumerate(SEED_ORDERS)
        for seed_count in SEED_COUNTS
        for size in PROMPT_SIZES
        for sizes in ((size,),)
    ]
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task index must be in [0,{len(tasks) - 1}]")
    return tasks[task_index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("linear", "mlp", "flow"), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--fit-seed", type=int, default=20260825)
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    replicate, seeds, sizes = task_specification(args.family, args.task_index)
    print(
        f"family={args.family} task={args.task_index} replicate={replicate} "
        f"seeds={seeds} sizes={sizes}",
        flush=True,
    )
    maximum = max(sizes)
    loaded_train = load_partition(args.input_dir, args.target_dir, "train", args.family, maximum)
    validation = load_partition(args.input_dir, args.target_dir, "validation", args.family, None)
    test = load_partition(args.input_dir, args.target_dir, "test", args.family, None)
    if len(validation.x) != 1000 or len(test.x) != 3000:
        raise RuntimeError(
            "unique-prompt rollout sweep requires 1000 validation / 3000 test prompts"
        )
    if args.load_only:
        print(
            f"load-only validation passed: train={len(loaded_train.x):,} "
            f"validation={len(validation.x):,} test={len(test.x):,}",
            flush=True,
        )
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    for n_train in sizes:
        cell_seed = (
            args.fit_seed
            + (0 if args.family == "flow" else 10_000_000 if args.family == "mlp" else 20_000_000)
            + replicate * 1_000_000
            + len(seeds) * 10_000
            + n_train
        )
        run_size(
            args.family,
            replicate,
            seeds,
            n_train,
            loaded_train,
            validation,
            test,
            args.output_dir.resolve(),
            cell_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
