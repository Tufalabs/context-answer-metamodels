"""Ablate ODE solver steps for a fixed full-dimensional conditional flow."""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file
from cam.models import flow as flow_base
from cam.models.attention_flow import AttentionConditionalFlow
from cam.training import scaling as run_cell

CURRENT_SOLVER = "euler"
CURRENT_STEPS = 32
DEFAULT_EULER_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_HEUN_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=("lmsys", "weirdchat"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--regime", choices=("lmsys", "combined"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--euler-steps", type=int, nargs="+", default=DEFAULT_EULER_STEPS)
    parser.add_argument("--heun-steps", type=int, nargs="+", default=DEFAULT_HEUN_STEPS)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--context-batch", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--sampling-seed", type=int, default=20_260_901)
    parser.add_argument("--max-contexts", type=int)
    return parser.parse_args()


@torch.inference_mode()
def sample_flow_solver(
    model: AttentionConditionalFlow,
    condition: torch.Tensor,
    n_samples: int,
    n_steps: int,
    seed: int,
    solver: str,
) -> torch.Tensor:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    contexts = len(condition)
    state = torch.randn(
        (contexts, n_samples, model.hidden),
        generator=generator,
        device=condition.device,
        dtype=condition.dtype,
    ).reshape(contexts * n_samples, model.hidden)
    encoded = model.encode_condition(condition)
    encoded = encoded[:, None, :].expand(-1, n_samples, -1).reshape(contexts * n_samples, -1)
    step_size = 1.0 / n_steps
    for step in range(n_steps):
        time_value = torch.full(
            (len(state),),
            step * step_size,
            device=state.device,
            dtype=state.dtype,
        )
        first = model(state, time_value, encoded)
        if solver == "euler":
            state = state + step_size * first
        elif solver == "heun":
            proposed = state + step_size * first
            second = model(proposed, time_value + step_size, encoded)
            state = state + 0.5 * step_size * (first + second)
        else:
            raise ValueError(f"unsupported solver: {solver}")
    return state.reshape(contexts, n_samples, model.hidden)


def paired_mean_ci(difference: np.ndarray, seed: int, draws: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        index = generator.integers(0, len(difference), size=len(difference))
        estimates[draw] = difference[index].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def evaluate_method(
    model: AttentionConditionalFlow,
    data: run_cell.DomainData,
    normalization: dict[str, torch.Tensor],
    device: torch.device,
    *,
    solver: str,
    n_steps: int,
    n_samples: int,
    context_batch: int,
    sampling_seed: int,
    projection_center: torch.Tensor,
    projection_basis: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    x = (data.x.to(device=device, dtype=torch.float32) - normalization["x_mean"]) / normalization[
        "x_scale"
    ]
    y_raw = data.y.to(device=device, dtype=torch.float32)
    y = (y_raw - normalization["y_mean"]) / normalization["y_scale"]
    original_sampler = flow_base.sample_flow
    sampling_seconds = 0.0
    projection_parts: list[np.ndarray] = []

    def sampler(
        selected_model: AttentionConditionalFlow,
        selected_condition: torch.Tensor,
        selected_samples: int,
        selected_steps: int,
        selected_seed: int,
    ) -> torch.Tensor:
        nonlocal sampling_seconds
        if selected_steps != n_steps or selected_samples != n_samples:
            raise RuntimeError("evaluation sampler received an unexpected protocol")
        torch.cuda.synchronize()
        started = time.monotonic()
        value = sample_flow_solver(
            selected_model,
            selected_condition,
            selected_samples,
            selected_steps,
            selected_seed,
            solver,
        )
        raw = (
            value * normalization["y_scale"][None, None, :] + normalization["y_mean"][None, None, :]
        )
        projected = (raw - projection_center[None, None, :]) @ projection_basis
        projection_parts.append(projected.cpu().float().numpy())
        torch.cuda.synchronize()
        sampling_seconds += time.monotonic() - started
        return value

    flow_base.sample_flow = sampler
    try:
        metrics, arrays, _ = flow_base.evaluate_flow(
            model,
            x,
            y,
            y_raw,
            normalization["y_mean"],
            normalization["y_scale"],
            n_samples=n_samples,
            n_steps=n_steps,
            context_batch=context_batch,
            n_bootstrap=500,
            seed=sampling_seed,
        )
    finally:
        flow_base.sample_flow = original_sampler
    clean_metrics = {key.removeprefix("test_"): value for key, value in metrics.items()}
    clean_metrics.update(
        {
            "solver": solver,
            "n_steps": n_steps,
            "vector_field_evaluations": n_steps * (1 if solver == "euler" else 2),
            "relative_vector_field_cost_vs_current": n_steps
            * (1 if solver == "euler" else 2)
            / CURRENT_STEPS,
            "sampling_seconds": sampling_seconds,
            "sampling_contexts_per_second": len(data.x) / sampling_seconds,
        }
    )
    return clean_metrics, arrays, np.concatenate(projection_parts, axis=0)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.bootstrap_draws <= 0 or args.n_samples <= 1 or args.context_batch <= 0:
        raise ValueError("bootstrap draws, samples, and context batch must be positive")
    methods = [
        *(("euler", value) for value in sorted(set(args.euler_steps))),
        *(("heun", value) for value in sorted(set(args.heun_steps))),
    ]
    if (CURRENT_SOLVER, CURRENT_STEPS) not in methods:
        raise ValueError("the method grid must contain the current 32-step Euler baseline")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint = args.checkpoint_dir.resolve()
    parent_result = run_cell.read_json(checkpoint / "results.json")
    architecture = parent_result["architecture"]
    if int(architecture["width"]) != 4096 or int(architecture["blocks"]) != 8:
        raise RuntimeError(f"unexpected paper checkpoint architecture: {architecture}")
    model = AttentionConditionalFlow(32, 4096, 8).to(device)
    model.load_state_dict(load_file(checkpoint / "flow.safetensors", device=str(device)))
    model.eval()
    normalization = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in load_file(checkpoint / "normalization.safetensors", device="cpu").items()
    }
    data = run_cell.load_partition(args.input_dir.resolve(), args.domain, args.split, "flow", None)
    if args.max_contexts is not None:
        if args.max_contexts <= 0:
            raise ValueError("max contexts must be positive")
        data = run_cell.DomainData(x=data.x[: args.max_contexts], y=data.y[: args.max_contexts])
    target_flat = data.y.float().reshape(-1, run_cell.HIDDEN).to(device)
    projection_center = target_flat.mean(0)
    torch.manual_seed(args.sampling_seed)
    _, _, projection_basis = torch.pca_lowrank(
        target_flat - projection_center,
        q=2,
        center=False,
        niter=4,
    )
    real_projection = ((target_flat - projection_center) @ projection_basis).cpu().float().numpy()
    del target_flat
    rows: list[dict[str, Any]] = []
    arrays_by_method: dict[str, dict[str, np.ndarray]] = {}
    projections_by_method: dict[str, np.ndarray] = {}
    for solver, n_steps in methods:
        print(f"evaluating solver={solver} steps={n_steps}", flush=True)
        metrics, arrays, generated_projection = evaluate_method(
            model,
            data,
            normalization,
            device,
            solver=solver,
            n_steps=n_steps,
            n_samples=args.n_samples,
            context_batch=args.context_batch,
            sampling_seed=args.sampling_seed,
            projection_center=projection_center,
            projection_basis=projection_basis,
        )
        method_key = f"{solver}_{n_steps:03d}"
        metrics["method_key"] = method_key
        rows.append(metrics)
        arrays_by_method[method_key] = arrays
        projections_by_method[method_key] = generated_projection
        print(json.dumps(metrics, sort_keys=True), flush=True)
    baseline_key = f"{CURRENT_SOLVER}_{CURRENT_STEPS:03d}"
    baseline_energy = arrays_by_method[baseline_key]["raw_energy_score"] / math.sqrt(
        run_cell.HIDDEN
    )
    baseline_mean = float(baseline_energy.mean())
    saved_arrays: dict[str, np.ndarray] = {}
    for row in rows:
        method_key = row["method_key"]
        arrays = arrays_by_method[method_key]
        energy = arrays["raw_energy_score"] / math.sqrt(run_cell.HIDDEN)
        difference = energy - baseline_energy
        low, high = paired_mean_ci(
            difference,
            args.sampling_seed + 3_000_000,
            args.bootstrap_draws,
        )
        row.update(
            {
                "raw_energy_delta_vs_euler32": float(difference.mean()),
                "raw_energy_delta_vs_euler32_ci_low": low,
                "raw_energy_delta_vs_euler32_ci_high": high,
                "raw_energy_relative_change_pct_vs_euler32": float(
                    100 * difference.mean() / baseline_mean
                ),
            }
        )
        for name in (
            "raw_energy_score",
            "raw_generated_variance_trace",
            "raw_real_variance_trace",
            "raw_generated_pair_distance",
            "raw_real_pair_distance",
        ):
            saved_arrays[f"{method_key}__{name}"] = arrays[name]
    rows.sort(key=lambda row: (row["solver"], row["n_steps"]))
    np.savez_compressed(output / "per_context_metrics.npz", **saved_arrays)
    np.savez_compressed(
        output / "projection_points.npz",
        real=real_projection,
        **projections_by_method,
    )
    result = {
        "schema_version": 1,
        "stage": "post_hoc_fixed_checkpoint_inference_step_ablation",
        "status": "complete",
        "paper_checkpoint": str(checkpoint),
        "parent_result": parent_result,
        "training_regime": args.regime,
        "evaluation_domain": args.domain,
        "evaluation_split": args.split,
        "test_was_already_opened": args.split == "test",
        "n_contexts": len(data.x),
        "n_target_rollouts": data.y.shape[1],
        "n_generated_samples": args.n_samples,
        "sampling_seed": args.sampling_seed,
        "common_initial_noise_across_methods": True,
        "current_protocol": {"solver": CURRENT_SOLVER, "n_steps": CURRENT_STEPS},
        "primary_metric": "raw energy score divided by sqrt(4096)",
        "visualization_projection": (
            "two-component PCA fitted only to real target rollouts in this evaluation partition"
        ),
        "rows": rows,
        "ranking_by_raw_energy": [
            row["method_key"]
            for row in sorted(rows, key=lambda value: value["raw_energy_score_per_sqrt_dimension"])
        ],
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    run_cell.write_json_atomic(output / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
