"""Project the paper Gaussian and a fixed flow trajectory into one PCA plane."""

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
from safetensors.torch import save_file
from cam.models.attention_flow import AttentionConditionalFlow
from cam.evaluation import trajectory
from cam.training import scaling as run_cell
from cam.models import gaussian as stochastic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--domain-method-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=500_000)
    parser.add_argument("--projection-contexts", type=int, default=8_192)
    parser.add_argument("--total-steps", type=int, default=trajectory.DEFAULT_TOTAL_STEPS)
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        default=trajectory.DEFAULT_SNAPSHOT_STEPS,
    )
    parser.add_argument(
        "--metric-steps",
        type=int,
        nargs="+",
        default=trajectory.DEFAULT_METRIC_STEPS,
    )
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--context-batch", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=20_260_901)
    parser.add_argument("--projection-seed", type=int, default=20_260_902)
    return parser.parse_args()


def joined_metrics(
    parts: dict[int, dict[str, list[np.ndarray]]],
    steps: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    saved: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for step in steps:
        joined = {name: np.concatenate(values) for name, values in parts[step].items()}
        for name, values in joined.items():
            saved[f"step_{step:04d}__{name}"] = values
        rows.append(
            {
                "trajectory_step": step,
                "flow_time": step / steps[-1],
                "trajectory_percent": 100 * step / steps[-1],
                "raw_energy_score_per_sqrt_dimension": float(
                    joined["energy"].mean() / math.sqrt(run_cell.HIDDEN)
                ),
                "raw_generated_real_variance_trace_ratio": float(
                    joined["generated_variance"].mean() / joined["real_variance"].mean()
                ),
                "marginal_90pct_coverage": float(joined["coverage"].mean()),
            }
        )
    return saved, rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.n_train <= 0 or args.projection_contexts <= 0:
        raise ValueError("training and projection context counts must be positive")
    if args.total_steps <= 0 or args.n_samples <= 1 or args.context_batch <= 0:
        raise ValueError("steps, samples, and context batch must be positive")
    snapshot_steps = tuple(sorted(set(args.snapshot_steps)))
    metric_steps = tuple(sorted(set(args.metric_steps) | set(snapshot_steps)))
    if metric_steps[0] < 0 or metric_steps[-1] != args.total_steps:
        raise ValueError("metric steps must span zero through total steps")

    started = time.monotonic()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    paper_result = run_cell.read_json(args.domain_method_result.resolve())
    if int(paper_result["n_train_contexts"]) != args.n_train:
        raise RuntimeError("the Gaussian selection and requested training scale differ")
    selected = paper_result["gaussian_selection"]["selected"]
    if selected["family"] not in {"isotropic", "diagonal", "lowrank_diagonal"}:
        raise RuntimeError(f"unexpected selected Gaussian family: {selected['family']}")

    input_dir = args.input_dir.resolve()
    train = run_cell.load_partition(input_dir, "lmsys", "train", "mlp", args.n_train)
    validation = run_cell.load_partition(input_dir, "lmsys", "validation", "mlp", None)
    test = run_cell.load_partition(input_dir, "lmsys", "test", "mlp", None)

    mlp, mlp_values = stochastic.load_mlp(args.mlp_checkpoint.resolve(), device)
    _, train_features = stochastic.mlp_outputs(mlp, mlp_values, train.x, device)
    validation_prediction, validation_features = stochastic.mlp_outputs(
        mlp, mlp_values, validation.x, device
    )
    test_prediction, test_features = stochastic.mlp_outputs(mlp, mlp_values, test.x, device)
    positions = torch.arange(run_cell.ROLLOUTS)
    train_scale_target, global_spread = stochastic.context_log_variance(train.y, positions)
    validation_scale_target, _ = stochastic.context_log_variance(
        validation.y, positions, global_spread
    )
    scale_head, scale_values, scale_training = stochastic.fit_scale_head(
        train_features,
        train_scale_target,
        validation_features,
        validation_scale_target,
        device,
        seed=20_262_901 + args.n_train,
        max_steps=5_000,
    )
    torch.set_grad_enabled(False)
    residual_model = stochastic.fit_residual_model(
        train.y,
        positions,
        device,
        seed=20_263_901 + args.n_train,
        pca_contexts=min(8_192, len(train.y)),
        # Match the fit used to select the paper Gaussian, even when validation
        # ultimately chooses a lower rank.
        max_rank=512,
    )

    projection_generator = torch.Generator().manual_seed(args.projection_seed)
    projection_indices = torch.randperm(len(train.y), generator=projection_generator)[
        : min(args.projection_contexts, len(train.y))
    ]
    projection_targets = (
        train.y.index_select(0, projection_indices).float().reshape(-1, run_cell.HIDDEN)
    )
    projection_center = projection_targets.mean(0).to(device)
    torch.manual_seed(args.projection_seed)
    _, _, projection_basis = torch.pca_lowrank(
        projection_targets.to(device) - projection_center,
        q=2,
        center=False,
        niter=4,
    )
    del projection_targets
    real_flat = test.y.float().reshape(-1, run_cell.HIDDEN).to(device)
    real_projection = ((real_flat - projection_center) @ projection_basis).cpu().float().numpy()
    del real_flat

    gaussian_sampler = stochastic.gaussian_sampler(
        residual_model,
        str(selected["family"]),
        int(selected["rank"]),
        device,
    )
    if selected["scale_mode"] == "global":
        context_scale = torch.ones(len(test.y))
    elif selected["scale_mode"] == "heteroscedastic":
        context_scale = stochastic.predict_context_scale(
            scale_head, scale_values, test_features, device
        )
    else:
        raise RuntimeError(f"unexpected Gaussian scale mode: {selected['scale_mode']}")
    gaussian_projection_parts: list[np.ndarray] = []
    gaussian_metric_parts: list[dict[str, np.ndarray]] = []
    gaussian_prediction_parts: list[torch.Tensor] = []
    gaussian_seed = 20_267_901 + args.n_train
    sample_scale = float(selected["sample_scale"])
    for start in range(0, len(test.y), args.context_batch):
        end = min(start + args.context_batch, len(test.y))
        noise = gaussian_sampler(end - start, args.n_samples, gaussian_seed + start)
        samples = (
            test_prediction[start:end].to(device)[:, None]
            + sample_scale * context_scale[start:end].to(device)[:, None, None] * noise
        )
        arrays, prediction = stochastic.sample_arrays(
            samples,
            test.y[start:end].to(device=device, dtype=torch.float32),
            projection_basis.T,
        )
        gaussian_metric_parts.append(arrays)
        gaussian_prediction_parts.append(prediction)
        gaussian_projection_parts.append(
            ((samples - projection_center[None, None]) @ projection_basis).cpu().float().numpy()
        )
    gaussian_arrays = stochastic.combine_parts(gaussian_metric_parts)
    gaussian_metrics = stochastic.summarize_arrays(
        gaussian_arrays,
        torch.cat(gaussian_prediction_parts),
        test.y,
        seed=20_268_901 + args.n_train,
        bootstrap_draws=500,
    )
    expected_energy = paper_result["distribution"]["gaussian"]["LMSYS"][
        "raw_energy_score_per_sqrt_dimension"
    ]
    if not math.isclose(
        gaussian_metrics["raw_energy_score_per_sqrt_dimension"],
        expected_energy,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "reconstructed Gaussian does not match the paper endpoint: "
            f"{gaussian_metrics['raw_energy_score_per_sqrt_dimension']} vs {expected_energy}"
        )

    flow, normalization = stochastic.load_flow(
        args.flow_checkpoint_dir.resolve() / "flow.safetensors",
        args.flow_checkpoint_dir.resolve() / "normalization.safetensors",
        device,
    )
    flow.requires_grad_(False)
    if not isinstance(flow, AttentionConditionalFlow):
        raise RuntimeError("unexpected flow checkpoint architecture")
    x = (test.x.to(device=device, dtype=torch.float32) - normalization["x_mean"]) / (
        normalization["x_scale"]
    )
    y_raw = test.y.to(device=device, dtype=torch.float32)
    y = (y_raw - normalization["y_mean"]) / normalization["y_scale"]
    metric_parts: dict[int, dict[str, list[np.ndarray]]] = {step: {} for step in metric_steps}
    projection_parts: dict[int, list[np.ndarray]] = {step: [] for step in snapshot_steps}
    sampling_seconds = 0.0
    step_size = 1.0 / args.total_steps
    for start in range(0, len(x), args.context_batch):
        end = min(start + args.context_batch, len(x))
        contexts = end - start
        generator = torch.Generator(device=device).manual_seed(args.sampling_seed + start)
        state = torch.randn(
            (contexts, args.n_samples, flow.hidden),
            generator=generator,
            device=device,
            dtype=x.dtype,
        )
        encoded = flow.encode_condition(x[start:end])
        encoded = (
            encoded[:, None].expand(-1, args.n_samples, -1).reshape(contexts * args.n_samples, -1)
        )
        torch.cuda.synchronize()
        batch_started = time.monotonic()
        for step in range(args.total_steps + 1):
            if step in metric_parts:
                arrays, projected = trajectory.record_snapshot(
                    state,
                    y[start:end],
                    y_raw[start:end],
                    normalization["y_mean"],
                    normalization["y_scale"],
                    projection_center,
                    projection_basis,
                )
                for name, values in arrays.items():
                    metric_parts[step].setdefault(name, []).append(values)
                if step in projection_parts:
                    projection_parts[step].append(projected)
            if step == args.total_steps:
                break
            flat_state = state.reshape(contexts * args.n_samples, flow.hidden)
            time_value = torch.full(
                (len(flat_state),),
                step * step_size,
                device=device,
                dtype=flat_state.dtype,
            )
            flat_state = flat_state + step_size * flow(flat_state, time_value, encoded)
            state = flat_state.reshape(contexts, args.n_samples, flow.hidden)
        torch.cuda.synchronize()
        sampling_seconds += time.monotonic() - batch_started
        print(f"contexts={end}/{len(x)}", flush=True)

    saved_metrics, rows = joined_metrics(metric_parts, metric_steps)
    np.savez_compressed(output / "per_context_metrics.npz", **saved_metrics)
    np.savez_compressed(
        output / "projection_points.npz",
        real=real_projection,
        gaussian=np.concatenate(gaussian_projection_parts),
        **{f"step_{step:04d}": np.concatenate(parts) for step, parts in projection_parts.items()},
    )
    save_file(
        {
            "center": projection_center.cpu().contiguous(),
            "basis": projection_basis.cpu().contiguous(),
        },
        output / "projection.safetensors",
    )
    result = {
        "schema_version": 2,
        "stage": "gaussian_and_fixed_fine_grid_flow_projection",
        "status": "complete",
        "training_regime": "lmsys",
        "evaluation_domain": "lmsys",
        "evaluation_split": "test",
        "test_was_already_opened": True,
        "n_contexts": len(test.x),
        "n_target_rollouts": test.y.shape[1],
        "n_generated_samples": args.n_samples,
        "sampling_seed": args.sampling_seed,
        "solver": "euler",
        "total_trajectory_steps": args.total_steps,
        "snapshot_steps": list(snapshot_steps),
        "metric_steps": list(metric_steps),
        "common_initial_noise_across_snapshots": True,
        "visualization_projection": {
            "method": "two-component PCA of raw answer activations",
            "fit_partition": "LMSYS train",
            "fit_contexts": len(projection_indices),
            "fit_rollouts": len(projection_indices) * run_cell.ROLLOUTS,
            "seed": args.projection_seed,
            "test_targets_used_to_fit": False,
        },
        "gaussian_selection": selected,
        "gaussian_scale_head_training": scale_training,
        "gaussian_test_metrics": gaussian_metrics,
        "paper_gaussian_expected_energy": expected_energy,
        "rows": rows,
        "sampling_seconds": sampling_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    run_cell.write_json_atomic(output / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
