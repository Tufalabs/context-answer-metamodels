"""Evaluate linear, MLP, Gaussian, and flow scaling on three frozen domains."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from cam.metrics import auxiliary
from cam.training import scaling as matrix
from cam.models import gaussian as stochastic
from cam.metrics.uncertainty import ranking_metrics

GAUSSIAN_FAMILIES = ("isotropic", "diagonal", "lowrank_diagonal")
GAUSSIAN_RANKS = (64, 256, 512)


def load_ifeval(path: Path, family: str) -> matrix.DomainData:
    key = "x_last" if family == "linear" else "prompt_bins"
    with safe_open(path, framework="pt", device="cpu") as handle:
        x = handle.get_tensor(key)
        y = handle.get_tensor("y_rollouts")
    if len(x) != 541 or y.shape != (541, 4, matrix.HIDDEN):
        raise RuntimeError(f"unexpected IFEval tensors: x={tuple(x.shape)} y={tuple(y.shape)}")
    return matrix.DomainData(x=x, y=y)


def summarize_distribution(
    arrays: dict[str, np.ndarray],
    prediction: torch.Tensor,
    target: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    result = stochastic.summarize_arrays(arrays, prediction, target, seed=seed, bootstrap_draws=500)
    result.update(matrix.retrieval_metrics(prediction, target, device))
    result["uncertainty"] = ranking_metrics(
        arrays["generated_variance_trace"], arrays["real_variance_trace"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ifeval-compact", type=Path, required=True)
    parser.add_argument("--linear-checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    device = torch.device("cuda")
    output = args.output_dir.resolve()
    result_path = output / "results.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"completed result already exists: {result_path}")
            return 0
    output.mkdir(parents=True, exist_ok=True)
    train_limit = min(args.n_train, 512) if args.smoke else args.n_train
    validation_limit = 64 if args.smoke else None

    train = matrix.load_partition(args.input_dir, "lmsys", "train", "mlp", train_limit)
    validation = matrix.load_partition(
        args.input_dir, "lmsys", "validation", "mlp", validation_limit
    )
    tests = {
        "LMSYS": matrix.load_partition(args.input_dir, "lmsys", "test", "mlp", validation_limit),
        "WeirdChat": matrix.load_partition(
            args.input_dir, "weirdchat", "test", "mlp", validation_limit
        ),
        "IFEval": load_ifeval(args.ifeval_compact.resolve(), "mlp"),
    }
    if args.smoke:
        tests["IFEval"] = matrix.DomainData(tests["IFEval"].x[:64], tests["IFEval"].y[:64])
    linear_tests = {
        "LMSYS": matrix.load_partition(args.input_dir, "lmsys", "test", "linear", validation_limit),
        "WeirdChat": matrix.load_partition(
            args.input_dir, "weirdchat", "test", "linear", validation_limit
        ),
        "IFEval": load_ifeval(args.ifeval_compact.resolve(), "linear"),
    }
    if args.smoke:
        linear_tests["IFEval"] = matrix.DomainData(
            linear_tests["IFEval"].x[:64], linear_tests["IFEval"].y[:64]
        )

    point_results: dict[str, dict[str, Any]] = {family: {} for family in ("linear", "mlp", "flow")}
    linear_state = load_file(args.linear_checkpoint.resolve(), device="cpu")
    for index, (domain, data) in enumerate(linear_tests.items()):
        prediction = matrix.linear_predict(linear_state, data, device)
        point_results["linear"][domain] = matrix.point_metrics(
            prediction, data.y, 20_260_901 + args.n_train + index, device
        )

    mlp, mlp_values = stochastic.load_mlp(args.mlp_checkpoint.resolve(), device)
    _, train_features = stochastic.mlp_outputs(mlp, mlp_values, train.x, device)
    validation_prediction, validation_features = stochastic.mlp_outputs(
        mlp, mlp_values, validation.x, device
    )
    test_predictions: dict[str, torch.Tensor] = {}
    test_features: dict[str, torch.Tensor] = {}
    for index, (domain, data) in enumerate(tests.items()):
        prediction, features = stochastic.mlp_outputs(mlp, mlp_values, data.x, device)
        test_predictions[domain] = prediction
        test_features[domain] = features
        point_results["mlp"][domain] = matrix.point_metrics(
            prediction, data.y, 20_261_901 + args.n_train + index, device
        )

    positions = torch.arange(matrix.ROLLOUTS)
    train_scale_target, global_spread = stochastic.context_log_variance(train.y, positions)
    validation_scale_target, _ = stochastic.context_log_variance(
        validation.y, positions, global_spread
    )
    scale_steps = 200 if args.smoke else 5_000
    scale_head, scale_values, scale_training = stochastic.fit_scale_head(
        train_features,
        train_scale_target,
        validation_features,
        validation_scale_target,
        device,
        seed=20_262_901 + args.n_train,
        max_steps=scale_steps,
    )
    validation_context_scale = stochastic.predict_context_scale(
        scale_head, scale_values, validation_features, device
    )
    available_rank = min(matrix.HIDDEN - 1, train_limit * matrix.ROLLOUTS - 1)
    ranks = tuple(rank for rank in GAUSSIAN_RANKS if rank <= available_rank)
    if args.smoke:
        ranks = (4, 8, 16)
    residual_model = stochastic.fit_residual_model(
        train.y,
        positions,
        device,
        seed=20_263_901 + args.n_train,
        pca_contexts=min(8_192, len(train.y)),
        max_rank=max(ranks),
    )
    projections = stochastic.projection_matrix(20_264_901 + args.n_train, 16, device)
    validation_scales = {
        "global": torch.ones(len(validation.y)),
        "heteroscedastic": validation_context_scale,
    }
    sampler_specs = [
        (
            "isotropic",
            0,
            stochastic.gaussian_sampler(residual_model, "isotropic", 0, device),
        ),
        (
            "diagonal",
            0,
            stochastic.gaussian_sampler(residual_model, "diagonal", 0, device),
        ),
    ]
    sampler_specs.extend(
        (
            "lowrank_diagonal",
            rank,
            stochastic.gaussian_sampler(residual_model, "lowrank_diagonal", rank, device),
        )
        for rank in ranks
    )
    sample_scales = (0.75, 1.0, 1.25) if args.smoke else stochastic.SAMPLE_SCALES
    n_samples = 8 if args.smoke else 64
    candidate_rows = []
    for family_index, (family, rank, sampler) in enumerate(sampler_specs):
        grid = stochastic.evaluate_grid(
            sampler,
            validation_prediction,
            validation.y,
            validation_scales,
            sample_scales,
            projections,
            device,
            n_samples,
            seed=20_265_901 + args.n_train + family_index * 10_000,
        )
        for (mode, sample_scale), (arrays, prediction) in grid.items():
            candidate_rows.append(
                {
                    "family": family,
                    "rank": rank,
                    "scale_mode": mode,
                    "sample_scale": sample_scale,
                    **stochastic.summarize_arrays(
                        arrays,
                        prediction,
                        validation.y,
                        seed=20_266_901 + family_index,
                        bootstrap_draws=50 if args.smoke else 500,
                    ),
                }
            )
    selected_by_family_mode = []
    for family in GAUSSIAN_FAMILIES:
        for mode in ("global", "heteroscedastic"):
            choices = [
                row
                for row in candidate_rows
                if row["family"] == family and row["scale_mode"] == mode
            ]
            if choices:
                selected_by_family_mode.append(
                    min(
                        choices,
                        key=lambda row: row["raw_energy_score_per_sqrt_dimension"],
                    )
                )
    selected = min(
        selected_by_family_mode,
        key=lambda row: row["raw_energy_score_per_sqrt_dimension"],
    )
    selected_family = str(selected["family"])
    selected_rank = int(selected["rank"])
    gaussian_sampler = stochastic.gaussian_sampler(
        residual_model, selected_family, selected_rank, device
    )

    distribution_results: dict[str, dict[str, Any]] = {
        "gaussian": {},
        "flow": {},
    }
    gaussian_scales: dict[str, torch.Tensor] = {}
    for index, (domain, data) in enumerate(tests.items()):
        context_scale = (
            torch.ones(len(data.y))
            if selected["scale_mode"] == "global"
            else stochastic.predict_context_scale(
                scale_head, scale_values, test_features[domain], device
            )
        )
        gaussian_scales[domain] = context_scale
        grid = stochastic.evaluate_grid(
            gaussian_sampler,
            test_predictions[domain],
            data.y,
            {str(selected["scale_mode"]): context_scale},
            (float(selected["sample_scale"]),),
            projections,
            device,
            n_samples,
            seed=20_267_901 + args.n_train + index * 10_000,
        )
        arrays, prediction = next(iter(grid.values()))
        distribution_results["gaussian"][domain] = summarize_distribution(
            arrays,
            prediction,
            data.y,
            20_268_901 + args.n_train + index,
            device,
        )

    flow, flow_values = stochastic.load_flow(
        args.flow_checkpoint_dir / "flow.safetensors",
        args.flow_checkpoint_dir / "normalization.safetensors",
        device,
    )
    for index, (domain, data) in enumerate(tests.items()):
        arrays, prediction = stochastic.evaluate_flow(
            flow,
            flow_values,
            data.x,
            data.y,
            projections,
            device,
            n_samples,
            2 if args.smoke else 32,
            seed=20_269_901 + args.n_train + index * 10_000,
        )
        metrics = summarize_distribution(
            arrays,
            prediction,
            data.y,
            20_270_901 + args.n_train + index,
            device,
        )
        distribution_results["flow"][domain] = metrics
        point_results["flow"][domain] = {
            key: value
            for key, value in metrics.items()
            if key.startswith("sample_mean_") or key.startswith("retrieval_")
        }

    reference = auxiliary.load_reference(args.reference.resolve())
    auxiliary_results = {
        "gaussian": auxiliary.gaussian_metrics(
            gaussian_sampler,
            test_predictions["LMSYS"],
            gaussian_scales["LMSYS"],
            float(selected["sample_scale"]),
            residual_model,
            selected_family,
            selected_rank,
            tests["LMSYS"].y,
            reference,
            seed=20_271_901 + args.n_train,
            n_samples=n_samples,
            likelihood_contexts=2 if args.smoke else 64,
            device=device,
        ),
        "flow": auxiliary.flow_metrics(
            flow,
            flow_values,
            tests["LMSYS"].x,
            tests["LMSYS"].y,
            reference,
            seed=20_272_901 + args.n_train,
            n_samples=n_samples,
            flow_steps=2 if args.smoke else 32,
            likelihood_contexts=2 if args.smoke else 64,
            likelihood_steps=2 if args.smoke else 32,
            hutchinson_probes=1 if args.smoke else 2,
        ),
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "training_domain": "LMSYS only",
        "n_train_contexts": args.n_train,
        "n_train_rollouts_per_context": matrix.ROLLOUTS,
        "domains": {domain: len(data.y) for domain, data in tests.items()},
        "point": point_results,
        "distribution": distribution_results,
        "gaussian_selection": {
            "validation_domain": "LMSYS",
            "selected": selected,
            "selected_by_family_mode": selected_by_family_mode,
            "scale_head_training": scale_training,
            "global_training_residual_variance_per_coordinate": global_spread,
        },
        "auxiliary_lmsys": auxiliary_results,
        "protocol": {
            "flow_and_gaussian_samples": n_samples,
            "flow_euler_steps": 2 if args.smoke else 32,
            "auxiliary_reference": str(args.reference.resolve()),
            "ifeval_scope": "all 541 prompts",
            "posthoc_evaluation": True,
        },
        "smoke": args.smoke,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    matrix.write_json_atomic(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0
