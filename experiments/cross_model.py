"""Matched cross-model fits on shared unique-text splits and reused rollouts."""

import argparse
import json
import os
import time
from pathlib import Path
import torch
from safetensors import safe_open
from cam.training import cross_model as base
from cam.training import cross_model_baselines as baseline
from cam.data.common import CROSS_SIZES
from cam.data.common import file_hash

TASKS = [
    (target, c, f, r, n)
    for target, conditions in [("qwen35", ("qwen25", "qwen35")), ("gemma", ("qwen25", "gemma"))]
    for c in conditions
    for f in ("linear", "mlp", "flow", "gaussian")
    for r in (range(1) if target == "gemma" and f == "linear" else range(3))
    for n in CROSS_SIZES
]


def load(condition, targets, part, family, limit=None):
    key = "x_last" if family == "linear" else "prompt_bins"
    xs = []
    ys = []
    loaded = 0
    for path in sorted((condition / "lmsys" / part).glob("*.safetensors")):
        if limit is not None and loaded >= limit:
            break
        target = targets / "lmsys" / part / path.name
        with (
            safe_open(path, framework="pt", device="cpu") as f,
            safe_open(target, framework="pt", device="cpu") as g,
        ):
            count = f.get_slice(key).get_shape()[0]
            if limit is not None:
                count = min(count, limit - loaded)
            assert torch.equal(
                f.get_tensor("source_global_indices")[:count],
                g.get_tensor("source_global_indices")[:count],
            )
            xs.append(f.get_slice(key)[:count].clone())
            ys.append(g.get_slice("y_rollouts")[:count].clone())
        loaded += count
    if limit is not None:
        assert loaded == limit
    return base.DomainData(torch.cat(xs), torch.cat(ys))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--condition", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--load-only", action="store_true")
    a = p.parse_args()
    target, conditioner, family, replicate, n = TASKS[a.task]
    dim = 4096 if target == "qwen35" else 3840
    base.TARGET_HIDDEN = dim
    base.TRAIN_ROLLOUTS = 8 if target == "qwen35" else 4
    base.seed_scaling.HIDDEN = dim
    baseline.stochastic.HIDDEN = dim
    base.auxiliary.HIDDEN = dim
    base.auxiliary.geometry.HIDDEN = dim
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "results.json").exists():
        result = json.loads((a.output / "results.json").read_text())
        assert result["status"] == "complete"
        return
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    datasets = {
        part: load(a.condition, a.targets, part, family, n if part == "train" else None)
        for part in ("train", "validation", "test")
    }
    if target == "qwen35":
        order = base.seed_scaling.SEED_ORDERS[replicate]
        train_seeds = list(order[:8])
        test_seeds = list(order[8:])
        for part in datasets:
            seeds = test_seeds if part == "test" else train_seeds
            datasets[part] = base.selected_rollouts(
                datasets[part], base.seed_scaling.seed_positions(seeds)
            )
    else:
        train_seeds = [43, 44, 45, 46]
        test_seeds = [47, 48, 49, 50]
    train, val, test = [datasets[p] for p in ("train", "validation", "test")]
    assert train.y.shape[1:] == (base.TRAIN_ROLLOUTS, dim)
    assert len(val.x) == 1000 and len(test.x) == 3000
    if a.load_only:
        print(
            json.dumps(
                {
                    "status": "load_verified",
                    "target": target,
                    "conditioner": conditioner,
                    "shapes": {k: [list(v.x.shape), list(v.y.shape)] for k, v in datasets.items()},
                }
            ),
            flush=True,
        )
        return
    seed = (
        (
            20260831 + (0 if family == "flow" else 20000000 if family == "linear" else 10000000)
            if target == "qwen35"
            else 20261001
        )
        + replicate * 1000000
        + n
    )
    started = time.monotonic()
    permutation = torch.randperm(
        len(test.x), generator=torch.Generator().manual_seed(seed + 90000000)
    )
    shuffled = base.DomainData(test.x[permutation], test.y)
    reference = base.auxiliary.load_reference(a.reference) if a.reference else None
    result = {
        "target": target,
        "conditioner": conditioner,
        "family": family,
        "replicate": replicate,
        "n_train": n,
        "n_validation": len(val.x),
        "n_test": len(test.x),
        "target_hidden": dim,
        "fit_seed": seed,
        "training_rollout_seeds": train_seeds,
        "test_rollout_seeds": test_seeds,
        "replicate_type": "rollout partition" if target == "qwen35" else "initialization",
        "condition_manifest_sha256": file_hash(a.condition / "manifest.json"),
        "target_manifest_sha256": file_hash(a.targets / "manifest.json"),
        "source_code_sha256": {m.__name__: file_hash(Path(m.__file__)) for m in (base, baseline)},
        "train_wrapper_sha256": file_hash(Path(__file__)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if family == "linear":
        base.matrix.HIDDEN = train.x.shape[-1]
        state, training = base.matrix.fit_linear(train, None, {"lmsys": val}, device)
        prediction = base.matrix.linear_predict(state, test, device)
        shuffled_prediction = base.matrix.linear_predict(state, shuffled, device)
        metrics = base.seed_scaling.point_metrics(prediction, test.y, seed + 100000, device)
        shuffled_metrics = base.seed_scaling.point_metrics(
            shuffled_prediction, test.y, seed + 200000, device
        )
        base.matrix.save_tensors_atomic(a.output / "model.safetensors", state)
    elif family in ("mlp", "flow"):
        values = base.normalization(train, family, device)
        if family == "mlp":
            model, training = base.fit_mlp(train, val, values, device, seed, 80000)
            prediction = base.mlp_predict(model, test.x, values, device)
            shuffled_prediction = base.mlp_predict(model, shuffled.x, values, device)
            metrics = base.seed_scaling.point_metrics(prediction, test.y, seed + 100000, device)
            shuffled_metrics = base.seed_scaling.point_metrics(
                shuffled_prediction, test.y, seed + 200000, device
            )
        else:
            model, training = base.fit_flow(train, val, values, device, seed, 80000)
            metrics, arrays, prediction = base.evaluate_flow(
                model, test, values, device, seed + 100000
            )
            shuffled_metrics, _, _ = base.evaluate_flow(
                model, shuffled, values, device, seed + 200000
            )
            base.save_npz_atomic(a.output / "distribution_metrics.npz", arrays)
            if reference is not None:
                metrics["auxiliary"] = base.auxiliary.flow_metrics(
                    model, values, test.x, test.y, reference, seed=seed + 300000
                )
        base.matrix.save_tensors_atomic(a.output / "model.safetensors", model.state_dict())
        base.matrix.save_tensors_atomic(a.output / "normalization.safetensors", values)
    else:
        baseline.GAUSSIAN_RANKS = tuple(
            r for r in (64, 256, 512) if r <= min(dim - 1, n * (base.TRAIN_ROLLOUTS - 1))
        )
        training, architecture, metrics, shuffled_metrics, arrays, parameters = (
            baseline.fit_gaussian(train, val, test, shuffled, device, seed, 80000, reference)
        )
        result.update(architecture=architecture, parameters=parameters)
        base.save_npz_atomic(
            a.output / "distribution_metrics.npz", baseline.with_raw_prefix(arrays)
        )
        prediction = None
    if prediction is not None:
        base.matrix.save_tensors_atomic(
            a.output / "test_predictions.safetensors", {"prediction": prediction.float()}
        )
    result.update(
        status="complete",
        training=training,
        test_metrics=metrics,
        shuffled_condition_test_metrics=shuffled_metrics,
        elapsed_seconds=time.monotonic() - started,
    )
    base.write_json_atomic(a.output / "results.json", result)
    print(
        json.dumps(
            {
                "status": "complete",
                "target": target,
                "conditioner": conditioner,
                "family": family,
                "n": n,
                "replicate": replicate,
                "elapsed_seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
