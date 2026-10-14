"""Repeat layer selection using the common unique 10k/1k/3k partitions."""

import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from cam.models import mlp as base
from cam.metrics import regression as scaling
from cam.data.common import write_json
from cam.data.common import file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    values = [load_file(path) for path in sorted(a.data.glob("task_*/layer_*.safetensors"))]
    assert len(values) == 14
    x = torch.cat([v["x_last"] for v in values]).float()
    y = torch.cat([v["y_rollouts"] for v in values]).float().mean(1)
    assert x.shape == y.shape == (14000, 4096)
    train = np.arange(10000)
    val = np.arange(10000, 11000)
    test = np.arange(11000, 14000)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    grid = [base.Recipe("mlp2", 8192, lr, wd) for lr in (3e-4, 1e-4) for wd in (1e-4, 1e-3)]
    selected, recipe = base.run_grid(
        output_dir=a.output,
        input_name="last_prompt_token",
        layer=a.layer,
        method="expected_target_recipe_train_10000",
        grid=grid,
        x=x,
        training_target=y,
        validation_target=y[val],
        train=train,
        validation=val,
        ridge_validation=None,
        seed=42,
        max_epochs=300,
        patience=20,
        device=device,
        normalize_target_rms=True,
    )
    metadata, vpred, tpred = base.train_predictor(
        x,
        y,
        train,
        val,
        test,
        recipe,
        seed=42,
        max_epochs=300,
        patience=20,
        device=device,
        normalize_target_rms=True,
    )
    result = {
        "status": "complete",
        "layer": a.layer,
        "n_train": 10000,
        "n_validation": 1000,
        "n_test": 3000,
        "validation": base.metric(vpred, y[val]),
        "test": base.metric(tpred, y[test]),
        "training": metadata,
        "selected_recipe": selected,
        "test_intervals": scaling.bootstrap_fields(
            tpred, y[test], 500, 42 + a.layer * 10000 + 10000
        ),
        "source_files_sha256": {
            str(path.relative_to(a.data)): file_hash(path)
            for path in sorted(a.data.glob("task_*/layer_*.safetensors"))
        },
    }
    write_json(a.output / "results.json", result)


if __name__ == "__main__":
    main()
