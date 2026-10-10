"""Fit common diagnostic coordinates using only unique training prompts."""

import argparse
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from cam.metrics.distribution import fit_pca
from cam.data.common import file_hash
from cam.data.common import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sums = squares = None
    count = 0
    prompts = 0
    sample = []
    paths = sorted((a.data / "lmsys/train").glob("*.safetensors"))
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            y = f.get_tensor("y_rollouts").float()
        flat = y.reshape(-1, y.shape[-1]).double()
        sums = flat.sum(0) if sums is None else sums + flat.sum(0)
        squares = flat.square().sum(0) if squares is None else squares + flat.square().sum(0)
        count += len(flat)
        if prompts < 4096:
            sample.append(y[: 4096 - prompts])
        prompts += len(y)
    mean = (sums / count).float()
    scale = (squares / count - (sums / count).square()).clamp_min(1e-12).sqrt().float()
    y = (torch.cat(sample) - mean) / scale
    dim = y.shape[-1]
    values = {"common_y_mean": mean, "common_y_scale": scale}
    for name, x in [("answer", y), ("residual", y - y.mean(1, keepdim=True))]:
        center, basis, eigen, retained = fit_pca(
            x.reshape(-1, dim), 128, torch.device("cuda"), seed=20260903 + (name == "residual")
        )
        values.update(
            {name + "_center": center, name + "_basis": basis, name + "_eigenvalues": eigen}
        )
    a.output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in values.items()}, a.output / "reference.safetensors")
    write_json(
        a.output / "reference.json",
        {
            "status": "complete",
            "fit_partition": "train",
            "normalization_prompts": prompts,
            "pca_prompts": len(y),
            "rank": 128,
            "target_dimension": dim,
            "data_manifest_sha256": file_hash(a.data / "manifest.json"),
            "policy": "One fixed reference per target; no validation/test examples used",
        },
    )


if __name__ == "__main__":
    main()
