"""Add the paper Gaussian baseline behind the new 500k mixed MLP.

All training statistics and validation selection use the same corrected mixed
activation dataset as the new MLP/flow, with equal weight for the two domains.
No behavior labels or test activations are loaded here.
"""

import argparse, json, math, os, time, hashlib
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from cam.models import gaussian as base
from cam.models import flow
from cam.training.scaling import clone_state


def load_features(root, domain, partition, model, norm, device):
    features = []
    targets = []
    means = []
    count = 0
    expected = {
        "lmsys": {"train": 500000, "validation": 1000},
        "weirdchat": {"train": 1596, "validation": 532},
    }[domain][partition]
    for path in sorted((root / domain / partition).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            x = f.get_tensor("prompt_bins")
            y = f.get_tensor("y_rollouts")
        prediction, pooled = base.mlp_outputs(model, norm, x, device)
        features.append(pooled)
        targets.append(y)
        count += len(y)
        if partition == "validation":
            means.append(prediction)
        print(f"pooled {domain}/{partition} {count}/{expected}", flush=True)
    assert count == expected, (domain, partition, count, expected)
    return {
        "features": torch.cat(features),
        "y": torch.cat(targets),
        "mean": torch.cat(means) if means else None,
    }


def spread(y):
    parts = []
    for chunk in y.split(512):
        z = chunk.float()
        parts.append(z.var(1, correction=1).mean(-1))
    return torch.cat(parts)


def fit_scale(train, val, device, seed):
    # Equal domain moments and minibatches, matching the mixed metamodel objective.
    stats = [base.feature_moments(d["features"]) for d in train]
    mean = torch.stack([a for a, b in stats]).mean(0)
    second = torch.stack([a.square() + b.square() for a, b in stats]).mean(0)
    scale = (second - mean.square()).clamp_min(1e-8).sqrt()
    global_spread = sum(float(spread(d["y"]).mean()) for d in train) / len(train)
    for d in train + val:
        d["target"] = (spread(d["y"]) / global_spread).clamp(1e-4, 1e4).log().clamp(-6, 6)
    torch.manual_seed(seed)
    model = base.ScaleHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    mean = mean.to(device)
    scale = scale.to(device)
    best = math.inf
    best_state = None
    stale = 0
    curve = []
    best_step = 0
    for step in range(1, 5001):
        losses = []
        for d in train:
            index = torch.randint(len(d["features"]), (512,), generator=generator)
            x = d["features"].index_select(0, index).to(device, dtype=torch.float32)
            target = d["target"].index_select(0, index).to(device)
            losses.append(torch.nn.functional.smooth_l1_loss(model((x - mean) / scale), target))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100:
            continue
        model.eval()
        losses = []
        with torch.inference_mode():
            for d in val:
                p = torch.cat(
                    [
                        model((c.to(device, dtype=torch.float32) - mean) / scale).cpu()
                        for c in d["features"].split(1024)
                    ]
                )
                losses.append(float(torch.nn.functional.smooth_l1_loss(p, d["target"])))
        metric = sum(losses) / len(losses)
        curve.append({"step": step, "macro_validation_smooth_l1": metric})
        print("scale", curve[-1], flush=True)
        if metric < best - 1e-5:
            best = metric
            best_step = step
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
        model.train()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    rms = []
    with torch.inference_mode():
        for d in train:
            logs = torch.cat(
                [
                    model((c.to(device, dtype=torch.float32) - mean) / scale).cpu()
                    for c in d["features"].split(1024)
                ]
            )
            rms.append(logs.clamp(-6, 6).exp().mean())
    values = {
        "feature_mean": mean.cpu(),
        "feature_scale": scale.cpu(),
        "rms_normalizer": torch.stack(rms).mean().sqrt().clamp_min(1e-6),
    }
    return (
        model,
        values,
        {
            "best_step": best_step,
            "curve": curve,
            "global_spread": global_spread,
            "domain_weight": 0.5,
        },
    )


def fit_covariance(train, device, seed):
    diagonals = []
    banks = []
    generator = torch.Generator().manual_seed(seed)
    for d in train:
        y = d["y"]
        total = torch.zeros(y.shape[-1], dtype=torch.float64)
        for c in y.split(256):
            total += c.double().var(1, correction=1).sum(0)
        diagonals.append((total / len(y)).float())
        chosen = torch.randperm(len(y), generator=generator)[: min(4096, len(y))]
        z = y.index_select(0, chosen).float()
        z = (z - z.mean(1, keepdim=True)) * math.sqrt(4 / 3)
        banks.append(z.reshape(-1, z.shape[-1]))
    # Weight each domain equally in the randomized PCA covariance estimate.
    count = sum(len(b) for b in banks)
    bank = torch.cat([b * math.sqrt(count / (len(banks) * len(b))) for b in banks]).to(device)
    torch.manual_seed(seed)
    _, singular, basis = torch.pca_lowrank(bank, q=512, center=False, niter=2)
    return {
        "diagonal_variance": torch.stack(diagonals).mean(0),
        "basis": basis.cpu(),
        "eigenvalues": (singular.square() / count).cpu(),
    }


@torch.inference_mode()
def select(residual, scale_head, scale_values, val, device):
    scales = [
        base.predict_context_scale(scale_head, scale_values, d["features"], device) for d in val
    ]
    rows = []
    for family_index, (family, rank) in enumerate(
        [("isotropic", 0), ("diagonal", 0)] + [("lowrank_diagonal", r) for r in (64, 256, 512)]
    ):
        sampler = base.gaussian_sampler(residual, family, rank, device)
        for mode in ("global", "heteroscedastic"):
            for multiplier in base.SAMPLE_SCALES:
                domain_scores = []
                for index, d in enumerate(val):
                    total = 0.0
                    for start in range(0, len(d["y"]), 16):
                        target = d["y"][start : start + 16].to(device, dtype=torch.float32)
                        center = d["mean"][start : start + 16].to(device, dtype=torch.float32)
                        context_scale = (
                            scales[index][start : start + len(target)].to(device)
                            if mode == "heteroscedastic"
                            else torch.ones(len(target), device=device)
                        )
                        noise = sampler(
                            len(target),
                            64,
                            20_765_901 + family_index * 10000 + index * 100000 + start,
                        )
                        samples = center[:, None] + noise * (
                            context_scale[:, None, None] * multiplier
                        )
                        total += float(
                            flow.energy_components(samples, target)["energy_score"].sum()
                        )
                    domain_scores.append(total / len(d["y"]) / math.sqrt(4096))
                row = {
                    "family": family,
                    "rank": rank,
                    "scale_mode": mode,
                    "sample_scale": multiplier,
                    "validation_domain_energy": dict(zip(("lmsys", "weirdchat"), domain_scores)),
                    "macro_validation_energy": sum(domain_scores) / len(domain_scores),
                }
                rows.append(row)
                print("candidate", row, flush=True)
    return min(rows, key=lambda r: r["macro_validation_energy"]), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--mlp-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mean-checkpoint-label")
    a = p.parse_args()
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    started = time.monotonic()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if (a.output_dir / "results.json").exists():
        assert json.loads((a.output_dir / "results.json").read_text())["status"] == "complete"
        return
    model, norm = base.load_mlp(a.mlp_checkpoint, device)
    train = [
        load_features(a.input_dir, d, "train", model, norm, device) for d in ("lmsys", "weirdchat")
    ]
    val = [
        load_features(a.input_dir, d, "validation", model, norm, device)
        for d in ("lmsys", "weirdchat")
    ]
    del model
    scale, values, training = fit_scale(train, val, device, 20_762_901)
    residual = fit_covariance(train, device, 20_763_901)
    selected, rows = select(residual, scale, values, val, device)
    state = {
        **residual,
        **{f"scale_head.{k}": v for k, v in scale.state_dict().items()},
        **{f"scale_values.{k}": v for k, v in values.items()},
    }
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in state.items()},
        a.output_dir / "residual_model.safetensors",
    )
    base.write_json_atomic(
        a.output_dir / "results.json",
        {
            "status": "complete",
            "selected": selected,
            "candidates": rows,
            "scale_training": training,
            "training_domain": "500000 LMSYS + 1596 WeirdChat; 50/50 domain weight",
            "training_data": "new 4096-cap WeirdChat targets plus current 500000 LMSYS; no test or Luna labels used",
            "mean_checkpoint": a.mean_checkpoint_label or str(a.mlp_checkpoint),
            "mean_checkpoint_sha256": hashlib.sha256(a.mlp_checkpoint.read_bytes()).hexdigest(),
            "test_partition_opened": False,
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


if __name__ == "__main__":
    main()
