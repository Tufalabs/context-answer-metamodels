"""Train one rare-safety event's answer and direct-context probes."""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from cam.probes.models import PromptRisk
from cam.metrics.binary import binary_metrics
from cam.probes.training import AnswerMLP
from cam.probes.training import answer_logits
from cam.probes.training import fit_answer
from cam.probes.training import fit_prompt
from cam.probes.training import prefix_state
from cam.probes.training import prompt_logits

HIDDEN = 4096
EVENT_ORDER = (
    "any_target_behavior",
    "any_safety_failure",
    "self_harm_or_eating_disorder",
    "deception_or_false_capability",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_atomic(path: Path, values: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file({key: value.detach().cpu().contiguous() for key, value in values.items()}, temporary)
    os.replace(temporary, path)


def load_partition(
    input_dir: Path, partition: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt, answer, last = [], [], []
    paths = sorted((input_dir / "weirdchat" / partition).glob(f"{partition}_*.safetensors"))
    if not paths:
        raise RuntimeError(f"no compact tensors found for {partition}")
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            prompt.append(handle.get_tensor("prompt_bins"))
            answer.append(handle.get_tensor("y_rollouts"))
            last.append(handle.get_tensor("x_last"))
    return torch.cat(prompt), torch.cat(answer), torch.cat(last)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def reduce_prompt(
    values: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    reducer: str,
) -> torch.Tensor:
    standardized = (values - mean.cpu()) / scale.cpu()
    if reducer == "mean":
        return standardized.mean(1)
    raise ValueError(reducer)


@torch.inference_mode()
def point_logits(
    model: torch.nn.Module,
    values: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            model(chunk.to(device=device, dtype=torch.float32)).reshape(-1).cpu()
            for chunk in values.split(512)
        ]
    )


def fit_point(
    model: torch.nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
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
    target = train_y.float().mean(1)
    val_target = val_y.float().mean(1)
    best_loss, best_state, best_step, stale = math.inf, None, 0, 0
    curve = []
    for step in range(1, 4_001):
        selected = torch.randint(len(train_x), (min(512, len(train_x)),), generator=generator)
        value = train_x.index_select(0, selected).to(device=device, dtype=torch.float32)
        selected_target = target.index_select(0, selected).to(device)
        logits = model(value).reshape(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, selected_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100:
            continue
        validation_logits = point_logits(model, val_x, device)
        validation_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(validation_logits, val_target)
        )
        curve.append(
            {"step": step, "train_log_loss": float(loss), "validation_log_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-5:
            best_loss, best_state, best_step, stale = validation_loss, clone_state(model), step, 0
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("direct context probe produced no checkpoint")
    return best_state, {
        "best_step": best_step,
        "best_validation_log_loss": best_loss,
        "stopping_step": curve[-1]["step"],
        "stop_reason": "validation_plateau" if stale >= 10 else "maximum_steps",
        "curve": curve,
    }


def broadcast_metrics(probability: np.ndarray, target: torch.Tensor) -> dict[str, Any]:
    expanded = np.broadcast_to(probability[:, None], tuple(target.shape))
    return binary_metrics(expanded, target.numpy())


def smoothed_priors(
    train_behavior: np.ndarray,
    train_target: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    global_prior = float((train_target.sum() + 1) / (train_target.numel() + 2))
    values = {}
    for behavior in sorted(set(train_behavior.tolist())):
        selected = train_behavior == behavior
        target = train_target[torch.from_numpy(selected)]
        values[behavior] = float((target.sum() + 1) / (target.numel() + 2))
    return global_prior, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if not 0 <= args.event_index < len(EVENT_ORDER):
        raise ValueError("event index is outside the configured event list")
    event = EVENT_ORDER[args.event_index]
    output = args.output_dir / event
    result_path = output / "training.json"
    if result_path.exists() and read_json(result_path).get("status") in ("complete", "skipped"):
        print(f"event already finished: {result_path}")
        return 0
    verification = read_json(args.label_dir / "verification.json")
    if verification.get("status") != "complete_luna_judged":
        raise RuntimeError("complete external Luna labels are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.monotonic()
    seed_event_index = (0, 1, 2, 4)[args.event_index]
    torch.manual_seed(args.seed + seed_event_index)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    data = {name: load_partition(args.input_dir, name) for name in ("train", "validation", "test")}
    with np.load(args.label_dir / "labels.npz") as handle:
        labels = {key: handle[key].copy() for key in handle.files}
    profile = read_json(args.label_dir / "event_profile.json")
    members = set(profile["groups"][event]["behaviors"])
    selected = {
        partition: np.isin(labels[f"{partition}_behavior_id"], list(members)) for partition in data
    }
    targets = {
        partition: torch.from_numpy(labels[f"{partition}_match"][mask].astype(np.float32))
        for partition, mask in selected.items()
    }
    behaviors = {
        partition: labels[f"{partition}_behavior_id"][mask] for partition, mask in selected.items()
    }
    counts = {
        partition: {
            "prompts": int(mask.sum()),
            "rollouts": int(targets[partition].numel()),
            "positive_rollouts": int(targets[partition].sum()),
            "prevalence": float(targets[partition].mean())
            if targets[partition].numel()
            else math.nan,
        }
        for partition, mask in selected.items()
    }
    sufficient = (
        0 < counts["train"]["positive_rollouts"] < counts["train"]["rollouts"]
        and counts["validation"]["prompts"] > 0
    )
    if not sufficient:
        write_json_atomic(
            result_path,
            {
                "status": "skipped",
                "reason": "insufficient_positive_outcomes",
                "event": event,
                "counts": counts,
            },
        )
        print(f"skipped sparse event {event}: {counts}")
        return 0
    subset = {
        partition: (
            data[partition][0][torch.from_numpy(selected[partition])],
            data[partition][1][torch.from_numpy(selected[partition])],
            data[partition][2][torch.from_numpy(selected[partition])],
        )
        for partition in data
    }
    with safe_open(args.flow_dir / "normalization.safetensors", framework="pt", device="cpu") as h:
        x_mean = h.get_tensor("x_mean").float()
        x_scale = h.get_tensor("x_scale").float()
        y_mean = h.get_tensor("y_mean").to(device=device, dtype=torch.float32)
        y_scale = h.get_tensor("y_scale").to(device=device, dtype=torch.float32)

    states: dict[str, torch.Tensor] = {}
    training: dict[str, Any] = {}
    metrics: dict[str, dict[str, Any]] = {}
    answer_models: dict[str, torch.nn.Module] = {
        "realized_answer_linear": torch.nn.Linear(HIDDEN, 1),
        "realized_answer_mlp": AnswerMLP(),
    }
    for number, (method, model) in enumerate(answer_models.items()):
        state, trace = fit_answer(
            model,
            subset["train"][1],
            targets["train"],
            subset["validation"][1],
            targets["validation"],
            y_mean,
            y_scale,
            device,
            args.seed + 100 * seed_event_index + number,
        )
        model.load_state_dict(state)
        model.to(device).eval()
        states.update(prefix_state(model, method))
        training[method] = trace
        metrics[method] = {
            partition: binary_metrics(
                torch.sigmoid(
                    answer_logits(model, subset[partition][1], y_mean, y_scale, device)
                ).numpy(),
                targets[partition].numpy(),
            )
            for partition in data
        }

    direct_models: dict[str, tuple[torch.nn.Module, str]] = {
        "prompt_last_linear": (torch.nn.Linear(HIDDEN, 1), "last"),
        "prompt_last_mlp": (AnswerMLP(), "last"),
        "prompt_mean_linear": (torch.nn.Linear(HIDDEN, 1), "mean"),
        "prompt_mean_mlp": (AnswerMLP(), "mean"),
    }
    last_mean = subset["train"][2].float().mean(0)
    last_scale = subset["train"][2].float().std(0, correction=0).clamp_min(1e-6)
    states["prompt_last_normalization.mean"] = last_mean
    states["prompt_last_normalization.scale"] = last_scale
    for number, (method, (model, reducer)) in enumerate(direct_models.items(), start=10):
        if reducer == "last":
            reduced = {
                partition: (subset[partition][2].float() - last_mean) / last_scale
                for partition in data
            }
        else:
            reduced = {
                partition: reduce_prompt(subset[partition][0].float(), x_mean, x_scale, reducer)
                for partition in data
            }
        state, trace = fit_point(
            model,
            reduced["train"],
            targets["train"],
            reduced["validation"],
            targets["validation"],
            device,
            args.seed + 100 * seed_event_index + number,
        )
        model.load_state_dict(state)
        model.to(device).eval()
        states.update(prefix_state(model, method))
        training[method] = trace
        metrics[method] = {
            partition: broadcast_metrics(
                torch.sigmoid(point_logits(model, reduced[partition], device)).numpy(),
                targets[partition],
            )
            for partition in data
        }

    prompt_model = PromptRisk(events=1)
    prompt_state, prompt_trace = fit_prompt(
        prompt_model,
        subset["train"][0],
        targets["train"],
        subset["validation"][0],
        targets["validation"],
        x_mean.to(device),
        x_scale.to(device),
        device,
        args.seed + 100 * seed_event_index + 20,
    )
    prompt_model.load_state_dict(prompt_state)
    prompt_model.to(device).eval()
    states.update(prefix_state(prompt_model, "prompt_attention_mlp"))
    training["prompt_attention_mlp"] = prompt_trace
    metrics["prompt_attention_mlp"] = {
        partition: broadcast_metrics(
            torch.sigmoid(
                prompt_logits(
                    prompt_model,
                    subset[partition][0],
                    x_mean.to(device),
                    x_scale.to(device),
                    device,
                )
            ).numpy(),
            targets[partition],
        )
        for partition in data
    }

    global_prior, behavior_prior = smoothed_priors(behaviors["train"], targets["train"])
    metrics["global_prevalence"] = {}
    metrics["behavior_conditioned_prevalence"] = {}
    for partition in data:
        global_probability = np.full(len(targets[partition]), global_prior)
        behavior_probability = np.asarray(
            [behavior_prior.get(value, global_prior) for value in behaviors[partition]],
            dtype=np.float64,
        )
        metrics["global_prevalence"][partition] = broadcast_metrics(
            global_probability, targets[partition]
        )
        metrics["behavior_conditioned_prevalence"][partition] = broadcast_metrics(
            behavior_probability, targets[partition]
        )

    output.mkdir(parents=True, exist_ok=True)
    save_atomic(output / "probes.safetensors", states)
    result = {
        "schema_version": 1,
        "status": "complete",
        "event": event,
        "behaviors": sorted(members),
        "counts": counts,
        "label_source": "GPT-6 Luna reasoning=none external rubric judgment",
        "architectures": {
            "realized_answer_linear": "logistic readout on realized answer-mean activation",
            "realized_answer_mlp": "4096->256->1 SiLU readout on realized answer mean",
            "prompt_last_linear": "logistic readout on exact final-token context activation",
            "prompt_last_mlp": "4096->256->1 SiLU readout on exact final-token context activation",
            "prompt_mean_linear": "logistic readout on mean standardized 32-bin context activation",
            "prompt_mean_mlp": "4096->256->1 SiLU readout on mean standardized 32-bin context activation",
            "prompt_attention_mlp": "learned-query attention over 32 context bins + 512-wide MLP",
        },
        "selection": "validation log-loss plateau; test never used for checkpoint selection",
        "training": training,
        "metrics": metrics,
        "priors": {"global": global_prior, "by_behavior": behavior_prior},
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_atomic(result_path, result)
    print(f"completed Qwen safety probe comparison for {event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
