"""Compare direct probes with the best full Qwen3.5 layer-18 flow."""

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
from cam.models import flow as flow_base
from cam.models.attention_flow import AttentionConditionalFlow
from experiments.behavior.train_event import EVENT_ORDER
from cam.models import gaussian as stochastic
from cam.probes.models import PromptRisk
from cam.probes.training import AnswerMLP
from cam.probes.attention import AttentionProbe

HIDDEN = 4096
SEQUENCE_BINS = 32
ROLLOUTS = 4
PARTITIONS = (("validation", 532), ("test", 532))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_npz_atomic(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def tensor_file(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def prefixed(values: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    needle = f"{prefix}."
    return {
        key.removeprefix(needle): value for key, value in values.items() if key.startswith(needle)
    }


def load_partition(
    input_dir: Path, partition: str, expected: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompts, answers, lasts = [], [], []
    for path in sorted((input_dir / "weirdchat" / partition).glob(f"{partition}_*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            prompts.append(handle.get_tensor("prompt_bins"))
            answers.append(handle.get_tensor("y_rollouts"))
            lasts.append(handle.get_tensor("x_last"))
    prompt = torch.cat(prompts)
    answer = torch.cat(answers)
    if prompt.shape != (expected, SEQUENCE_BINS, HIDDEN) or answer.shape != (
        expected,
        ROLLOUTS,
        HIDDEN,
    ):
        raise RuntimeError(f"unexpected {partition} compact shapes")
    last = torch.cat(lasts)
    if last.shape != (expected, HIDDEN):
        raise RuntimeError(f"unexpected {partition} final-context shape")
    return prompt, answer, last


def expand_rollouts(value: torch.Tensor) -> torch.Tensor:
    return value[:, :, None].expand(-1, -1, ROLLOUTS)


@torch.inference_mode()
def estimate(
    prompt_raw: torch.Tensor,
    answer_raw: torch.Tensor,
    prompt_last_raw: torch.Tensor,
    flow: AttentionConditionalFlow,
    mean_mlp: AttentionProbe,
    models: list[dict[str, Any]],
    flow_norm: dict[str, torch.Tensor],
    mlp_norm: dict[str, torch.Tensor],
    probe_norm: dict[str, torch.Tensor],
    n_samples: int,
    n_steps: int,
    seed: int,
    device: torch.device,
    linear_state: dict[str, torch.Tensor],
    gaussian: dict[str, Any],
) -> dict[str, torch.Tensor]:
    raw = prompt_raw.to(device=device, dtype=torch.float32)
    flow_condition = (raw - flow_norm["x_mean"]) / flow_norm["x_scale"]
    samples = flow_base.sample_flow(flow, flow_condition, n_samples, n_steps, seed)
    samples_in_probe_space = (
        samples * (flow_norm["y_scale"] / probe_norm["y_scale"])
        + (flow_norm["y_mean"] - probe_norm["y_mean"]) / probe_norm["y_scale"]
    )
    sample_mean = samples_in_probe_space.mean(1)
    mlp_condition = (raw - mlp_norm["x_mean"]) / mlp_norm["x_scale"]
    mlp_standardized, mlp_weights = mean_mlp(mlp_condition)
    mlp_raw = mlp_standardized * mlp_norm["y_scale"] + mlp_norm["y_mean"]
    mlp_in_probe_space = (mlp_raw - probe_norm["y_mean"]) / probe_norm["y_scale"]
    # Gaussian residuals are in raw activation units, and the frozen MLP is its mean.
    selected_gaussian = gaussian["selected"]
    context_scale = torch.ones(len(raw), device=device)
    if selected_gaussian["scale_mode"] == "heteroscedastic":
        pooled = torch.einsum("bt,btd->bd", mlp_weights, mlp_condition).to(torch.bfloat16)
        context_scale = stochastic.predict_context_scale(
            gaussian["scale_head"], gaussian["scale_values"], pooled, device
        ).to(device)
    residual_samples = gaussian["sampler"](len(raw), n_samples, seed + 700_000_000)
    gaussian_raw = mlp_raw[:, None] + residual_samples * (
        context_scale[:, None, None] * float(selected_gaussian["sample_scale"])
    )
    gaussian_samples = (gaussian_raw - probe_norm["y_mean"]) / probe_norm["y_scale"]
    linear_raw = (
        (prompt_last_raw.to(device=device, dtype=torch.float32) - linear_state["x_mean"])
        / linear_state["x_scale"]
    ) @ linear_state["weight"] + linear_state["y_mean"]
    linear_in_probe_space = (linear_raw - probe_norm["y_mean"]) / probe_norm["y_scale"]
    realized = (
        answer_raw.to(device=device, dtype=torch.float32) - probe_norm["y_mean"]
    ) / probe_norm["y_scale"]
    probe_condition = (raw - probe_norm["x_mean"]) / probe_norm["x_scale"]
    context_mean = probe_condition.mean(1)
    prompt_methods: dict[str, list[torch.Tensor]] = {
        "gaussian_distribution_linear": [],
        "gaussian_distribution_mlp": [],
        "gaussian_mean_linear": [],
        "gaussian_mean_mlp": [],
        "flow_distribution_linear": [],
        "flow_distribution_mlp": [],
        "flow_mean_linear": [],
        "flow_mean_mlp": [],
        "mlp_mean_linear": [],
        "mlp_mean_mlp": [],
        "linear_mean_linear": [],
        "linear_mean_mlp": [],
        "prompt_last_linear": [],
        "prompt_last_mlp": [],
        "prompt_mean_linear": [],
        "prompt_mean_mlp": [],
        "prompt_attention_mlp": [],
    }
    realized_methods: dict[str, list[torch.Tensor]] = {
        "realized_answer_linear": [],
        "realized_answer_mlp": [],
    }
    dispersion: dict[str, list[torch.Tensor]] = {
        "gaussian_probe_std_linear": [],
        "gaussian_probe_std_mlp": [],
        "gaussian_mc_standard_error_linear": [],
        "gaussian_mc_standard_error_mlp": [],
        "flow_probe_std_linear": [],
        "flow_probe_std_mlp": [],
        "flow_mc_standard_error_linear": [],
        "flow_mc_standard_error_mlp": [],
    }
    for bundle in models:
        for head in ("linear", "mlp"):
            readout = bundle[f"realized_answer_{head}"]
            logits = readout(gaussian_samples)
            mean_logits = readout(gaussian_samples.mean(1))
            if head == "linear":
                logits = logits.squeeze(-1)
                mean_logits = mean_logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
            prompt_methods[f"gaussian_distribution_{head}"].append(probabilities.mean(1))
            prompt_methods[f"gaussian_mean_{head}"].append(torch.sigmoid(mean_logits))
            std = probabilities.std(1, correction=1)
            dispersion[f"gaussian_probe_std_{head}"].append(std)
            dispersion[f"gaussian_mc_standard_error_{head}"].append(std / n_samples**0.5)
        context_last = (
            prompt_last_raw.to(device=device, dtype=torch.float32) - bundle["prompt_last_mean"]
        ) / bundle["prompt_last_scale"]
        linear_sample = torch.sigmoid(
            bundle["realized_answer_linear"](samples_in_probe_space).squeeze(-1)
        )
        nonlinear_sample = torch.sigmoid(bundle["realized_answer_mlp"](samples_in_probe_space))
        prompt_methods["flow_distribution_linear"].append(linear_sample.mean(1))
        prompt_methods["flow_distribution_mlp"].append(nonlinear_sample.mean(1))
        prompt_methods["flow_mean_linear"].append(
            torch.sigmoid(bundle["realized_answer_linear"](sample_mean).squeeze(-1))
        )
        prompt_methods["flow_mean_mlp"].append(
            torch.sigmoid(bundle["realized_answer_mlp"](sample_mean))
        )
        prompt_methods["mlp_mean_linear"].append(
            torch.sigmoid(bundle["realized_answer_linear"](mlp_in_probe_space).squeeze(-1))
        )
        prompt_methods["mlp_mean_mlp"].append(
            torch.sigmoid(bundle["realized_answer_mlp"](mlp_in_probe_space))
        )
        prompt_methods["linear_mean_linear"].append(
            torch.sigmoid(bundle["realized_answer_linear"](linear_in_probe_space).squeeze(-1))
        )
        prompt_methods["linear_mean_mlp"].append(
            torch.sigmoid(bundle["realized_answer_mlp"](linear_in_probe_space))
        )
        prompt_methods["prompt_last_linear"].append(
            torch.sigmoid(bundle["prompt_last_linear"](context_last).squeeze(-1))
        )
        prompt_methods["prompt_last_mlp"].append(
            torch.sigmoid(bundle["prompt_last_mlp"](context_last))
        )
        prompt_methods["prompt_mean_linear"].append(
            torch.sigmoid(bundle["prompt_mean_linear"](context_mean).squeeze(-1))
        )
        prompt_methods["prompt_mean_mlp"].append(
            torch.sigmoid(bundle["prompt_mean_mlp"](context_mean))
        )
        prompt_methods["prompt_attention_mlp"].append(
            torch.sigmoid(bundle["prompt_attention_mlp"](probe_condition).squeeze(-1))
        )
        realized_methods["realized_answer_linear"].append(
            torch.sigmoid(bundle["realized_answer_linear"](realized).squeeze(-1))
        )
        realized_methods["realized_answer_mlp"].append(
            torch.sigmoid(bundle["realized_answer_mlp"](realized))
        )
        linear_std = linear_sample.std(1, correction=1)
        nonlinear_std = nonlinear_sample.std(1, correction=1)
        dispersion["flow_probe_std_linear"].append(linear_std)
        dispersion["flow_probe_std_mlp"].append(nonlinear_std)
        dispersion["flow_mc_standard_error_linear"].append(linear_std / n_samples**0.5)
        dispersion["flow_mc_standard_error_mlp"].append(nonlinear_std / n_samples**0.5)
    result = {
        key: expand_rollouts(torch.stack(value, 1)).cpu() for key, value in prompt_methods.items()
    }
    result.update({key: torch.stack(value, 1).cpu() for key, value in realized_methods.items()})
    result.update({key: torch.stack(value, 1).cpu() for key, value in dispersion.items()})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--mlp-dir", type=Path, required=True)
    parser.add_argument("--linear-dir", type=Path, required=True)
    parser.add_argument("--gaussian-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--probe-normalization-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, default=28)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--context-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--linear-checkpoint-label")
    parser.add_argument(
        "--flow-checkpoint-label",
        default="metamodels/flow/combined/train_500000",
    )
    parser.add_argument(
        "--mlp-checkpoint-label",
        default="metamodels/mlp/combined/train_500000",
    )
    parser.add_argument(
        "--probe-normalization-label",
        default="metamodels/flow/combined/train_500000",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    completed_events = [
        event
        for event in EVENT_ORDER
        if (args.probe_dir / event / "training.json").exists()
        and read_json(args.probe_dir / event / "training.json").get("status") == "complete"
    ]
    if not completed_events:
        raise RuntimeError("no complete safety event probes are available")
    if not 0 <= args.task_index < args.num_tasks:
        raise ValueError("task index is outside the configured task count")
    marker_path = args.output_dir / f"predictions_task_{args.task_index:03d}.json"
    output_path = args.output_dir / f"predictions_task_{args.task_index:03d}.npz"
    if marker_path.exists() and read_json(marker_path).get("status") == "complete":
        print(f"sampling task already complete: {marker_path}")
        return 0
    started = time.monotonic()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    partitions = [load_partition(args.input_dir, name, size) for name, size in PARTITIONS]
    combined_prompt = torch.cat([value[0] for value in partitions])
    combined_answer = torch.cat([value[1] for value in partitions])
    combined_last = torch.cat([value[2] for value in partitions])
    all_indices = np.arange(len(combined_prompt), dtype=np.int64)
    selected = np.array_split(all_indices, args.num_tasks)[args.task_index]

    flow = AttentionConditionalFlow(SEQUENCE_BINS, HIDDEN, 8).to(device)
    flow.load_state_dict(tensor_file(args.flow_dir / "flow.safetensors"))
    flow.eval()
    flow_norm = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in tensor_file(args.flow_dir / "normalization.safetensors").items()
    }
    probe_normalization_dir = args.probe_normalization_dir or args.flow_dir
    probe_norm = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in tensor_file(probe_normalization_dir / "normalization.safetensors").items()
    }
    mlp_values = tensor_file(args.mlp_dir / "model.safetensors")
    mean_mlp = AttentionProbe(HIDDEN, 8192).to(device)
    mean_mlp.load_state_dict(
        {key: value for key, value in mlp_values.items() if not key.startswith("normalization.")}
    )
    mean_mlp.eval()
    mlp_norm = {
        key.removeprefix("normalization."): value.to(device=device, dtype=torch.float32)
        for key, value in mlp_values.items()
        if key.startswith("normalization.")
    }
    linear_state = {
        k: v.to(device=device, dtype=torch.float32)
        for k, v in tensor_file(args.linear_dir / "model.safetensors").items()
    }
    gaussian_values = tensor_file(args.gaussian_dir / "residual_model.safetensors")
    gaussian_selected = read_json(args.gaussian_dir / "results.json")["selected"]
    scale_head = stochastic.ScaleHead().to(device)
    scale_head.load_state_dict(prefixed(gaussian_values, "scale_head"))
    scale_head.eval()
    gaussian = {
        "selected": gaussian_selected,
        "scale_head": scale_head,
        "scale_values": prefixed(gaussian_values, "scale_values"),
        "sampler": stochastic.gaussian_sampler(
            gaussian_values, gaussian_selected["family"], gaussian_selected["rank"], device
        ),
    }
    models: list[dict[str, Any]] = []
    for event in completed_events:
        values = tensor_file(args.probe_dir / event / "probes.safetensors")
        bundle: dict[str, torch.nn.Module] = {
            "realized_answer_linear": torch.nn.Linear(HIDDEN, 1),
            "realized_answer_mlp": AnswerMLP(),
            "prompt_last_linear": torch.nn.Linear(HIDDEN, 1),
            "prompt_last_mlp": AnswerMLP(),
            "prompt_mean_linear": torch.nn.Linear(HIDDEN, 1),
            "prompt_mean_mlp": AnswerMLP(),
            "prompt_attention_mlp": PromptRisk(events=1),
        }
        for name, model in bundle.items():
            model.load_state_dict(prefixed(values, name))
            model.to(device).eval()
        bundle["prompt_last_mean"] = values["prompt_last_normalization.mean"].to(
            device=device, dtype=torch.float32
        )
        bundle["prompt_last_scale"] = values["prompt_last_normalization.scale"].to(
            device=device, dtype=torch.float32
        )
        models.append(bundle)

    outputs: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(selected), args.context_batch):
        indices = selected[start : start + args.context_batch]
        torch_indices = torch.from_numpy(indices)
        values = estimate(
            combined_prompt.index_select(0, torch_indices),
            combined_answer.index_select(0, torch_indices),
            combined_last.index_select(0, torch_indices),
            flow,
            mean_mlp,
            models,
            flow_norm,
            mlp_norm,
            probe_norm,
            args.n_samples,
            args.n_steps,
            args.seed + int(indices[0]) * 10_000,
            device,
            linear_state,
            gaussian,
        )
        for key, value in values.items():
            outputs.setdefault(key, []).append(value)
        print(f"task={args.task_index} contexts={start + len(indices)}/{len(selected)}", flush=True)
    arrays = {key: torch.cat(value).numpy() for key, value in outputs.items()}
    save_npz_atomic(output_path, combined_index=selected, **arrays)
    write_json_atomic(
        marker_path,
        {
            "schema_version": 1,
            "status": "complete",
            "task_index": args.task_index,
            "num_tasks": args.num_tasks,
            "contexts": len(selected),
            "n_flow_samples": args.n_samples,
            "n_gaussian_samples": args.n_samples,
            "gaussian_selected": gaussian_selected,
            "flow_euler_steps": args.n_steps,
            "flow_checkpoint": args.flow_checkpoint_label,
            "mlp_checkpoint": args.mlp_checkpoint_label,
            "linear_checkpoint": args.linear_checkpoint_label or str(args.linear_dir),
            "probe_normalization_checkpoint": args.probe_normalization_label,
            "event_names": completed_events,
            "methods": sorted(arrays),
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
