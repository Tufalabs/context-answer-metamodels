"""Summarize held-out Qwen3.5 rare-safety probe comparisons."""

from __future__ import annotations
import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from cam.metrics.binary import binary_metrics
from experiments.behavior.train_event import EVENT_ORDER

PARTITIONS = (("validation", 532), ("test", 532))
DIRECT_METHODS = (
    "prompt_last_linear",
    "prompt_last_mlp",
    "prompt_mean_linear",
    "prompt_mean_mlp",
    "prompt_attention_mlp",
    "behavior_conditioned_prevalence",
    "global_prevalence",
)
GAUSSIAN_METHODS = ("gaussian_distribution_linear", "gaussian_distribution_mlp")
GAUSSIAN_MEAN_METHODS = ("gaussian_mean_linear", "gaussian_mean_mlp")
FLOW_DISTRIBUTION_METHODS = ("flow_distribution_linear", "flow_distribution_mlp")
FLOW_MEAN_METHODS = ("flow_mean_linear", "flow_mean_mlp")
MLP_MEAN_METHODS = ("mlp_mean_linear", "mlp_mean_mlp")
LINEAR_MEAN_METHODS = ("linear_mean_linear", "linear_mean_mlp")
REALIZED_METHODS = ("realized_answer_linear", "realized_answer_mlp")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_predictions(path: Path, tasks: int) -> tuple[list[str], dict[str, np.ndarray]]:
    markers = [read_json(path / f"predictions_task_{task:03d}.json") for task in range(tasks)]
    if any(marker.get("status") != "complete" for marker in markers):
        raise RuntimeError("flow prediction array is incomplete")
    event_names = markers[0]["event_names"]
    if any(marker["event_names"] != event_names for marker in markers):
        raise RuntimeError("event ordering changed across prediction tasks")
    parts: dict[str, list[np.ndarray]] = {}
    indices = []
    for task in range(tasks):
        with np.load(path / f"predictions_task_{task:03d}.npz") as handle:
            indices.append(handle["combined_index"].copy())
            for key in handle.files:
                if key != "combined_index":
                    parts.setdefault(key, []).append(handle[key].copy())
    order = np.argsort(np.concatenate(indices))
    values = {key: np.concatenate(part, axis=0)[order] for key, part in parts.items()}
    if len(order) != 1_064 or not np.array_equal(np.concatenate(indices)[order], np.arange(1_064)):
        raise RuntimeError("prediction indices do not cover validation+test exactly")
    return event_names, values


def bootstrap_brier(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    seed: int,
    draws: int = 5_000,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    left_error = np.square(left - target).mean(1)
    right_error = np.square(right - target).mean(1)
    difference = left_error - right_error
    indices = generator.integers(0, len(target), size=(draws, len(target)))
    values = difference[indices].mean(1)
    return {
        "brier_difference_left_minus_right": float(difference.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_left_better": float((values < 0).mean()),
    }


def choose(metrics: dict[str, dict[str, Any]], candidates: tuple[str, ...]) -> str:
    available = [method for method in candidates if method in metrics]
    if not available:
        raise RuntimeError(f"none of the selection candidates are available: {candidates}")
    return min(available, key=lambda method: metrics[method]["log_loss"])


def svg_chart(rows: list[dict[str, Any]], path: Path) -> None:
    methods = (
        ("primary_direct", "Best direct context", "#4c78a8"),
        ("primary_linear", "Linear CAM mean", "#9467bd"),
        ("primary_mean", "MLP CAM mean", "#f58518"),
        ("primary_gaussian", "Gaussian distribution", "#b279a2"),
        ("primary_flow", "Best flow distribution", "#54a24b"),
        ("primary_realized", "Actual-answer probe", "#e45756"),
    )
    width, height = 1200, max(520, 110 + 70 * len(rows))
    left, top, plot_width, plot_height = 250, 60, 700, height - 130
    group_height = plot_height / max(1, len(rows))
    bar_height = min(12, group_height / 7)
    values = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Qwen3.5 rare-safety held-out AUPRC</text>',
    ]
    for tick in np.arange(0.0, 1.0001, 0.2):
        x = left + tick * plot_width
        values.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_height}" stroke="#ddd"/>'
        )
        values.append(
            f'<text x="{x}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick:.1f}</text>'
        )
    for row_index, row in enumerate(rows):
        center = top + (row_index + 0.5) * group_height
        values.append(
            f'<text x="{left - 12}" y="{center + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{html.escape(row["event"])}</text>'
        )
        for method_index, (key, _label, color) in enumerate(methods):
            y = center + (method_index - 2.5) * (bar_height + 2) - bar_height / 2
            score = float(row[key]["auprc"])
            values.append(
                f'<rect x="{left}" y="{y}" width="{score * plot_width}" height="{bar_height}" fill="{color}"/>'
            )
    legend_x, legend_y = 990, 80
    for index, (_key, label, color) in enumerate(methods):
        y = legend_y + index * 28
        values.append(f'<rect x="{legend_x}" y="{y - 11}" width="16" height="16" fill="{color}"/>')
        values.append(
            f'<text x="{legend_x + 24}" y="{y + 2}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    values.append(
        f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">AUPRC (fixed 0.2 ticks; higher is better)</text>'
    )
    values.append("</svg>")
    write_text_atomic(path, "\n".join(values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260826)
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
    event_names, predictions = load_predictions(args.prediction_dir, args.num_tasks)
    with np.load(args.label_dir / "labels.npz") as handle:
        labels = {key: handle[key].copy() for key in handle.files}
    profile = read_json(args.label_dir / "event_profile.json")
    offsets = {"validation": (0, 532), "test": (532, 1_064)}
    result_events: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    chart_rows = []
    for event_index, event in enumerate(event_names):
        training = read_json(args.probe_dir / event / "training.json")
        members = set(profile["groups"][event]["behaviors"])
        partition_results = {}
        raw_test: dict[str, np.ndarray] = {}
        selection = {}
        for partition, _size in PARTITIONS:
            start, end = offsets[partition]
            behavior = labels[f"{partition}_behavior_id"]
            selected = np.isin(behavior, list(members))
            target = labels[f"{partition}_match"][selected].astype(np.float64)
            method_metrics = {}
            for method, values in predictions.items():
                if values.ndim != 3 or values.shape[1] <= event_index:
                    continue
                probability = values[start:end, event_index][selected].astype(np.float64)
                method_metrics[method] = binary_metrics(probability, target)
                if partition == "test":
                    raw_test[method] = probability
            priors = training["priors"]
            global_probability = np.full_like(target, float(priors["global"]), dtype=np.float64)
            behavior_probability = np.asarray(
                [priors["by_behavior"].get(value, priors["global"]) for value in behavior],
                dtype=np.float64,
            )[selected, None]
            behavior_probability = np.broadcast_to(behavior_probability, target.shape)
            method_metrics["global_prevalence"] = binary_metrics(global_probability, target)
            method_metrics["behavior_conditioned_prevalence"] = binary_metrics(
                behavior_probability, target
            )
            if partition == "test":
                raw_test["global_prevalence"] = global_probability
                raw_test["behavior_conditioned_prevalence"] = behavior_probability
            partition_results[partition] = {
                "prompts": int(selected.sum()),
                "positive_rollouts": int(target.sum()),
                "prevalence": float(target.mean()),
                "methods": method_metrics,
            }
        validation_metrics = partition_results["validation"]["methods"]
        selection = {
            "primary_direct": choose(validation_metrics, DIRECT_METHODS),
            "primary_gaussian": choose(validation_metrics, GAUSSIAN_METHODS),
            "primary_gaussian_mean": choose(validation_metrics, GAUSSIAN_MEAN_METHODS),
            "primary_flow": choose(validation_metrics, FLOW_DISTRIBUTION_METHODS),
            "primary_flow_mean": choose(validation_metrics, FLOW_MEAN_METHODS),
            "primary_mean": choose(validation_metrics, MLP_MEAN_METHODS),
            "primary_linear": choose(validation_metrics, LINEAR_MEAN_METHODS),
            "primary_realized": choose(validation_metrics, REALIZED_METHODS),
        }
        test_metrics = partition_results["test"]["methods"]
        selected_metrics = {key: test_metrics[method] for key, method in selection.items()}
        test_behavior = labels["test_behavior_id"]
        test_selected = np.isin(test_behavior, list(members))
        test_target = labels["test_match"][test_selected].astype(np.float64)
        comparisons = {
            "flow_distribution_vs_gaussian": bootstrap_brier(
                raw_test[selection["primary_flow"]],
                raw_test[selection["primary_gaussian"]],
                test_target,
                args.seed + event_index * 10 + 5,
            ),
            "gaussian_distribution_vs_predicted_mean": bootstrap_brier(
                raw_test[selection["primary_gaussian"]],
                raw_test[selection["primary_mean"]],
                test_target,
                args.seed + event_index * 10 + 6,
            ),
            "flow_distribution_vs_direct_context": bootstrap_brier(
                raw_test[selection["primary_flow"]],
                raw_test[selection["primary_direct"]],
                test_target,
                args.seed + event_index * 10 + 1,
            ),
            "flow_distribution_vs_flow_mean": bootstrap_brier(
                raw_test[selection["primary_flow"]],
                raw_test[selection["primary_flow_mean"]],
                test_target,
                args.seed + event_index * 10 + 2,
            ),
            "flow_distribution_vs_linear_mean": bootstrap_brier(
                raw_test[selection["primary_flow"]],
                raw_test[selection["primary_linear"]],
                test_target,
                args.seed + event_index * 10 + 4,
            ),
            "flow_distribution_vs_predicted_mean": bootstrap_brier(
                raw_test[selection["primary_flow"]],
                raw_test[selection["primary_mean"]],
                test_target,
                args.seed + event_index * 10 + 3,
            ),
        }
        result_events[event] = {
            "behaviors": sorted(members),
            "selection_by_validation_log_loss": selection,
            "partitions": partition_results,
            "selected_test_metrics": selected_metrics,
            "paired_prompt_bootstrap": comparisons,
        }
        row = {
            "event": event,
            "test_prevalence": partition_results["test"]["prevalence"],
            "test_positive_rollouts": partition_results["test"]["positive_rollouts"],
            **{f"{key}_method": method for key, method in selection.items()},
        }
        for key, metrics in selected_metrics.items():
            for metric in ("brier", "log_loss", "auprc", "auroc", "top_10pct_lift"):
                row[f"{key}_{metric}"] = metrics[metric]
        summary_rows.append(row)
        chart_rows.append({"event": event, **selected_metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "selected_test_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    os.replace(temporary, csv_path)
    result = {
        "schema_version": 1,
        "status": "complete" if set(event_names) == set(EVENT_ORDER) else "incomplete_events",
        "missing_events": sorted(set(EVENT_ORDER) - set(event_names)),
        "model_id": "Qwen/Qwen3.5-9B",
        "layer": 18,
        "label_source": "GPT-6 Luna reasoning=none external rubric judgments",
        "label_caveat": "External-model-judged benchmark; not independent human ground truth.",
        "flow": {
            "checkpoint": args.flow_checkpoint_label,
            "architecture": "354.8M full-dimensional 8-block attention-conditioned rectified flow",
            "samples_per_prompt": 1024,
            "euler_steps": 32,
        },
        "gaussian": {"mean_checkpoint": args.mlp_checkpoint_label, "samples_per_prompt": 1024},
        "mlp": {
            "checkpoint": args.mlp_checkpoint_label,
            "architecture": "learned-query attention answer-mean predictor",
        },
        "behavior_readout_normalization": args.probe_normalization_label,
        "selection_policy": (
            "Choose within each baseline family by validation log loss, then report the untouched test split."
        ),
        "events": result_events,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_atomic(args.output_dir / "results.json", result)
    svg_chart(chart_rows, args.output_dir / "auprc_comparison.svg")
    lines = [
        "# Qwen3.5 rare-safety probe comparison",
        "",
        "Qwen3.5 answers use temperature 1, top-p .95, 4096 tokens, thinking disabled. "
        "Labels are external GPT-6 Luna no-thinking rubric judgments, not human labels.",
        "",
        f"The flow checkpoint is `{args.flow_checkpoint_label}` (354.8M parameters). "
        f"The MLP checkpoint is `{args.mlp_checkpoint_label}`. Behavior readouts retain "
        f"the normalization from `{args.probe_normalization_label}`. Model variants within "
        "each family are selected on validation log loss before the test metrics are read.",
        "",
        "| Event | Base rate | Direct AUPRC | Linear CAM AUPRC | MLP CAM AUPRC | Gaussian AUPRC | Flow AUPRC | Actual-answer AUPRC | Flow-direct Brier Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        comparison = result_events[row["event"]]["paired_prompt_bootstrap"][
            "flow_distribution_vs_direct_context"
        ]
        lines.append(
            f"| {row['event']} | {row['test_prevalence']:.4f} | "
            f"{row['primary_direct_auprc']:.4f} | {row['primary_linear_auprc']:.4f} | {row['primary_mean_auprc']:.4f} | "
            f"{row['primary_gaussian_auprc']:.4f} | {row['primary_flow_auprc']:.4f} | {row['primary_realized_auprc']:.4f} | "
            f"{comparison['brier_difference_left_minus_right']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "A negative Flow-direct Brier difference favors the flow distribution. The realized-answer "
            "probe is a post-generation reference: it observes the actual answer activation and is not a formal upper bound.",
        ]
    )
    write_text_atomic(args.output_dir / "REPORT.md", "\n".join(lines) + "\n")
    print(f"wrote held-out comparisons for {len(result_events)} safety events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
