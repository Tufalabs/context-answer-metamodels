"""Summarize the paper’s LMSYS inference-step ablation."""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = [
        read(path)
        for path in sorted(root.glob("*/*/*/results.json"))
        if read(path).get("status") == "complete"
    ]
    expected = {
        ("lmsys", "lmsys", "validation"),
        ("lmsys", "lmsys", "test"),
    }
    observed = {
        (value["training_regime"], value["evaluation_domain"], value["evaluation_split"])
        for value in results
    }
    if observed != expected:
        raise RuntimeError(f"incomplete ablation results: observed={observed}, expected={expected}")
    flat: list[dict[str, Any]] = []
    for result in results:
        for row in result["rows"]:
            flat.append(
                {
                    "training_regime": result["training_regime"],
                    "evaluation_domain": result["evaluation_domain"],
                    "evaluation_split": result["evaluation_split"],
                    "n_contexts": result["n_contexts"],
                    **row,
                }
            )
    fields = [
        "training_regime",
        "evaluation_domain",
        "evaluation_split",
        "n_contexts",
        "solver",
        "n_steps",
        "vector_field_evaluations",
        "relative_vector_field_cost_vs_current",
        "raw_energy_score_per_sqrt_dimension",
        "raw_energy_delta_vs_euler32",
        "raw_energy_delta_vs_euler32_ci_low",
        "raw_energy_delta_vs_euler32_ci_high",
        "raw_energy_relative_change_pct_vs_euler32",
        "flow_sample_mean_r2",
        "raw_generated_real_variance_trace_ratio",
        "raw_generated_real_pair_distance_ratio",
        "marginal_90pct_coverage",
        "sampling_seconds",
        "sampling_contexts_per_second",
    ]
    csv_path = output / "results.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in flat)
    os.replace(temporary_csv, csv_path)

    lines = [
        "# Full-flow inference-step ablation results",
        "",
        "All rows use fixed trained weights and 64 common-noise samples per prompt. "
        "Energy deltas and paired intervals are relative to the current 32-step Euler protocol. "
        "The test partitions had already been opened; this is a post-hoc numerical diagnostic.",
        "",
    ]
    split_order = {"validation": 0, "test": 1}
    for result in sorted(
        results,
        key=lambda value: (
            split_order[value["evaluation_split"]],
            value["training_regime"],
            value["evaluation_domain"],
        ),
    ):
        setting = (
            f"{result['training_regime']} -> {result['evaluation_domain']} "
            f"{result['evaluation_split']} (n={result['n_contexts']})"
        )
        lines.extend(
            [
                f"## {setting}",
                "",
                "| Solver | Steps | VFEs | Energy | Delta vs E32 [95% CI] | "
                "Mean R2 | Var ratio | Coverage | Sampling s |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(result["rows"], key=lambda value: (value["solver"], value["n_steps"])):
            lines.append(
                f"| {row['solver']} | {row['n_steps']} | {row['vector_field_evaluations']} | "
                f"{row['raw_energy_score_per_sqrt_dimension']:.6f} | "
                f"{row['raw_energy_delta_vs_euler32']:+.6f} "
                f"[{row['raw_energy_delta_vs_euler32_ci_low']:+.6f}, "
                f"{row['raw_energy_delta_vs_euler32_ci_high']:+.6f}] | "
                f"{row['flow_sample_mean_r2']:.4f} | "
                f"{row['raw_generated_real_variance_trace_ratio']:.3f} | "
                f"{row['marginal_90pct_coverage']:.3f} | {row['sampling_seconds']:.1f} |"
            )
        best = min(result["rows"], key=lambda value: value["raw_energy_score_per_sqrt_dimension"])
        lines.extend(
            [
                "",
                f"Lowest observed energy: **{best['solver']} {best['n_steps']} steps** "
                f"({best['vector_field_evaluations']} VFEs, "
                f"{best['raw_energy_score_per_sqrt_dimension']:.6f}).",
                "",
            ]
        )
    atomic_text(output / "RESULTS.md", "\n".join(lines) + "\n")

    plt.rcParams.update({"font.family": "serif", "axes.titleweight": "regular", "font.size": 13})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    colors = {("lmsys", "lmsys"): "#2563eb"}
    setting_labels = {
        ("lmsys", "lmsys"): "LMSYS-only $\\to$ LMSYS",
    }
    for axis, split in zip(axes, ("validation", "test"), strict=True):
        for result in results:
            if (
                result["evaluation_split"] != split
                or result["training_regime"] != "lmsys"
                or result["evaluation_domain"] != "lmsys"
            ):
                continue
            setting_key = (result["training_regime"], result["evaluation_domain"])
            label_root = setting_labels[setting_key]
            for solver, marker, linestyle in (
                ("euler", "o", "-"),
                ("heun", "s", "--"),
            ):
                rows = sorted(
                    (row for row in result["rows"] if row["solver"] == solver),
                    key=lambda value: value["vector_field_evaluations"],
                )
                x = [row["vector_field_evaluations"] for row in rows]
                y = [row["raw_energy_score_per_sqrt_dimension"] for row in rows]
                scales = [
                    row["raw_energy_score"] / row["raw_energy_score_per_sqrt_dimension"]
                    for row in rows
                ]
                low = [
                    row["raw_energy_score_ci_low"] / scale
                    for row, scale in zip(rows, scales, strict=True)
                ]
                high = [
                    row["raw_energy_score_ci_high"] / scale
                    for row, scale in zip(rows, scales, strict=True)
                ]
                axis.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=linestyle,
                    color=colors[setting_key],
                    label=f"{label_root} {solver}",
                )
                axis.fill_between(x, low, high, color=colors[setting_key], alpha=0.10)
        axis.axvline(32, color="#6b7280", linewidth=1, linestyle=":")
        axis.set_xscale("log", base=2)
        axis.set_title(split.capitalize())
        axis.grid(alpha=0.2)
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)
    fig.supxlabel("Vector-field evaluations per sample", fontsize=13, y=0.055)
    fig.supylabel(r"Energy score / $\sqrt{d}$ (lower is better)", fontsize=13, x=0.025)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.055, 0.075, 1, 0.86))
    fig.savefig(output / "inference_step_ablation.svg", bbox_inches="tight")
    fig.savefig(output / "inference_step_ablation.pdf", bbox_inches="tight")
    plt.close(fig)

    atomic_text(
        output / "results.json",
        json.dumps({"schema_version": 1, "results": results, "rows": flat}, indent=2) + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
