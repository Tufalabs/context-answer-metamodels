"""Render intermediate snapshots from fixed fine-grid flow trajectories."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.trajectory_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = [read(root / "results.json")]
    if results[0]["status"] != "complete":
        raise RuntimeError("The projection run is incomplete")
    expected = {("lmsys", "lmsys")}
    observed = {(value["training_regime"], value["evaluation_domain"]) for value in results}
    if observed != expected:
        raise RuntimeError(f"incomplete trajectory results: {observed=}, {expected=}")

    lines = [
        "# Gaussian and fixed-trajectory flow projections",
        "",
        "Flow panels snapshot the same base particles during one fixed 1,000-update "
        "Euler trajectory. The LMSYS diagnostic also projects the validation-selected "
        "Gaussian forecast into the identical PCA basis and axes.",
        "",
    ]
    plt.rcParams.update({"font.family": "serif", "axes.titleweight": "regular", "font.size": 11})
    for result in results:
        regime = result["training_regime"]
        domain = result["evaluation_domain"]
        total_steps = result["total_trajectory_steps"]
        projection_path = root / "projection_points.npz"
        with np.load(projection_path) as projection:
            real = projection["real"]
            gaussian = projection["gaussian"] if "gaussian" in projection.files else None
            selected = {step: projection[f"step_{step:04d}"] for step in result["snapshot_steps"]}

        random = np.random.default_rng(20260901)
        real_index = random.choice(len(real), size=min(12_000, len(real)), replace=False)
        real_plot = real[real_index]
        figure = plt.figure(figsize=(10.8, 8.8))
        grid = figure.add_gridspec(
            3,
            3,
            height_ratios=(1, 1, 1.05),
            hspace=0.42,
            wspace=0.44,
        )
        generated_panels: list[tuple[str, np.ndarray, str]] = []
        for step, generated in selected.items():
            progress = 100 * step / total_steps
            descriptor = "base noise" if step == 0 else f"update {step}"
            generated_panels.append((f"Flow: {progress:g}% ({descriptor})", generated, "#D946C5"))
        if gaussian is not None:
            generated_panels.append(("Gaussian forecast", gaussian, "#2A9D8F"))
        panel_labels = tuple(f"({chr(ord('a') + index)})" for index in range(8))
        overlay_axes = []
        for index, (title, generated, color) in enumerate(generated_panels):
            axis = figure.add_subplot(grid[index // 3, index % 3])
            overlay_axes.append(axis)
            generated_flat = generated.reshape(-1, 2)
            generated_index = random.choice(
                len(generated_flat),
                size=min(12_000, len(generated_flat)),
                replace=False,
            )
            axis.scatter(
                generated_flat[generated_index, 0],
                generated_flat[generated_index, 1],
                s=2,
                alpha=0.22,
                color=color,
                edgecolors="none",
                rasterized=True,
            )
            axis.scatter(
                real_plot[:, 0],
                real_plot[:, 1],
                s=3,
                alpha=0.45,
                color="#F2C94C",
                edgecolors="none",
                rasterized=True,
            )
            axis.set_title(
                f"{panel_labels[index]} {title}",
                fontsize=13,
                pad=8,
            )
            axis.grid(False)
        x_limits = (
            min(axis.get_xlim()[0] for axis in overlay_axes),
            max(axis.get_xlim()[1] for axis in overlay_axes),
        )
        y_limits = (
            min(axis.get_ylim()[0] for axis in overlay_axes),
            max(axis.get_ylim()[1] for axis in overlay_axes),
        )
        for axis in overlay_axes:
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
        bottom_row_by_column = {
            column: max(index // 3 for index in range(len(overlay_axes)) if index % 3 == column)
            for column in {index % 3 for index in range(len(overlay_axes))}
        }
        for index, axis in enumerate(overlay_axes):
            row, column = divmod(index, 3)
            if column == 0:
                axis.set_ylabel("PC 2")
            else:
                axis.tick_params(axis="y", which="both", left=False, labelleft=False)
            if row == bottom_row_by_column[column]:
                axis.set_xlabel("PC 1")
            else:
                axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        curve_axis = figure.add_subplot(grid[2, 1:] if gaussian is not None else grid[2, :])
        rows = sorted(result["rows"], key=lambda value: value["trajectory_step"])
        curve_axis.plot(
            [row["trajectory_percent"] for row in rows],
            [row["raw_energy_score_per_sqrt_dimension"] for row in rows],
            color="#D946C5",
            marker="o",
            markersize=4,
            linewidth=2,
        )
        curve_axis.set_xlim(0, 100)
        curve_axis.set_xlabel("Progress through one fixed 1,000-update trajectory (%)", fontsize=12)
        curve_axis.set_ylabel(r"Energy score / $\sqrt{d}$", fontsize=12)
        curve_axis.set_title(
            f"{panel_labels[len(generated_panels)]} Intermediate-state energy score",
            fontsize=13,
            pad=9,
        )
        curve_axis.grid(alpha=0.25)
        regime_label = "LMSYS-only" if regime == "lmsys" else "LMSYS + WeirdChat"
        domain_label = "LMSYS" if domain == "lmsys" else "WeirdChat"
        figure.suptitle(
            (
                f"Gaussian forecast and fixed flow trajectory: {regime_label} on {domain_label}"
                if gaussian is not None
                else f"One fixed flow trajectory: {regime_label} model on {domain_label}"
            ),
            fontsize=16,
            y=0.995,
        )
        legend_handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=5,
                color="#F2C94C",
                label="Real rollouts",
            ),
        ]
        if gaussian is not None:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    color="#2A9D8F",
                    label="Gaussian forecast",
                )
            )
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=5,
                color="#D946C5",
                label="Flow state",
            )
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.958),
            frameon=False,
            ncol=len(legend_handles),
            fontsize=10,
        )
        figure.subplots_adjust(top=0.88)
        figure.savefig(output / f"pca_trajectory_{domain}.svg", bbox_inches="tight", dpi=180)
        figure.savefig(output / f"pca_trajectory_{domain}.pdf", bbox_inches="tight", dpi=180)
        plt.close(figure)

        lines.extend(
            [
                f"## {regime_label} model on {domain_label} test prompts",
                "",
                "| Progress | Update | Energy score / sqrt(d) | Variance ratio | 90% coverage |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['trajectory_percent']:.1f}% | {row['trajectory_step']} | "
                f"{row['raw_energy_score_per_sqrt_dimension']:.6f} | "
                f"{row['raw_generated_real_variance_trace_ratio']:.3f} | "
                f"{row['marginal_90pct_coverage']:.3f} |"
            )
        lines.append("")
        if gaussian is not None:
            selected_gaussian = result["gaussian_selection"]
            lines.extend(
                [
                    "The Gaussian panel uses the paper endpoint selected on LMSYS "
                    "validation: "
                    f"`{selected_gaussian['family']}` (rank "
                    f"{selected_gaussian['rank']}), "
                    f"`{selected_gaussian['scale_mode']}` scaling, and multiplier "
                    f"{selected_gaussian['sample_scale']}. The PCA basis is fit only "
                    "on training targets.",
                    "",
                ]
            )

    atomic_text(output / "RESULTS.md", "\n".join(lines) + "\n")
    atomic_text(
        output / "results.json",
        json.dumps({"schema_version": 1, "results": results}, indent=2) + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
