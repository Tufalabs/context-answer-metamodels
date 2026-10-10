"""Verify the active R² results and export standalone Appendix A figures."""

import argparse
import csv
import json
import math
from pathlib import Path
from cam.data.representation_spec import ACTIVE_TASKS
from cam.data.representation_spec import FAMILIES
from cam.data.representation_spec import REPRESENTATIONS
from cam.data.representation_spec import DOMAINS
from cam.data.representation_spec import cell
from cam.data.representation_spec import validate_evaluations
from cam.data.common import APPENDIX_REPRESENTATIONS
from cam.data.common import write_json
from cam.data.common import file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    a = p.parse_args()
    root = a.campaign / "appendix_a"
    rows = []
    sources = {}
    for family, representation, n in ACTIVE_TASKS:
        path = root / "fits" / cell(family, representation, n) / "results.json"
        value = json.loads(path.read_text())
        assert (
            value["status"] == "complete"
            and value["model_id"] == "Qwen/Qwen3.5-9B"
            and value["layer"] == 18
        )
        assert (value["family"], value["representation"], value["n_train"]) == (
            family,
            representation,
            n,
        )
        assert value["n_validation"] == 1000 and set(value["evaluations"]) == set(DOMAINS)
        validate_evaluations(
            value["evaluations"],
            allow_unrecorded_ci=value.get("reused") is True
            and (family, representation) == ("flow", "32_bins"),
        )
        for domain, count in DOMAINS.items():
            metrics = value["evaluations"][domain]
            assert metrics["n_contexts"] == count
            rows.append(
                {
                    "family": family,
                    "representation": representation,
                    "n_train": n,
                    "domain": domain,
                    **metrics,
                    "r2_ci_status": metrics.get("r2_ci_status", "recorded"),
                    "reused_core_fit": value["reused"],
                }
            )
        sources[str(path.relative_to(root))] = file_hash(path)
    assert len(rows) == len(ACTIVE_TASKS) * len(DOMAINS)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "r2_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2563eb", "#e76f00", "#15915d", "#8c4fc8")
    labels = {
        "32_bins": "32 bins",
        "every_token": "Every token",
        "last_token": "Last token",
        "mean_token": "Mean token",
    }
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, layout="constrained")
    for i, family in enumerate(FAMILIES):
        for j, domain in enumerate(DOMAINS):
            ax = axes[i, j]
            for representation, color in zip(REPRESENTATIONS, colors):
                if representation not in APPENDIX_REPRESENTATIONS[family]:
                    continue
                selected = [
                    r
                    for r in rows
                    if (r["family"], r["domain"], r["representation"])
                    == (family, domain, representation)
                ]
                x = [r["n_train"] for r in selected]
                y = [r["r2"] for r in selected]
                ax.plot(x, y, color=color, label=labels[representation], linewidth=1.8)
                ax.fill_between(
                    x,
                    [r["r2_ci_low"] if r["r2_ci_low"] is not None else math.nan for r in selected],
                    [
                        r["r2_ci_high"] if r["r2_ci_high"] is not None else math.nan
                        for r in selected
                    ],
                    color=color,
                    alpha=0.1,
                )
            ax.set_xscale("log")
            ax.grid(alpha=0.18)
            ax.set_title(f"{family.upper() if family == 'mlp' else family.title()} · {domain}")
            if j == 0:
                ax.set_ylabel("R² of the predicted answer mean")
            if i == 2:
                ax.set_xlabel("Unique LMSYS training prompts")
    handles, names = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(output / "appendix_a_r2.pdf")
    fig.savefig(output / "appendix_a_r2.png", dpi=200)
    plt.close(fig)
    write_json(
        root / "APPENDIX_COMPLETE.json",
        {
            "status": "complete",
            "model_id": "Qwen/Qwen3.5-9B",
            "layer": 18,
            "fit_cells": len(ACTIVE_TASKS),
            "domain_r2_rows": len(rows),
            "representations": list(REPRESENTATIONS),
            "families": list(FAMILIES),
            "representations_by_family": APPENDIX_REPRESENTATIONS,
            "source_result_sha256": sources,
            "paper_updates": False,
            "r2_ci_unavailable_rows": sum(r["r2_ci_low"] is None for r in rows),
            "confidence_interval_note": "Reused core flow estimates have no saved R2 confidence intervals; their R2 values are preserved and no interval is fabricated.",
        },
    )


if __name__ == "__main__":
    main()
