"""Export completed experiment metrics to CSV; refuse missing fits."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--campaign", type=Path, required=True)
parser.add_argument("--behavior", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.campaign.resolve()
SRC = ROOT
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
SIZES = [
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    20000,
    28600,
    50000,
    75000,
    100000,
    200000,
    300000,
    500000,
]
DOMAINS = {"LMSYS": 3000, "WeirdChat": 2660, "IFEval": 541}
REPRESENTATIONS = {
    "linear": ("32_bins", "last_token", "mean_token"),
    "mlp": ("32_bins", "every_token", "last_token", "mean_token"),
    "flow": ("32_bins", "every_token", "last_token", "mean_token"),
}


def read(path):
    return json.loads(path.read_text())


def records(pattern):
    result = []
    for path in sorted(SRC.glob(pattern)):
        r = read(path)
        assert r["status"] == "complete", path
        r["_source"] = str(path.relative_to(ROOT))
        result.append(r)
    return result


def write(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


main = records("evaluations/main/*/results.json")
cross = records("fits/cross/*/*/*/*/*/results.json")
rollout = records("fits/rollout_count/*/*/*/train_100000/results.json")
appendix = [
    r
    for r in records("appendix_a/fits/*/*/*/results.json")
    if r["representation"] in REPRESENTATIONS[r["family"]]
]


def reportable(r):
    return r is not None


layers = records("layers/fits/*/results.json")
assert (len(main), len(cross), len(rollout), len(layers)) == (15, 352, 45, 32)
assert read(SRC / "prepared/validation.json")["zero_split_overlap"]
rows = []
aux = []
selections = []
for r in sorted(main, key=lambda r: r["n_train_contexts"]):
    n = r["n_train_contexts"]
    assert r["domains"] == DOMAINS
    for f in ("linear", "mlp", "flow", "gaussian"):
        for d in DOMAINS:
            m = r["distribution"][f][d] if f == "gaussian" else r["point"][f][d]
            z = {
                "n_train": n,
                "domain": d,
                "family": f,
                "mean_r2": m["sample_mean_r2"],
                "retrieval": m["retrieval_top1_accuracy"],
                "n_retrieval_candidates": DOMAINS[d],
                "source": r["_source"],
            }
            if f in ("flow", "gaussian"):
                m = r["distribution"][f][d]
                z.update(
                    energy=m["raw_energy_score_per_sqrt_dimension"],
                    variance_spearman=m["uncertainty"]["spearman"],
                    variance_ratio=m["raw_generated_real_variance_trace_ratio"],
                )
            rows.append(z)
    for f, m in r["auxiliary_lmsys"].items():
        aux.append(
            {
                "n_train": n,
                "family": f,
                **{k: m[k] for k in ("nll_raw", "predictive_mmd2", "predictive_sliced_w2")},
                "source": r["_source"],
            }
        )
    s = r["gaussian_selection"]
    selected = s["selected"]
    selections.append(
        {
            "n_train": n,
            **{k: selected[k] for k in ("family", "rank", "scale_mode", "sample_scale")},
            **{
                mode + "_validation_energy": min(
                    x["raw_energy_score_per_sqrt_dimension"]
                    for x in s["selected_by_family_mode"]
                    if x["scale_mode"] == mode
                )
                for mode in ("global", "heteroscedastic")
            },
            "source": r["_source"],
        }
    )
write("unique_main_metrics.csv", rows)
write("unique_main_auxiliary.csv", aux)
write("unique_gaussian_selection.csv", selections)
for target in ("qwen35", "gemma"):
    rows = []
    aux = []
    for r in cross:
        if r["target"] != target:
            continue
        m = r["test_metrics"]
        f = r["family"]
        c = r["conditioner"]
        z = {
            "condition_hidden": 3584 if c == "qwen25" else 4096 if c == "qwen35" else 3840,
            "conditioner": c,
            "family": f,
            "n_train_contexts": r["n_train"],
            "replicate": r["replicate"],
            "sample_mean_r2": m["sample_mean_r2"],
            "retrieval_top1_accuracy": m["retrieval_top1_accuracy"],
            "n_retrieval_candidates": r["n_test"],
            "source": r["_source"],
        }
        if f in ("gaussian", "flow"):
            z.update(
                distribution_energy_per_sqrt_dimension=m[
                    "test_raw_energy_score_per_sqrt_dimension"
                ],
                generated_real_variance_trace_ratio=m[
                    "test_raw_generated_real_variance_trace_ratio"
                ],
                uncertainty_spearman=m["uncertainty"]["spearman"],
            )
            label = {"qwen25": "Qwen2.5", "qwen35": "Qwen3.5", "gemma": "Gemma 4"}
            aux.append(
                {
                    "conditioner": f"{label[c]} -> {label[target]}",
                    "family": f,
                    "n_train": r["n_train"],
                    "replicate": r["replicate"],
                    **{
                        k: m["auxiliary"][k]
                        for k in ("nll_raw", "predictive_mmd2", "predictive_sliced_w2")
                    },
                    "source": r["_source"],
                }
            )
        rows.append(z)
    write(f"unique_cross_{target}.csv", rows)
    write(f"unique_cross_{target}_auxiliary.csv", aux)
rows = []
for f in ("linear", "mlp", "flow"):
    for n in [100000]:
        for s in (2, 4, 8, 12, 16):
            group = [
                r
                for r in rollout
                if (r["family"], r["n_train_contexts"], r["n_train_rollout_seeds"]) == (f, n, s)
            ]
            assert len(group) == 3
            v = [
                next(m["sample_mean_r2"] for m in r["evaluations"] if m["split"] == "test")
                for r in group
            ]
            rows.append(
                {
                    "family": f,
                    "n_train_contexts": n,
                    "n_train_rollout_seeds": s,
                    "split": "test",
                    "sample_mean_r2_mean": statistics.mean(v),
                    "sample_mean_r2_std": statistics.stdev(v),
                    "source": ";".join(r["_source"] for r in group),
                }
            )
write("unique_rollout_count.csv", rows)
app = {(r["family"], r["representation"], r["n_train"]): r for r in appendix}
assert len(app) == len(appendix) == 165
flat = []
missing = []
for f in ("linear", "mlp", "flow"):
    for rep in REPRESENTATIONS[f]:
        for n in SIZES:
            r = app.get((f, rep, n))
            if r is None:
                missing.append({"family": f, "representation": rep, "n_train": n})
            for d, count in DOMAINS.items():
                if r:
                    assert r["evaluations"][d]["n_contexts"] == count
                    assert math.isfinite(r["evaluations"][d]["r2"])
                flat.append(
                    {
                        "family": f,
                        "representation": rep,
                        "n_train": n,
                        "domain": d,
                        "r2": r["evaluations"][d]["r2"] if reportable(r) else "",
                        "raw_r2": r["evaluations"][d]["r2"] if r else "",
                        "status": "complete" if r else "pending",
                        "source": r["_source"] if r else "",
                    }
                )
write("unique_appendix_r2.csv", flat)
for f in ("mlp", "flow"):
    for d in DOMAINS:
        wide = []
        for n in SIZES:
            z = {"n_train": n, "n_lmsys_train": n}
            for rep, key in [
                ("32_bins", "binned_32"),
                ("every_token", "exact_token" if f == "mlp" else "token_query"),
                ("last_token", "last_token"),
                ("mean_token", "mean_token"),
            ]:
                r = app.get((f, rep, n))
                z[key + ("_mlp" if f == "mlp" else "_sample_mean_r2")] = (
                    r["evaluations"][d]["r2"] if reportable(r) else ""
                )
            wide.append(z)
        write(f"unique_{f}_representations_{d.lower()}.csv", wide)
linear = []
for rep, key in [("32_bins", "binned"), ("last_token", "last"), ("mean_token", "mean")]:
    for n in SIZES:
        r = app.get(("linear", rep, n))
        linear.append(
            {
                "representation": key,
                "n_train": n,
                **{
                    d.lower() + "_r2": r["evaluations"][d]["r2"] if reportable(r) else ""
                    for d in DOMAINS
                },
            }
        )
write("unique_linear_representations.csv", linear)
write(
    "unique_layer_selection.csv",
    [
        {
            "layer": r["layer"],
            "validation_r2": r["validation"]["r2"],
            "test_r2": r["test"]["r2"],
            "source": r["_source"],
        }
        for r in sorted(layers, key=lambda r: r["layer"])
    ],
)
probe_source = args.behavior / "summary/selected_test_metrics.csv"
probe = list(csv.DictReader(probe_source.open()))
order = [
    "any_target_behavior",
    "any_safety_failure",
    "self_harm_or_eating_disorder",
    "deception_or_false_capability",
]
write("unique_probes.csv", sorted(probe, key=lambda r: order.index(r["event"])))
summary = {
    "source_campaign": str(ROOT),
    "probe_campaign": str(args.behavior),
    "main_sizes": len(main),
    "cross_fits": len(cross),
    "rollout_fits": len(rollout),
    "layer_fits": len(layers),
    "representation_fits": len(appendix),
    "missing": missing,
}
(OUT / "export_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
