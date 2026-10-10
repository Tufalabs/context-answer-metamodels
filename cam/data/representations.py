"""Aligned compact and exact-token inputs for context representation ablations."""

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from cam.training import scaling as compact
from cam.training import tokens as exact
from cam.data.common import read_rows
from cam.data.tokens import SparseTokenStore
from cam.data.tokens import prepare_stores
from cam.data.tokens import load_data


def vector_mean(root, part, limit=None):
    pieces = []
    loaded = 0
    for path in sorted((root / "lmsys" / part).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            n = f.get_slice("x_mean").get_shape()[0]
            take = n if limit is None else min(n, limit - loaded)
            pieces.append(f.get_slice("x_mean")[:take].clone())
        loaded += take
        if limit is not None and loaded == limit:
            break
    if limit is not None:
        assert loaded == limit
    return torch.cat(pieces)


def load_compact(main, appendix, representation, n):
    family = "mlp" if representation == "32_bins" else "linear"
    data = {
        part: compact.load_partition(main, "lmsys", part, family, n if part == "train" else None)
        for part in ("train", "validation", "test")
    }
    if representation == "mean_token":
        for part, value in data.items():
            value.x = vector_mean(appendix / "vectors", part, n if part == "train" else None)
    tests = {"LMSYS": data["test"]}
    key = {"32_bins": "prompt_bins", "last_token": "x_last", "mean_token": "x_mean"}[representation]
    for domain, filename in [("WeirdChat", "weirdchat"), ("IFEval", "ifeval")]:
        v = load_file(appendix / "ood" / f"{filename}.safetensors")
        tests[domain] = compact.DomainData(v[key], v["y_rollouts"])
    assert len(data["validation"].x) == 1000
    return data["train"], data["validation"], tests


def load_exact(project, campaign, main, appendix, cache, n):
    rows = {
        part: read_rows(
            campaign / "prepared/lmsys" / f"{part}.jsonl", n if part == "train" else None
        )
        for part in ("train", "validation", "test")
    }
    stores = prepare_stores(project, campaign, cache, sum(rows.values(), []))
    data = {
        part: load_data(main, stores["lmsys"], "lmsys", part, values)
        for part, values in rows.items()
    }
    weird = sorted(
        sum(
            (
                read_rows(campaign / "prepared/weirdchat" / f"{part}.jsonl")
                for part in ("train", "validation", "test")
            ),
            [],
        ),
        key=lambda r: r["source_domain_index"],
    )
    weird = [{**r, "source_global_index": r["source_domain_index"]} for r in weird]
    tests = {
        "LMSYS": data["test"],
        "WeirdChat": load_data(main, stores["weirdchat"], "weirdchat", "all", weird),
    }
    ifeval_store = SparseTokenStore()
    ifeval_store.add(appendix / "ood/ifeval_tokens.safetensors", "global_indices")
    v = load_file(appendix / "ood/ifeval.safetensors")
    indices = torch.arange(541)
    tests["IFEval"] = exact.DomainData(
        ifeval_store,
        indices,
        v["y_rollouts"],
        torch.tensor([ifeval_store.length(i) for i in indices], dtype=torch.int32),
    )
    assert len(data["validation"]) == 1000
    return data["train"], data["validation"], tests
