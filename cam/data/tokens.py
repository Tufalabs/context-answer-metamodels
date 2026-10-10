"""Reuse saved prompt-token activations with explicit unique-text row alignment."""

import fcntl
import json
import os
import shutil
import torch
from safetensors import safe_open
from cam.training import tokens as base
from cam.data.common import file_hash
from cam.data.common import read_rows


class SparseTokenStore:
    def __init__(self):
        self.parts = []
        self.index = {}

    def add(self, path, index_key):
        with safe_open(path, framework="pt", device="cpu") as handle:
            tokens = handle.get_tensor("tokens")
            offsets = handle.get_tensor("offsets")
            indices = handle.get_tensor(index_key).tolist()
        assert tokens.dtype == torch.bfloat16 and tokens.shape[1] == 4096
        assert len(offsets) == len(indices) + 1 and offsets[0] == 0 and offsets[-1] == len(tokens)
        assert (offsets[1:] > offsets[:-1]).all()
        part = len(self.parts)
        self.parts.append((tokens, offsets))
        for local, index in enumerate(indices):
            if index in self.index:
                raise ValueError(f"duplicate token source index {index}")
            self.index[index] = (part, local)

    def sequence(self, index):
        part, local = self.index[int(index)]
        tokens, offsets = self.parts[part]
        return tokens[int(offsets[local]) : int(offsets[local + 1])]

    def length(self, index):
        part, local = self.index[int(index)]
        offsets = self.parts[part][1]
        return int(offsets[local + 1] - offsets[local])


def cached_file(source, dest, expected_hash=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = dest.with_suffix(dest.suffix + ".verified.json")
    if dest.exists() and marker.exists():
        info = json.loads(marker.read_text())
        if info["size"] == source.stat().st_size == dest.stat().st_size and (
            expected_hash is None or info["sha256"] == expected_hash
        ):
            return dest
    temp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temp)
    digest = file_hash(temp)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"hash mismatch for {source}")
    os.replace(temp, dest)
    base.write_json_atomic(
        marker, {"source": str(source), "size": dest.stat().st_size, "sha256": digest}
    )
    return dest


def prepare_stores(project, campaign, cache, rows):
    version = file_hash(campaign / "prepared/manifest.json")
    root = cache / version
    root.mkdir(parents=True, exist_ok=True)
    stores = {"lmsys": SparseTokenStore(), "weirdchat": SparseTokenStore()}
    old_indices = {r["qwen_source_index"] for r in rows if r["qwen_source_index"] is not None}
    new_indices = {r["source_global_index"] for r in rows if r["qwen_source_index"] is None}
    old_root = project / "datasets/Qwen3.5-9B_layer18_full_prompt_tokens_500k"
    with (root / "stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for domain in stores:
            for path in sorted((old_root / domain / "parts").glob("tokens_*.safetensors")):
                lo, hi = map(int, path.stem.split("_")[-2:])
                if domain == "lmsys" and not any(lo <= i <= hi for i in old_indices):
                    continue
                staged = cached_file(path, root / "original" / domain / path.name)
                stores[domain].add(staged, "global_indices")
                print(f"mapped saved exact tokens {domain}/{path.name}", flush=True)
        if new_indices:
            tasks = json.loads((campaign / "prepared/collection_tasks.json").read_text())
            for task in tasks:
                if task["model"] != "qwen35" or task["seed"] != 43:
                    continue
                task_rows = read_rows(campaign / "prepared" / task["prompts"])
                if not any(r["source_global_index"] in new_indices for r in task_rows):
                    continue
                source = campaign / "collection" / f"task_{task['task']:05d}"
                complete = json.loads((source / "task_complete.json").read_text())
                assert complete["task"] == task and not complete["smoke"]
                expected = complete["verification"]["prompt_tokens_sha256"]
                staged = cached_file(
                    source / "prompt_tokens.safetensors",
                    root / "new" / f"task_{task['task']:05d}.safetensors",
                    expected,
                )
                stores["lmsys"].add(staged, "source_global_indices")
    for row in rows:
        index = (
            row["qwen_source_index"]
            if row["qwen_source_index"] is not None
            else row["source_global_index"]
        )
        assert stores["lmsys"].length(index) > 0
    return stores


def load_data(root, store, domain, partition, rows):
    tensors, identifiers = [], []
    count = 0
    for path in sorted((root / domain / partition).glob("*.safetensors")):
        if count == len(rows):
            break
        with safe_open(path, framework="pt", device="cpu") as handle:
            take = min(len(rows) - count, handle.get_slice("source_global_indices").get_shape()[0])
            tensors.append(handle.get_tensor("y_rollouts")[:take])
            identifiers.extend(handle.get_tensor("source_global_indices")[:take].tolist())
        count += take
    assert count == len(rows)
    assert identifiers == [r["source_global_index"] for r in rows]
    indices = torch.tensor(
        [
            (
                r["qwen_source_index"]
                if r["qwen_source_index"] is not None
                else r["source_global_index"]
            )
            if domain == "lmsys"
            else r["source_domain_index"]
            for r in rows
        ]
    )
    lengths = torch.tensor([store.length(i) for i in indices], dtype=torch.int32)
    print(f"{domain}/{partition}: {count} unique prompts, longest={int(lengths.max())}", flush=True)
    return base.DomainData(store, indices, torch.cat(tensors), lengths)
