"""Resolve rollout and activation rows from a verified extension overlay."""

import json
from pathlib import Path

from .common import file_hash, read_rows


def iter_run_rows(*, overlay, source_run):
    overlay = Path(overlay)
    manifest = json.loads((overlay / "manifest.json").read_text())
    if manifest["status"] != "verified_complete":
        raise ValueError("Collection extension is incomplete")
    run = next(run for run in manifest["runs"] if run["run"] == source_run)
    path = overlay / run["row_index"]
    assert file_hash(path=path) == run["row_index_sha256"]
    for row in read_rows(path=path):
        yield row


def answer_tensor_location(*, row):
    root, index = Path(row["path"]), row["index"]
    for path in sorted((root / "activations/answer").glob("answer_*.safetensors")):
        start, end = map(int, path.stem.split("_")[-2:])
        if start <= index <= end:
            return path, index - start
    raise FileNotFoundError(f"No answer activation for {root}:{index}")
