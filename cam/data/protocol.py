"""Validate the answer-token cap of evaluation inputs and results."""

import json
from pathlib import Path


def require_collection_cap(*, directory, expected=4096):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if (
        manifest.get("status") != "verified_complete"
        or manifest.get("generation", {}).get("max_new_tokens") != expected
    ):
        raise ValueError(
            f"{directory}: expected a verified {expected}-token evaluation collection; initialize a new run with matching inputs"
        )


def require_tensor_cap(*, path, expected=4096):
    from safetensors import safe_open

    with safe_open(filename=path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("answer_max_new_tokens") != str(expected):
        raise ValueError(
            f"{path}: missing or mismatched evaluation answer-token cap (expected {expected})"
        )


def require_result_cap(*, result, path, expected=4096):
    if result.get("evaluation_generation_max_new_tokens") != expected:
        raise ValueError(
            f"{path}: results do not attest the {expected}-token evaluation cap; "
            "rerun evaluation in a new output directory"
        )
