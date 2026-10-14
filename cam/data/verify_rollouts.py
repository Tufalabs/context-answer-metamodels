#!/usr/bin/env python3
"""Verify full-token activation collection structure without rereading terabytes."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open

from cam.data import activation_collection as base

SCHEMA_VERSION = 3


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    file_size = path.stat().st_size
    data_size = file_size - 8 - header_size
    for key, value in header.items():
        if key == "__metadata__":
            continue
        start, end = value["data_offsets"]
        if not 0 <= start <= end <= data_size:
            raise RuntimeError(f"invalid safetensors offsets for {path}:{key}")
    return header


def jsonl_rows(paths: list[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in base.jsonl_load(path)]


def check_summary_header(
    path: Path,
    expected: dict[str, tuple[int, int, int]],
) -> None:
    header = read_safetensors_header(path)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if set(tensors) != set(expected):
        raise RuntimeError(f"unexpected summary keys in {path}: {set(tensors)}")
    for key, shape in expected.items():
        if tensors[key]["dtype"] != "F32" or tuple(tensors[key]["shape"]) != shape:
            raise RuntimeError(f"unexpected summary tensor {path}:{key}: {tensors[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    generation_complete = json.loads(
        (run_dir / "generation_complete.json").read_text(encoding="utf-8")
    )
    collection_complete = json.loads(
        (run_dir / "collection_complete.json").read_text(encoding="utf-8")
    )
    n = int(config["n_contexts_requested"])
    layers = len(config["layers"])
    hidden = int(config["hidden_size"])
    if generation_complete.get("status") != "generation_complete":
        raise RuntimeError("generation completion record is invalid")
    if collection_complete.get("status") != "collection_complete_pending_verification":
        raise RuntimeError("collection completion record is invalid")
    if config.get("activation_storage_dtype") != "bfloat16":
        raise RuntimeError("full-token activation dtype is not bfloat16")
    if config.get("upstream_rollout_artifacts_used") is not False:
        raise RuntimeError("fresh rollout provenance is not attested")

    prompts = base.jsonl_load(run_dir / "data" / "prompts.jsonl")
    context_rows = jsonl_rows(
        sorted((run_dir / "data" / "context_tokens").glob("*.jsonl"))
    )
    rollouts = jsonl_rows(sorted((run_dir / "data" / "rollouts").glob("*.jsonl")))
    expected_indices = list(range(n))
    for label, rows in (
        ("prompts", prompts),
        ("context tokens", context_rows),
        ("rollouts", rollouts),
    ):
        if len(rows) != n or [row["index"] for row in rows] != expected_indices:
            raise RuntimeError(f"{label} coverage is invalid")
    for index, (context, rollout) in enumerate(zip(context_rows, rollouts, strict=True)):
        generated = rollout.get("generated_token_ids")
        if rollout.get("generated_token_count") != len(generated):
            raise RuntimeError(f"generated-token count mismatch at row {index}")
        if len(context["prompt_token_ids"]) + len(generated) > config["generation"][
            "max_model_len"
        ]:
            raise RuntimeError(f"model length exceeded at row {index}")

    manifest_paths = sorted(
        (run_dir / "activations" / "shard_manifests").glob("full_*.json")
    )
    if not manifest_paths:
        raise RuntimeError("no activation shard manifests found")
    covered: list[int] = []
    trusted_records: dict[str, dict[str, Any]] = {}
    total_activation_tokens = 0
    full_activation_bytes = 0
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        start = int(manifest["start"])
        end = int(manifest["end_exclusive"])
        if manifest.get("finite") is not True or manifest.get("n_rows") != end - start:
            raise RuntimeError(f"invalid capture attestation: {manifest_path}")
        covered.extend(range(start, end))
        records = {Path(record["path"]).parts[1]: record for record in manifest["files"]}
        if set(records) != {"full_tokens", "context", "answer", "activation_spans"}:
            raise RuntimeError(f"unexpected shard file set: {manifest_path}")
        for record in manifest["files"]:
            relative = record["path"]
            if relative in trusted_records:
                raise RuntimeError(f"duplicate artifact record: {relative}")
            path = run_dir / relative
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise RuntimeError(f"missing or wrong-sized shard artifact: {path}")
            trusted_records[relative] = record

        span_rows = base.jsonl_load(run_dir / records["activation_spans"]["path"])
        if [row["index"] for row in span_rows] != list(range(start, end)):
            raise RuntimeError(f"activation span coverage mismatch: {manifest_path}")
        spans = {row["index"]: row for row in span_rows}
        full_path = run_dir / records["full_tokens"]["path"]
        full_header = read_safetensors_header(full_path)
        tensor_header = {
            key: value for key, value in full_header.items() if key != "__metadata__"
        }
        expected_keys = {
            key
            for index in range(start, end)
            for key in (f"hidden_{index:06d}", f"token_ids_{index:06d}")
        }
        if set(tensor_header) != expected_keys:
            raise RuntimeError(f"unexpected full-token keys in {full_path}")
        with safe_open(full_path, framework="pt", device="cpu") as handle:
            for index in range(start, end):
                prompt_ids = context_rows[index]["prompt_token_ids"]
                generated_ids = rollouts[index]["generated_token_ids"]
                span = spans[index]
                suffix_ids = span["canonical_assistant_suffix_token_ids"]
                overlap = int(span["generated_suffix_overlap"])
                appended_suffix_ids = span["appended_suffix_token_ids"]
                expected_overlap = (
                    base.suffix_overlap(generated_ids, suffix_ids)
                    if rollouts[index]["finish_reason"] == "stop"
                    else 0
                )
                if overlap != expected_overlap:
                    raise RuntimeError(f"assistant-suffix overlap mismatch at row {index}")
                expected_appended = (
                    suffix_ids[overlap:]
                    if rollouts[index]["finish_reason"] == "stop"
                    else []
                )
                if appended_suffix_ids != expected_appended:
                    raise RuntimeError(f"appended assistant suffix mismatch at row {index}")
                expected_ids = [*prompt_ids, *generated_ids, *appended_suffix_ids]
                hidden_key = f"hidden_{index:06d}"
                ids_key = f"token_ids_{index:06d}"
                hidden_info = tensor_header[hidden_key]
                ids_info = tensor_header[ids_key]
                expected_shape = (layers, len(expected_ids), hidden)
                if hidden_info["dtype"] != "BF16" or tuple(hidden_info["shape"]) != expected_shape:
                    raise RuntimeError(
                        f"unexpected hidden tensor at row {index}: {hidden_info}"
                    )
                if ids_info["dtype"] != "I64" or tuple(ids_info["shape"]) != (
                    len(expected_ids),
                ):
                    raise RuntimeError(f"unexpected token tensor at row {index}: {ids_info}")
                stored_ids = handle.get_tensor(ids_key).tolist()
                if stored_ids != expected_ids:
                    raise RuntimeError(f"stored token IDs differ at row {index}")
                total_activation_tokens += len(expected_ids)

        shard_n = end - start
        check_summary_header(
            run_dir / records["context"]["path"],
            {
                "cx_last": (shard_n, layers, hidden),
                "cx_mean": (shard_n, layers, hidden),
            },
        )
        check_summary_header(
            run_dir / records["answer"]["path"],
            {"v_answer_mean": (shard_n, layers, hidden)},
        )
        for row in span_rows:
            index = row["index"]
            if row["prompt_length"] != len(context_rows[index]["prompt_token_ids"]):
                raise RuntimeError(f"prompt span mismatch at row {index}")
            if row["generated_length"] != rollouts[index]["generated_token_count"]:
                raise RuntimeError(f"answer span mismatch at row {index}")
            generated_end = row["prompt_length"] + row["generated_length"]
            if row["generated_span_end_exclusive"] != generated_end:
                raise RuntimeError(f"generated span boundary mismatch at row {index}")
            if row["assistant_suffix_span_start"] != generated_end:
                raise RuntimeError(f"assistant suffix start mismatch at row {index}")
            if row["appended_suffix_length"] != len(row["appended_suffix_token_ids"]):
                raise RuntimeError(f"assistant suffix length mismatch at row {index}")
            if row["full_length"] != (
                generated_end + row["appended_suffix_length"]
            ):
                raise RuntimeError(f"full activation span mismatch at row {index}")
        total_activation_tokens_in_manifest = int(manifest["total_tokens"])
        if total_activation_tokens_in_manifest != sum(
            len(context_rows[index]["prompt_token_ids"])
            + rollouts[index]["generated_token_count"]
            + spans[index]["appended_suffix_length"]
            for index in range(start, end)
        ):
            raise RuntimeError(f"token total mismatch: {manifest_path}")
        full_activation_bytes += records["full_tokens"]["bytes"]
    if covered != expected_indices:
        raise RuntimeError("activation shard coverage is incomplete or overlapping")

    excluded = {"artifact_manifest.json", "verification.json"}
    inventory: dict[str, dict[str, Any]] = dict(trusted_records)
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded or relative in inventory:
            continue
        inventory[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
    ordered_inventory = [inventory[key] for key in sorted(inventory)]
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": base.utc_now(),
        "run_dir": str(run_dir),
        "n_files": len(ordered_inventory),
        "total_bytes": sum(record["bytes"] for record in ordered_inventory),
        "full_activation_bytes": full_activation_bytes,
        "full_activation_hash_provenance": (
            "SHA-256 computed on node-local completed shards before atomic durable copy; "
            "durable byte sizes and safetensors headers independently verified"
        ),
        "files": ordered_inventory,
    }
    base.json_dump_atomic(run_dir / "artifact_manifest.json", artifact_manifest)

    lengths = [row["generated_token_count"] for row in rollouts]
    finish_reasons = Counter(str(row.get("finish_reason")) for row in rollouts)
    verification = {
        "schema_version": SCHEMA_VERSION,
        "verified_at": base.utc_now(),
        "status": "verified_complete",
        "n_contexts": n,
        "fresh_rollouts_attested": True,
        "full_token_activations": {
            "rows": n,
            "layers": layers,
            "hidden_size": hidden,
            "dtype": "torch.bfloat16",
            "finite_at_capture": True,
            "total_context_and_answer_tokens": total_activation_tokens,
            "bytes": full_activation_bytes,
            "layout": (
                "[layers, prompt_plus_generated_plus_template_suffix_tokens, hidden] "
                "per row"
            ),
        },
        "derived_summaries": {
            "context_keys": ["cx_last", "cx_mean"],
            "answer_keys": ["v_answer_mean"],
            "answer_span": (
                "exact generated tokens plus non-overlapping canonical assistant "
                "suffix, matching issue 779"
            ),
            "dtype": "torch.float32",
        },
        "generation": {
            "total_generated_tokens": sum(lengths),
            "maximum_generated_tokens": max(lengths),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "length_terminated": finish_reasons.get("length", 0),
        },
        "manifest_sha256": base.sha256_file(run_dir / "artifact_manifest.json"),
        "total_artifact_bytes": artifact_manifest["total_bytes"],
    }
    temporary = run_dir / f".verification.json.tmp-{os.getpid()}"
    base.json_dump_atomic(temporary, verification)
    os.replace(temporary, run_dir / "verification.json")
    print(json.dumps(verification, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
