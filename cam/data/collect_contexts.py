"""Collect compact 32-bin prompt activations without generating answers."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from cam.data import activation_collection as base

SEQUENCE_BINS = 32


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_atomic(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file({key: value.contiguous() for key, value in tensors.items()}, temporary, metadata)
    os.replace(temporary, path)


def prompt_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['index']}|{row['source_global_index']}|{row['prompt_sha256']}\n".encode()
        )
    return digest.hexdigest()


def make_batches(
    token_ids: list[list[int]], max_sequences: int, max_batch_tokens: int
) -> list[list[int]]:
    order = sorted(range(len(token_ids)), key=lambda index: (len(token_ids[index]), index))
    batches: list[list[int]] = []
    batch: list[int] = []
    maximum = 0
    for index in order:
        candidate_maximum = max(maximum, len(token_ids[index]))
        if batch and (
            len(batch) == max_sequences or candidate_maximum * (len(batch) + 1) > max_batch_tokens
        ):
            batches.append(batch)
            batch = []
            maximum = 0
        batch.append(index)
        maximum = max(maximum, len(token_ids[index]))
    if batch:
        batches.append(batch)
    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=sorted(base.MODEL_PROFILES), required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    profile = base.MODEL_PROFILES[args.model_id]
    if profile["loader"] != "causal_lm":
        raise RuntimeError("prefill-only collector currently supports causal-LM profiles")
    if not 0 <= args.layer < int(profile["expected_layers"]):
        raise ValueError("selected layer is outside the model")
    output = args.output_dir.resolve()
    complete = output / "verification.json"
    if complete.exists() and json.loads(complete.read_text())["status"] == "verified_complete":
        print(f"verified context collection already exists: {output}")
        return 0
    rows = base.jsonl_load(args.prompts_file.resolve())
    if not rows or [int(row["index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("prompt file indices are not contiguous")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=profile["revision"])
    tokenizer.padding_side = "right"
    rendered: list[str] = []
    token_ids: list[list[int]] = []
    for row in rows:
        text, ids = base.render_prompt(tokenizer, row["prompt"], profile["chat_template_kwargs"])
        if not ids or len(ids) > args.max_prompt_tokens:
            raise RuntimeError(
                f"prompt {row['index']} has {len(ids)} tokens; expected 1..{args.max_prompt_tokens}"
            )
        rendered.append(text)
        token_ids.append(ids)
    batches = make_batches(token_ids, args.max_sequences, args.max_batch_tokens)
    torch.set_float32_matmul_precision("high")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=profile["revision"],
        dtype=torch.bfloat16,
        device_map={"": torch.device("cuda:0")},
    )
    model.eval()
    blocks = base.resolve_decoder_blocks(model)
    config = getattr(model.config, "text_config", model.config)
    if (len(blocks), int(config.hidden_size)) != (
        int(profile["expected_layers"]),
        int(profile["expected_hidden"]),
    ):
        raise RuntimeError("loaded architecture does not match the pinned model profile")
    hidden_size = int(profile["expected_hidden"])
    prompt_bins = torch.empty((len(rows), SEQUENCE_BINS, hidden_size), dtype=torch.bfloat16)
    x_last = torch.empty((len(rows), hidden_size), dtype=torch.float32)
    device = next(model.parameters()).device
    completed = 0
    for batch_index, indices in enumerate(batches):
        maximum = max(len(token_ids[index]) for index in indices)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has neither a pad nor EOS token")
        input_ids = torch.full(
            (len(indices), maximum), int(pad_id), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_position, index in enumerate(indices):
            length = len(token_ids[index])
            input_ids[row_position, :length] = torch.tensor(
                token_ids[index], dtype=torch.long, device=device
            )
            attention_mask[row_position, :length] = 1
        captured = base.capture_block_outputs(
            model, blocks, input_ids, attention_mask, [args.layer]
        )[args.layer]
        for row_position, index in enumerate(indices):
            length = len(token_ids[index])
            hidden = captured[row_position, :length].float()
            pooled = F.adaptive_avg_pool1d(hidden.T.unsqueeze(0), SEQUENCE_BINS).squeeze(0).T
            prompt_bins[index].copy_(pooled.to(device="cpu", dtype=torch.bfloat16))
            x_last[index].copy_(hidden[-1].to(device="cpu", dtype=torch.float32))
        completed += len(indices)
        if batch_index % 25 == 0 or completed == len(rows):
            print(
                f"{args.model_id}: completed {completed:,}/{len(rows):,} prompts "
                f"({batch_index + 1:,}/{len(batches):,} batches)",
                flush=True,
            )
        del input_ids, attention_mask, captured
    if not torch.isfinite(prompt_bins.float()).all() or not torch.isfinite(x_last).all():
        raise RuntimeError("collected prompt activations contain non-finite values")
    raw_template = tokenizer.chat_template
    if isinstance(raw_template, dict):
        raw_template = json.dumps(raw_template, sort_keys=True)
    if not isinstance(raw_template, str):
        raise RuntimeError("tokenizer has no resolved chat template")
    activation_path = output / "contexts.safetensors"
    save_atomic(
        activation_path,
        {
            "prompt_bins": prompt_bins,
            "x_last": x_last,
            "prompt_lengths": torch.tensor([len(ids) for ids in token_ids], dtype=torch.int64),
            "source_global_indices": torch.tensor(
                [int(row["source_global_index"]) for row in rows], dtype=torch.int64
            ),
        },
        {
            "schema_version": "1",
            "model_id": args.model_id,
            "model_revision": profile["revision"],
            "layer": str(args.layer),
            "sequence_bins": str(SEQUENCE_BINS),
            "prompt_digest": prompt_digest(rows),
        },
    )
    manifest = {
        "schema_version": 1,
        "status": "verified_complete",
        "model_id": args.model_id,
        "model_revision": profile["revision"],
        "layer": args.layer,
        "layer_fraction": (args.layer + 1) / int(profile["expected_layers"]),
        "contexts": len(rows),
        "sequence_bins": SEQUENCE_BINS,
        "prompt_digest": prompt_digest(rows),
        "prompt_tokens": {
            "minimum": min(map(len, token_ids)),
            "maximum": max(map(len, token_ids)),
            "mean": sum(map(len, token_ids)) / len(token_ids),
        },
        "chat_template_sha256": base.sha256_text(raw_template),
        "chat_template_kwargs": profile["chat_template_kwargs"],
        "activation_file": activation_path.name,
        "activation_sha256": base.sha256_file(activation_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_atomic(output / "manifest.json", manifest)
    write_json_atomic(output / "verification.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
