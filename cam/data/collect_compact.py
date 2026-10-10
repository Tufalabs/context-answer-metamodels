"""Collect compact answer means; optionally retain exact prompt-token states.

Adapted from the verified cross-family compact collector. Decoding, activation
boundaries, and answer pooling remain unchanged. No answer-token tensors persist.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
from cam.data import activation_collection as base
from cam.data import rollouts as collect_full


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


def make_batches(lengths: list[int], max_sequences: int, max_batch_tokens: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    result: list[list[int]] = []
    batch: list[int] = []
    maximum = 0
    for index in order:
        candidate_maximum = max(maximum, lengths[index])
        if batch and (
            len(batch) == max_sequences or candidate_maximum * (len(batch) + 1) > max_batch_tokens
        ):
            result.append(batch)
            batch = []
            maximum = 0
        batch.append(index)
        maximum = max(maximum, lengths[index])
    if batch:
        result.append(batch)
    return result


def prompt_digest(rows: list[dict[str, Any]]) -> str:
    value = hashlib.sha256()
    for row in rows:
        value.update(
            f"{row['index']}|{row['source_global_index']}|{row['prompt_sha256']}\n".encode()
        )
    return value.hexdigest()


def response_digest(rows: list[dict[str, Any]]) -> str:
    value = hashlib.sha256()
    for row in rows:
        value.update(f"{row['index']}|{row['response_sha256']}\n".encode())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(base.MODEL_PROFILES), required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-contexts", type=int, required=True)
    parser.add_argument("--max-full-tokens", type=int, default=8_192)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=8_192)
    parser.add_argument("--save-prompt-tokens", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_dir = args.run_dir.resolve()
    verification_path = run_dir / "onpolicy_verification.json"
    if (
        verification_path.exists()
        and json.loads(verification_path.read_text()).get("status") == "verified_complete"
    ):
        print(f"verified compact on-policy activations already exist: {run_dir}")
        return 0
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if config["model_id"] != args.model_id or int(config["generation"]["seed"]) < 0:
        raise RuntimeError("generation config does not match requested candidate model")
    if not (run_dir / "generation_complete.json").exists():
        raise RuntimeError("candidate generation is incomplete")
    prompts = base.jsonl_load(run_dir / "data" / "prompts.jsonl")
    context_rows = [
        row
        for path in sorted((run_dir / "data" / "context_tokens").glob("*.jsonl"))
        for row in base.jsonl_load(path)
    ]
    rollouts = collect_full.load_rollouts(run_dir, args.n_contexts)
    if not (
        len(prompts) == len(context_rows) == len(rollouts) == args.n_contexts
        and [int(row["index"]) for row in prompts] == list(range(args.n_contexts))
    ):
        raise RuntimeError("on-policy prompt, token, or rollout coverage is incomplete")
    profile = base.MODEL_PROFILES[args.model_id]
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=profile["revision"])
    tokenizer.padding_side = "right"
    _suffix_text, suffix_ids = base.assistant_suffix(tokenizer, profile["chat_template_kwargs"])
    token_ids: list[list[int]] = []
    answer_starts: list[int] = []
    for index, (context, rollout) in enumerate(zip(context_rows, rollouts, strict=True)):
        prompt_ids = list(context["prompt_token_ids"])
        generated_ids = list(rollout["generated_token_ids"])
        if not generated_ids:
            raise RuntimeError(f"empty candidate rollout at row {index}")
        overlap = (
            base.suffix_overlap(generated_ids, suffix_ids)
            if rollout["finish_reason"] == "stop"
            else 0
        )
        appended = suffix_ids[overlap:] if rollout["finish_reason"] == "stop" else []
        full = [*prompt_ids, *generated_ids, *appended]
        if len(full) > args.max_full_tokens:
            raise RuntimeError(f"candidate row {index} has {len(full)} full tokens")
        token_ids.append(full)
        answer_starts.append(len(prompt_ids))
    lengths = list(map(len, token_ids))
    batches = make_batches(lengths, args.max_sequences, args.max_batch_tokens)
    torch.set_float32_matmul_precision("high")
    loader = AutoModelForCausalLM if profile["loader"] == "causal_lm" else AutoModelForMultimodalLM
    model = loader.from_pretrained(
        args.model_id,
        revision=profile["revision"],
        dtype=torch.bfloat16,
        device_map={"": torch.device("cuda:0")},
    )
    model.eval()
    blocks = base.resolve_decoder_blocks(model)
    hidden = int(profile["expected_hidden"])
    model_config = getattr(model.config, "text_config", model.config)
    if (len(blocks), int(model_config.hidden_size)) != (
        int(profile["expected_layers"]),
        hidden,
    ):
        raise RuntimeError("loaded architecture does not match the pinned model profile")
    if not 0 <= args.layer < len(blocks):
        raise RuntimeError("selected layer is outside the candidate model")
    device = next(model.parameters()).device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise RuntimeError("tokenizer has neither a pad nor EOS token")
    targets = torch.empty((args.n_contexts, hidden), dtype=torch.bfloat16)
    prompt_bins = torch.empty((args.n_contexts, 32, hidden), dtype=torch.bfloat16)
    x_last = torch.empty((args.n_contexts, hidden), dtype=torch.float32)
    x_mean = torch.empty_like(x_last)
    prompt_tokens = [None] * args.n_contexts if args.save_prompt_tokens else None
    answer_lengths = torch.empty(args.n_contexts, dtype=torch.int64)
    completed = 0
    for batch_index, indices in enumerate(batches):
        maximum = max(lengths[index] for index in indices)
        input_ids = torch.full(
            (len(indices), maximum), int(pad_id), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for position, index in enumerate(indices):
            length = lengths[index]
            input_ids[position, :length] = torch.tensor(token_ids[index], device=device)
            attention_mask[position, :length] = 1
        captured = base.capture_block_outputs(
            model, blocks, input_ids, attention_mask, [args.layer]
        )[args.layer]
        for position, index in enumerate(indices):
            start = answer_starts[index]
            end = lengths[index]
            prompt_hidden = captured[position, :start].float()
            pooled = (
                torch.nn.functional.adaptive_avg_pool1d(prompt_hidden.T.unsqueeze(0), 32)
                .squeeze(0)
                .T
            )
            prompt_bins[index].copy_(pooled.cpu().bfloat16())
            x_last[index].copy_(prompt_hidden[-1].cpu())
            x_mean[index].copy_(prompt_hidden.mean(0).cpu())
            if prompt_tokens is not None:
                prompt_tokens[index] = captured[position, :start].cpu().bfloat16().clone()
            value = captured[position, start:end].float().mean(0)
            targets[index].copy_(value.to(device="cpu", dtype=torch.bfloat16))
            answer_lengths[index] = end - start
        completed += len(indices)
        if batch_index % 25 == 0 or completed == args.n_contexts:
            print(
                f"{args.model_id}: completed {completed:,}/{args.n_contexts:,} on-policy answers "
                f"({batch_index + 1:,}/{len(batches):,} batches)",
                flush=True,
            )
        del input_ids, attention_mask, captured
    if not all(torch.isfinite(v).all() for v in (targets, prompt_bins, x_last)):
        raise RuntimeError("on-policy activations contain non-finite values")
    path = run_dir / "onpolicy.safetensors"
    save_atomic(
        path,
        {
            "y_onpolicy": targets,
            "prompt_bins": prompt_bins,
            "x_last": x_last,
            "x_mean": x_mean,
            "answer_lengths": answer_lengths,
            "source_global_indices": torch.tensor(
                [int(row["source_global_index"]) for row in prompts], dtype=torch.int64
            ),
        },
        {
            "schema_version": "1",
            "model_id": args.model_id,
            "model_revision": profile["revision"],
            "layer": str(args.layer),
            "generation_seed": str(config["generation"]["seed"]),
            "prompt_digest": prompt_digest(prompts),
            "response_digest": response_digest(rollouts),
        },
    )
    prompt_token_sha256 = None
    if prompt_tokens is not None:
        token_path = run_dir / "prompt_tokens.safetensors"
        offsets = torch.tensor([0] + [len(t) for t in prompt_tokens]).cumsum(0)
        save_atomic(
            token_path,
            {
                "tokens": torch.cat(prompt_tokens),
                "offsets": offsets,
                "source_global_indices": torch.tensor(
                    [int(r["source_global_index"]) for r in prompts]
                ),
            },
            {"prompt_digest": prompt_digest(prompts), "layer": str(args.layer)},
        )
        prompt_token_sha256 = base.sha256_file(token_path)
    verification = {
        "schema_version": 1,
        "status": "verified_complete",
        "model_id": args.model_id,
        "model_revision": profile["revision"],
        "layer": args.layer,
        "generation_seed": config["generation"]["seed"],
        "n_contexts": args.n_contexts,
        "prompt_digest": prompt_digest(prompts),
        "response_digest": response_digest(rollouts),
        "tensor_shape": list(targets.shape),
        "answer_tokens": {
            "minimum": int(answer_lengths.min()),
            "maximum": int(answer_lengths.max()),
            "mean": float(answer_lengths.float().mean()),
        },
        "activation_file": path.name,
        "prompt_tokens_sha256": prompt_token_sha256,
        "activation_sha256": base.sha256_file(path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "run_dir": str(run_dir),
        "scratch_dir": os.environ.get("SCRATCH_DIR"),
    }
    write_json_atomic(verification_path, verification)
    print(json.dumps(verification, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
