"""Collect fresh rollouts and every token's post-block hidden state."""

from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
from cam.data import activation_collection as base  # noqa: E402

SCHEMA_VERSION = 3


def requested_layers(args: argparse.Namespace, profile: dict[str, Any]) -> list[int]:
    layers = (
        list(range(int(profile["expected_layers"]))) if args.layers is None else list(args.layers)
    )
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("activation layers must be a non-empty unique list")
    if layers != sorted(layers):
        raise ValueError("activation layers must be sorted")
    if layers[0] < 0 or layers[-1] >= int(profile["expected_layers"]):
        raise ValueError(
            f"activation layers must be within [0, {int(profile['expected_layers']) - 1}]"
        )
    return layers


def full_shard_stem(start: int, end: int) -> str:
    return f"full_{start:06d}_{end - 1:06d}"


def durable_shard_paths(run_dir: Path, start: int, end: int) -> dict[str, Path]:
    stem = full_shard_stem(start, end)
    return {
        "full": run_dir / "activations" / "full_tokens" / f"{stem}.safetensors",
        "context": (
            run_dir / "activations" / "context" / f"context_{start:06d}_{end - 1:06d}.safetensors"
        ),
        "answer": (
            run_dir / "activations" / "answer" / f"answer_{start:06d}_{end - 1:06d}.safetensors"
        ),
        "spans": run_dir / "data" / "activation_spans" / f"{stem}.jsonl",
        "shard_manifest": (run_dir / "activations" / "shard_manifests" / f"{stem}.json"),
    }


def publish_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
    )
    if temporary.exists():
        temporary.unlink()
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=32 * 1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if temporary.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"short durable copy: {source} -> {temporary}")
    os.replace(temporary, destination)


def write_and_publish_json(destination: Path, value: Any, staging_dir: Path) -> None:
    source = staging_dir / destination.name
    base.json_dump_atomic(source, value)
    publish_atomic(source, destination)


def context_token_path(run_dir: Path, start: int, end: int) -> Path:
    stem = f"context_{start:06d}_{end - 1:06d}"
    return run_dir / "data" / "context_tokens" / f"{stem}.jsonl"


def prepare_context_tokens(
    run_dir: Path,
    prompts: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    chat_template_kwargs: dict[str, Any],
    chunk_size: int,
) -> tuple[list[str], list[list[int]]]:
    rendered: list[str | None] = [None] * len(prompts)
    token_ids: list[list[int] | None] = [None] * len(prompts)
    for start in range(0, len(prompts), chunk_size):
        end = min(start + chunk_size, len(prompts))
        path = context_token_path(run_dir, start, end)
        if path.exists():
            rows = base.jsonl_load(path)
            if [row["index"] for row in rows] != list(range(start, end)):
                raise RuntimeError(f"invalid context-token checkpoint: {path}")
        else:
            rows = []
            for index in range(start, end):
                text, ids = base.render_prompt(
                    tokenizer,
                    prompts[index]["prompt"],
                    chat_template_kwargs,
                )
                rows.append(
                    {
                        "index": index,
                        "prompt_sha256": prompts[index]["prompt_sha256"],
                        "rendered_prompt": text,
                        "rendered_prompt_sha256": base.sha256_text(text),
                        "prompt_token_ids": ids,
                        "prompt_length": len(ids),
                        "context_last_position": len(ids) - 1,
                    }
                )
            base.jsonl_dump_atomic(path, rows)
        for row in rows:
            rendered[row["index"]] = row["rendered_prompt"]
            token_ids[row["index"]] = row["prompt_token_ids"]
    if any(value is None for value in rendered) or any(value is None for value in token_ids):
        raise RuntimeError("context-token coverage is incomplete")
    return (
        [value for value in rendered if value is not None],
        [value for value in token_ids if value is not None],
    )


def load_rollouts(run_dir: Path, n_contexts: int) -> list[dict[str, Any]]:
    rows = [
        row
        for path in sorted((run_dir / "data" / "rollouts").glob("*.jsonl"))
        for row in base.jsonl_load(path)
    ]
    if len(rows) != n_contexts or [row["index"] for row in rows] != list(range(n_contexts)):
        raise RuntimeError("rollout checkpoint coverage is incomplete")
    return rows


def code_hashes(directory: Path) -> dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        project / "pyproject.toml",
        project / "uv.lock",
        project / "cam/data/activation_collection.py",
    ]
    return {
        str(path.relative_to(project)): base.sha256_file(path) for path in paths if path.is_file()
    }


def archive_verified_prefix(args: argparse.Namespace, n_contexts: int) -> dict[str, Any]:
    """Archive small verification metadata before extending a durable run in place."""
    durable_run_dir = args.durable_run_dir
    snapshot_dir = durable_run_dir / "snapshots" / f"n{n_contexts}"
    snapshot_record_path = snapshot_dir / "snapshot.json"
    metadata_names = (
        "run_config.json",
        "generation_complete.json",
        "collection_complete.json",
        "artifact_manifest.json",
        "verification.json",
    )
    if snapshot_record_path.exists():
        snapshot = json.loads(snapshot_record_path.read_text(encoding="utf-8"))
        if snapshot.get("n_contexts") != n_contexts:
            raise RuntimeError(f"invalid extension snapshot: {snapshot_record_path}")
    else:
        verification_path = durable_run_dir / "verification.json"
        if not verification_path.exists():
            raise RuntimeError(
                f"cannot extend an unverified run without a snapshot: {durable_run_dir}"
            )
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if (
            verification.get("status") != "verified_complete"
            or verification.get("n_contexts") != n_contexts
        ):
            raise RuntimeError("the existing run is not a verified prefix")
        temporary = snapshot_dir.with_name(
            f".{snapshot_dir.name}.tmp-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name in metadata_names:
            source = durable_run_dir / name
            if not source.is_file():
                raise RuntimeError(f"missing extension metadata: {source}")
            destination = temporary / name
            shutil.copy2(source, destination)
            hashes[name] = base.sha256_file(destination)
        snapshot = {
            "schema_version": 1,
            "created_at": base.utc_now(),
            "status": "verified_prefix_snapshot",
            "n_contexts": n_contexts,
            "source_run": str(durable_run_dir),
            "metadata_sha256": hashes,
            "note": "Activation shards remain immutable in the parent run directory.",
        }
        base.json_dump_atomic(temporary / "snapshot.json", snapshot)
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, snapshot_dir)

    for run_dir in {args.run_dir, durable_run_dir}:
        for name in metadata_names[1:]:
            (run_dir / name).unlink(missing_ok=True)
    return snapshot


def ensure_config(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    profile: dict[str, Any],
) -> dict[str, Any]:
    path = args.run_dir / "run_config.json"
    layers = requested_layers(args, profile)
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "model_id": args.model_id,
            "layers": layers,
            "activation_shard_size": args.activation_shard_size,
            "activation_storage_dtype": "bfloat16",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"existing config mismatch for {key}: {config.get(key)!r} != {value!r}"
                )
        generation = config["generation"]
        for key, value in {
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "chunk_size": args.generation_chunk_size,
        }.items():
            if generation.get(key) != value:
                raise RuntimeError(
                    f"existing generation config mismatch for {key}: "
                    f"{generation.get(key)!r} != {value!r}"
                )
        existing_n_contexts = int(config.get("n_contexts_requested", 0))
        if existing_n_contexts != args.n_contexts:
            if not args.allow_extend or not 0 < existing_n_contexts < args.n_contexts:
                raise RuntimeError(
                    "existing config mismatch for n_contexts_requested: "
                    f"{existing_n_contexts!r} != {args.n_contexts!r}"
                )
            snapshot = archive_verified_prefix(args, existing_n_contexts)
            config["n_contexts_requested"] = args.n_contexts
            config["durable_run_dir"] = str(args.durable_run_dir)
            config.setdefault("extensions", []).append(
                {
                    "extended_at": base.utc_now(),
                    "from_n_contexts": existing_n_contexts,
                    "to_n_contexts": args.n_contexts,
                    "verified_prefix_snapshot": str(
                        args.durable_run_dir / "snapshots" / f"n{existing_n_contexts}"
                    ),
                    "source_manifest_sha256": snapshot["metadata_sha256"]["artifact_manifest.json"],
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                }
            )
            project_dir = Path(__file__).resolve().parents[2]
            config["project_git"] = base.safe_git_revision(project_dir)
            config["code_sha256"] = code_hashes(project_dir)
            base.json_dump_atomic(path, config)
            if path.resolve() != (args.durable_run_dir / "run_config.json").resolve():
                base.json_dump_atomic(args.durable_run_dir / "run_config.json", config)
        return config

    raw_template = tokenizer.chat_template
    if isinstance(raw_template, dict):
        raw_template = json.dumps(raw_template, ensure_ascii=False, sort_keys=True)
    if not isinstance(raw_template, str):
        raise RuntimeError(f"{args.model_id} does not expose a usable chat template")
    project_dir = Path(__file__).resolve().parents[2]
    expected_layers = int(profile["expected_layers"])
    prompt_source = str(args.prompts_file) if args.prompts_file is not None else base.DATASET_ID
    config = {
        "schema_version": SCHEMA_VERSION,
        "started_at": base.utc_now(),
        "run_dir": str(args.run_dir),
        "durable_run_dir": str(args.durable_run_dir),
        "purpose": (
            "fresh 4096-token context-to-answer-map collection with every token's "
            "raw post-block hidden state"
        ),
        "upstream_spec": base.UPSTREAM_SPEC,
        "upstream_commit": base.UPSTREAM_COMMIT,
        "upstream_rollout_artifacts_used": False,
        "upstream_activation_artifacts_used": False,
        "data_inputs": [prompt_source],
        "model_id": args.model_id,
        "model_revision": profile["revision"],
        "model_loader": profile["loader"],
        "model_architecture_expected": {
            "layers": expected_layers,
            "hidden_size": profile["expected_hidden"],
        },
        "dataset_id": base.DATASET_ID,
        "dataset_revision": base.DATASET_REVISION,
        "n_contexts_requested": args.n_contexts,
        "layers": layers,
        "source_prompt_range": {
            "start": args.prompt_index_start,
            "end_exclusive": args.prompt_index_start + args.n_contexts,
            "prompts_file": str(args.prompts_file) if args.prompts_file else None,
        },
        "hidden_size": profile["expected_hidden"],
        "layer_convention": (
            f"raw decoder block outputs at configured layers {layers}; block "
            f"{expected_layers - 1} is before the model's final normalization"
        ),
        "activation_layout": {
            "full_shard_key": "hidden_<six-digit context index>",
            "full_shard_shape": (
                "[layers, prompt_plus_generated_plus_template_suffix_tokens, hidden]"
            ),
            "token_axis": (
                "prompt token IDs, exact vLLM generated token IDs, then the "
                "non-overlapping canonical assistant suffix for stopped answers"
            ),
            "context_span": "[0, prompt_length)",
            "generated_span": "[prompt_length, generated_span_end_exclusive)",
            "assistant_suffix_span": (
                "[generated_span_end_exclusive, full_length); empty for length-terminated answers"
            ),
            "original_experiment_answer_span": "[prompt_length, full_length)",
            "summary_sidecars": ["cx_last", "cx_mean", "v_answer_mean"],
        },
        "context_summaries": ["last prompt token", "mean over prompt tokens"],
        "answer_summary": (
            "mean over exact generated tokens plus the non-overlapping canonical "
            "assistant suffix, matching the issue-779 target"
        ),
        "chat_template": {
            "sha256": base.sha256_text(raw_template),
            "kwargs": profile["chat_template_kwargs"],
            "thinking_mode": profile["thinking_mode"],
        },
        "generation": {
            "backend": "vllm",
            "n": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "chunk_size": args.generation_chunk_size,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "standard_chat_template": True,
            "pretokenized_prompt_input": True,
            "vllm_language_model_only": profile["vllm_language_model_only"],
            "vllm_attention_backend": profile["vllm_attention_backend"],
        },
        "activation_storage_dtype": "bfloat16",
        "summary_storage_dtype": "float32",
        "activation_shard_size": args.activation_shard_size,
        "kv_cache_storage": (
            "not persisted; exact prefix token IDs are retained so a Transformers or "
            "vLLM prefill can rebuild KV state and continue with caching"
        ),
        "environment": base.record_environment(args.run_dir),
        "project_git": base.safe_git_revision(project_dir),
        "code_sha256": code_hashes(project_dir),
    }
    base.json_dump_atomic(path, config)
    return config


def generate_stage(args: argparse.Namespace) -> None:
    profile = base.MODEL_PROFILES[args.model_id]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=profile["revision"],
    )
    ensure_config(args, tokenizer, profile)
    prompts = base.collect_or_load_prompts(
        args.run_dir,
        n_contexts=args.n_contexts,
        dataset_revision=base.DATASET_REVISION,
        prompts_file=args.prompts_file,
        allow_extend=args.allow_extend,
    )
    rendered, token_ids = prepare_context_tokens(
        args.run_dir,
        prompts,
        tokenizer,
        profile["chat_template_kwargs"],
        args.generation_chunk_size,
    )
    llm, rows = base.generate_rollouts(
        args.run_dir,
        prompts,
        rendered,
        token_ids,
        model_id=args.model_id,
        model_revision=profile["revision"],
        thinking_mode=profile["thinking_mode"],
        vllm_language_model_only=profile["vllm_language_model_only"],
        vllm_attention_backend=profile["vllm_attention_backend"],
        generation_chunk_size=args.generation_chunk_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    del llm
    torch.cuda.empty_cache()
    lengths = [row["generated_token_count"] for row in rows]
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "generation_complete",
        "completed_at": base.utc_now(),
        "n_contexts": len(rows),
        "total_generated_tokens": sum(lengths),
        "maximum_generated_tokens": max(lengths),
        "length_terminated": sum(row["finish_reason"] == "length" for row in rows),
        "finish_reasons": {
            reason: sum(row["finish_reason"] == reason for row in rows)
            for reason in sorted({row["finish_reason"] for row in rows})
        },
    }
    base.json_dump_atomic(args.run_dir / "generation_complete.json", completion)


def load_model(args: argparse.Namespace, profile: dict[str, Any]) -> torch.nn.Module:
    loader = AutoModelForCausalLM if profile["loader"] == "causal_lm" else AutoModelForMultimodalLM
    model = loader.from_pretrained(
        args.model_id,
        revision=profile["revision"],
        dtype=torch.bfloat16,
        device_map={"": torch.device("cuda:0")},
    )
    model.eval()
    return model


def valid_completed_shard(paths: dict[str, Path], start: int, end: int) -> bool:
    manifest_path = paths["shard_manifest"]
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("start") != start or manifest.get("end_exclusive") != end:
        return False
    for record in manifest.get("files", []):
        path = manifest_path.parents[2] / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            return False
    return len(manifest.get("files", [])) == 4 and manifest.get("finite") is True


def remove_stale_temporary_files(run_dir: Path) -> None:
    """Remove unpublished files left by a terminated atomic durable copy."""
    for path in run_dir.rglob(".*.tmp-*"):
        if path.is_file():
            path.unlink()


def capture_activation_stage(args: argparse.Namespace) -> None:
    if not (args.run_dir / "generation_complete.json").exists():
        raise RuntimeError("generation stage is incomplete")
    profile = base.MODEL_PROFILES[args.model_id]
    prompts = base.jsonl_load(args.run_dir / "data" / "prompts.jsonl")
    context_rows = [
        row
        for path in sorted((args.run_dir / "data" / "context_tokens").glob("*.jsonl"))
        for row in base.jsonl_load(path)
    ]
    rollouts = load_rollouts(args.run_dir, args.n_contexts)
    if len(context_rows) != args.n_contexts:
        raise RuntimeError("context-token row count mismatch")
    remove_stale_temporary_files(args.durable_run_dir)

    model = load_model(args, profile)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=profile["revision"],
    )
    suffix_text, suffix_ids = base.assistant_suffix(
        tokenizer,
        profile["chat_template_kwargs"],
    )
    blocks = base.resolve_decoder_blocks(model)
    layers = requested_layers(args, profile)
    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    if (len(blocks), hidden_size) != (
        profile["expected_layers"],
        profile["expected_hidden"],
    ):
        raise RuntimeError("loaded model architecture does not match the pinned profile")
    device = next(model.parameters()).device
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, args.n_contexts, args.activation_shard_size):
        end = min(start + args.activation_shard_size, args.n_contexts)
        durable_paths = durable_shard_paths(args.durable_run_dir, start, end)
        if valid_completed_shard(durable_paths, start, end):
            print(f"activation shard [{start}, {end}) exists; skipping", flush=True)
            continue
        shard_started = time.monotonic()
        shard_stage = args.staging_dir / full_shard_stem(start, end)
        if shard_stage.exists():
            shutil.rmtree(shard_stage)
        shard_stage.mkdir(parents=True)
        full_tensors: dict[str, torch.Tensor] = {}
        context_last: list[torch.Tensor] = []
        context_mean: list[torch.Tensor] = []
        answer_mean: list[torch.Tensor] = []
        span_rows = []
        finite = True
        total_tokens = 0
        for index in range(start, end):
            prompt_ids = context_rows[index]["prompt_token_ids"]
            generated_ids = rollouts[index]["generated_token_ids"]
            overlap = (
                base.suffix_overlap(generated_ids, suffix_ids)
                if rollouts[index]["finish_reason"] == "stop"
                else 0
            )
            appended_suffix_ids = (
                suffix_ids[overlap:] if rollouts[index]["finish_reason"] == "stop" else []
            )
            full_ids = [*prompt_ids, *generated_ids, *appended_suffix_ids]
            if not generated_ids:
                raise RuntimeError(f"empty generated token span at row {index}")
            if len(full_ids) > args.max_model_len:
                raise RuntimeError(
                    f"row {index} has {len(full_ids)} tokens, above {args.max_model_len}"
                )
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            captured = base.capture_block_outputs(
                model,
                blocks,
                input_ids,
                attention_mask,
                layers,
            )
            hidden = torch.stack(
                [captured[layer][0].detach().to("cpu", dtype=torch.bfloat16) for layer in layers],
                dim=0,
            ).contiguous()
            row_finite = bool(torch.isfinite(hidden).all())
            finite = finite and row_finite
            if not row_finite:
                raise RuntimeError(f"non-finite activation at row {index}")
            prompt_length = len(prompt_ids)
            generated_end = prompt_length + len(generated_ids)
            context = hidden[:, :prompt_length, :].float()
            answer = hidden[:, prompt_length:, :].float()
            context_last.append(context[:, -1, :].contiguous())
            context_mean.append(context.mean(dim=1).contiguous())
            answer_mean.append(answer.mean(dim=1).contiguous())
            full_tensors[f"hidden_{index:06d}"] = hidden
            full_tensors[f"token_ids_{index:06d}"] = torch.tensor(
                full_ids,
                dtype=torch.int64,
            )
            span_rows.append(
                {
                    "index": index,
                    "prompt_sha256": prompts[index]["prompt_sha256"],
                    "response_sha256": rollouts[index]["response_sha256"],
                    "prompt_length": prompt_length,
                    "generated_length": len(generated_ids),
                    "generated_span_start": prompt_length,
                    "generated_span_end_exclusive": generated_end,
                    "canonical_assistant_suffix": suffix_text,
                    "canonical_assistant_suffix_token_ids": suffix_ids,
                    "generated_suffix_overlap": overlap,
                    "appended_suffix_token_ids": appended_suffix_ids,
                    "appended_suffix_length": len(appended_suffix_ids),
                    "assistant_suffix_span_start": generated_end,
                    "assistant_suffix_span_end_exclusive": len(full_ids),
                    "full_length": len(full_ids),
                    "context_span_start": 0,
                    "context_span_end_exclusive": prompt_length,
                    "answer_span_start": prompt_length,
                    "answer_span_end_exclusive": len(full_ids),
                    "answer_span_includes": (
                        "exact generated tokens plus non-overlapping canonical assistant suffix"
                    ),
                    "finish_reason": rollouts[index]["finish_reason"],
                    "activation_key": f"hidden_{index:06d}",
                    "token_ids_key": f"token_ids_{index:06d}",
                    "finite": row_finite,
                }
            )
            total_tokens += len(full_ids)
            del captured, hidden, context, answer, input_ids, attention_mask

        local_paths = {
            "full": shard_stage / durable_paths["full"].name,
            "context": shard_stage / durable_paths["context"].name,
            "answer": shard_stage / durable_paths["answer"].name,
            "spans": shard_stage / durable_paths["spans"].name,
        }
        base.save_tensor_shard(
            local_paths["full"],
            full_tensors,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "full_token_post_block_hidden_states",
                "start": start,
                "end_exclusive": end,
                "layers": layers,
                "layout": "each hidden_<index> is [layers, tokens, hidden]",
                "token_axis": (
                    "prompt tokens, exact generated tokens, then non-overlapping "
                    "canonical assistant suffix for stopped answers"
                ),
                "dtype": "bfloat16",
                "model_id": args.model_id,
                "model_revision": profile["revision"],
            },
        )
        base.save_tensor_shard(
            local_paths["context"],
            {
                "cx_last": torch.stack(context_last),
                "cx_mean": torch.stack(context_mean),
            },
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "derived_context_summaries",
                "start": start,
                "end_exclusive": end,
                "layers": layers,
                "dtype": "float32",
                "source": durable_paths["full"].relative_to(args.durable_run_dir).as_posix(),
            },
        )
        base.save_tensor_shard(
            local_paths["answer"],
            {"v_answer_mean": torch.stack(answer_mean)},
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "derived_answer_summary",
                "start": start,
                "end_exclusive": end,
                "layers": layers,
                "dtype": "float32",
                "answer_span": (
                    "exact generated tokens plus non-overlapping canonical assistant "
                    "suffix, matching issue 779"
                ),
                "source": durable_paths["full"].relative_to(args.durable_run_dir).as_posix(),
            },
        )
        base.jsonl_dump_atomic(local_paths["spans"], span_rows)

        records = []
        for label, local_path in local_paths.items():
            destination = durable_paths[label]
            sha256 = base.sha256_file(local_path)
            size = local_path.stat().st_size
            publish_atomic(local_path, destination)
            if destination.stat().st_size != size:
                raise RuntimeError(f"durable size mismatch after publishing {destination}")
            records.append(
                {
                    "path": destination.relative_to(args.durable_run_dir).as_posix(),
                    "bytes": size,
                    "sha256": sha256,
                }
            )
        shard_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": base.utc_now(),
            "start": start,
            "end_exclusive": end,
            "n_rows": end - start,
            "total_tokens": total_tokens,
            "finite": finite,
            "files": records,
        }
        write_and_publish_json(
            durable_paths["shard_manifest"],
            shard_manifest,
            shard_stage,
        )
        shutil.rmtree(shard_stage)
        print(
            f"captured activation shard [{start}, {end}) tokens={total_tokens} "
            f"seconds={time.monotonic() - shard_started:.1f}",
            flush=True,
        )

    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "collection_complete_pending_verification",
        "completed_at": base.utc_now(),
        "n_contexts": args.n_contexts,
        "run_config_sha256": base.sha256_file(args.durable_run_dir / "run_config.json"),
        "activation_shards": (args.n_contexts + args.activation_shard_size - 1)
        // args.activation_shard_size,
        "activation_storage_dtype": "bfloat16",
    }
    base.json_dump_atomic(args.durable_run_dir / "collection_complete.json", completion)
    del model
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "activations"), required=True)
    parser.add_argument("--model-id", choices=sorted(base.MODEL_PROFILES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--durable-run-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--prompt-index-start", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--n-contexts", type=int, default=11400)
    parser.add_argument("--generation-chunk-size", type=int, default=100)
    parser.add_argument("--activation-shard-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--allow-extend", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.durable_run_dir = args.durable_run_dir.resolve()
    if args.prompts_file is not None:
        args.prompts_file = args.prompts_file.resolve()
    if args.staging_dir is None:
        args.staging_dir = args.run_dir / "activation_stage"
    args.staging_dir = args.staging_dir.resolve()
    if args.n_contexts <= 0 or args.activation_shard_size <= 0 or args.prompt_index_start < 0:
        parser.error("context count and activation shard size must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run through Slurm")
    torch.set_float32_matmul_precision("high")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "generate":
        generate_stage(args)
    else:
        capture_activation_stage(args)
    print(
        json.dumps(
            {
                "status": "stage_complete",
                "stage": args.stage,
                "model_id": args.model_id,
                "n_contexts": args.n_contexts,
                "host": platform.node(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
