"""Shared prompt, rollout, and activation collection utilities.

Layer L is the raw output of decoder block L, before final normalization.
"""

from __future__ import annotations
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

DATASET_ID = "lmsys/lmsys-chat-1m"
DATASET_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
UPSTREAM_COMMIT = "0bd3db3eca3736a10afc3cd4058661bdd4d4be01"
UPSTREAM_SPEC = (
    "https://github.com/superkaiba/explore-persona-space/blob/"
    f"{UPSTREAM_COMMIT}/tasks/awaiting_promotion/779/artifacts/"
    "issue779_writeup_A_v2_single_context_answer_map.md"
)
EXPECTED_FIRST_PROMPT = (
    "how can identity protection services help protect me against identity theft"
)


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "loader": "causal_lm",
        "expected_layers": 28,
        "expected_hidden": 3584,
        "chat_template_kwargs": {},
        "thinking_mode": "not_supported",
        "vllm_language_model_only": False,
        "vllm_attention_backend": "FLASH_ATTN",
    },
    "Qwen/Qwen3.5-9B": {
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "loader": "multimodal_lm",
        "expected_layers": 32,
        "expected_hidden": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_mode": "disabled_via_chat_template",
        "vllm_language_model_only": True,
        "vllm_attention_backend": "FLASH_ATTN",
    },
    "google/gemma-4-12B-it": {
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "loader": "multimodal_lm",
        "expected_layers": 48,
        "expected_hidden": 3840,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_mode": "disabled_via_chat_template",
        "vllm_language_model_only": True,
        "vllm_attention_backend": "TRITON_ATTN",
    },
}
LOG = logging.getLogger("collect")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_dump_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def jsonl_dump_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def jsonl_load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_git_revision(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result[name] = dist.version
    return dict(sorted(result.items(), key=lambda item: item[0].lower()))


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def record_environment(run_dir: Path) -> dict[str, Any]:
    environment_dir = run_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    packages = package_versions()
    json_dump_atomic(environment_dir / "packages.json", packages)
    (environment_dir / "pip_freeze.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in packages.items()),
        encoding="utf-8",
    )
    (environment_dir / "nvidia_smi.txt").write_text(
        command_output(["nvidia-smi", "-q"]) + "\n", encoding="utf-8"
    )
    slurm_names = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
    ]
    slurm = {name: os.environ.get(name) for name in slurm_names}
    json_dump_atomic(environment_dir / "slurm.json", slurm)
    path_names = [
        "SCRATCH_DIR",
        "UV_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "VLLM_USE_FLASHINFER_SAMPLER",
    ]
    runtime_paths = {name: os.environ.get(name) for name in path_names}
    json_dump_atomic(environment_dir / "runtime_paths.json", runtime_paths)
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "runtime_paths": runtime_paths,
        "gpu": gpu,
        "slurm": slurm,
    }


def first_prompt_from_row(row: dict[str, Any]) -> str | None:
    """Match issue779_collect.load_train_contexts exactly for the LMSYS source."""
    value = row.get("conversation")
    if isinstance(value, list) and value and isinstance(value[0], dict):
        prompt = value[0].get("content") or value[0].get("value")
    elif isinstance(value, str):
        prompt = value
    else:
        prompt = None
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    return prompt.strip()


def collect_or_load_prompts(
    run_dir: Path,
    *,
    n_contexts: int,
    dataset_revision: str,
    prompts_file: Path | None = None,
    allow_extend: bool = False,
) -> list[dict[str, Any]]:
    path = run_dir / "data" / "prompts.jsonl"
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        existing_rows = jsonl_load(path)
        if [row["index"] for row in existing_rows] != list(range(len(existing_rows))):
            raise RuntimeError(f"existing {path} has non-contiguous indices")
        if len(existing_rows) == n_contexts:
            LOG.info("Loaded %d checkpointed prompts from %s", len(existing_rows), path)
            return existing_rows
        if len(existing_rows) > n_contexts or not allow_extend:
            raise RuntimeError(
                f"existing {path} has {len(existing_rows)} rows, expected {n_contexts}"
            )
        LOG.info(
            "Extending %s from %d to %d prompts",
            path,
            len(existing_rows),
            n_contexts,
        )

    if prompts_file is not None:
        source_rows = jsonl_load(prompts_file)
        if len(source_rows) < n_contexts:
            raise RuntimeError(f"prompt file has {len(source_rows)} rows, expected {n_contexts}")
        rows = []
        for index, source_row in enumerate(source_rows[:n_contexts]):
            prompt = source_row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise RuntimeError(f"prompt file row {index} has no non-empty prompt")
            rows.append(
                {
                    **source_row,
                    "index": index,
                    "prompt": prompt.strip(),
                    "prompt_sha256": sha256_text(prompt.strip()),
                }
            )
        jsonl_dump_atomic(path, rows)
        LOG.info("Loaded %d prompts from %s", len(rows), prompts_file)
        return rows

    LOG.info("Streaming %d prompts from %s at %s", n_contexts, DATASET_ID, dataset_revision)
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )
    rows: list[dict[str, Any]] = []
    for dataset_row_index, source_row in enumerate(dataset):
        prompt = first_prompt_from_row(source_row)
        if prompt is None:
            continue
        candidate = {
            "index": len(rows),
            "dataset_row_index": dataset_row_index,
            "dataset_id": DATASET_ID,
            "dataset_revision": dataset_revision,
            "conversation_id": source_row.get("conversation_id"),
            "source_model": source_row.get("model"),
            "language": source_row.get("language"),
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
        }
        if len(rows) < len(existing_rows):
            existing = existing_rows[len(rows)]
            if existing != candidate:
                raise RuntimeError(
                    "pinned dataset no longer matches the existing prompt prefix at "
                    f"index {len(rows)}"
                )
            rows.append(existing)
        else:
            rows.append(candidate)
        if len(rows) == n_contexts:
            break
    if len(rows) != n_contexts:
        raise RuntimeError(f"dataset yielded only {len(rows)} valid prompts, expected {n_contexts}")
    if rows[0]["prompt"] != EXPECTED_FIRST_PROMPT:
        raise RuntimeError(
            "LMSYS order/content no longer matches the referenced experiment: "
            f"first prompt was {rows[0]['prompt']!r}"
        )
    jsonl_dump_atomic(path, rows)
    LOG.info("Wrote %d source prompts to %s", len(rows), path)
    return rows


def _unwrap(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def resolve_decoder_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = [
        ("model.language_model.layers", ("model", "language_model", "layers")),
        ("model.layers", ("model", "layers")),
    ]
    for _label, parts in candidates:
        value: Any = model
        for part in parts:
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, torch.nn.ModuleList):
            return value
    tried = ", ".join(label for label, _parts in candidates)
    raise RuntimeError(f"could not locate decoder blocks; tried {tried}")


@torch.no_grad()
def capture_block_outputs(
    model: torch.nn.Module,
    blocks: torch.nn.ModuleList,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Capture raw post-block outputs, matching the issue-779 hook convention."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer] = _unwrap(output).detach()

        return hook

    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": False,
    }
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        forward_kwargs["logits_to_keep"] = 1
    try:
        model(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(layers) - set(captured)
    if missing:
        raise RuntimeError(f"forward hooks did not fire for layers {sorted(missing)}")
    return captured


def apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> Any:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **chat_template_kwargs,
    )


def render_prompt(
    tokenizer: AutoTokenizer,
    prompt: str,
    chat_template_kwargs: dict[str, Any],
) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": prompt}]
    text = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    token_ids = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    return text, token_ids


def assistant_suffix(
    tokenizer: AutoTokenizer,
    chat_template_kwargs: dict[str, Any],
) -> tuple[str, list[int]]:
    marker = "CAM_RESPONSE_SENTINEL_4F8C12D59A"
    messages = [
        {"role": "user", "content": "CAM prompt sentinel"},
        {"role": "assistant", "content": marker},
    ]
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if rendered.count(marker) != 1:
        raise RuntimeError("could not identify the canonical assistant suffix")
    suffix = rendered.split(marker, maxsplit=1)[1]
    suffix_ids = tokenizer(suffix, padding=False, add_special_tokens=False)["input_ids"]
    if not suffix_ids:
        raise RuntimeError("canonical assistant suffix tokenized to an empty sequence")
    return suffix, suffix_ids


def suffix_overlap(generated_ids: list[int], suffix_ids: list[int]) -> int:
    maximum = min(len(generated_ids), len(suffix_ids))
    for size in range(maximum, 0, -1):
        if generated_ids[-size:] == suffix_ids[:size]:
            return size
    return 0


def save_tensor_shard(
    path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        temporary,
        metadata={key: json.dumps(value, sort_keys=True) for key, value in metadata.items()},
    )
    os.replace(temporary, path)


def rollout_chunk_path(run_dir: Path, start: int, end: int) -> Path:
    return run_dir / "data" / "rollouts" / f"rollouts_{start:06d}_{end - 1:06d}.jsonl"


def completion_record(completion: Any) -> dict[str, Any]:
    stop_reason = getattr(completion, "stop_reason", None)
    if not isinstance(stop_reason, (str, int, float, bool, type(None))):
        stop_reason = str(stop_reason)
    return {
        "response": completion.text,
        "response_sha256": sha256_text(completion.text),
        "generated_token_ids": list(completion.token_ids),
        "generated_token_count": len(completion.token_ids),
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": stop_reason,
        "cumulative_logprob": getattr(completion, "cumulative_logprob", None),
    }


def generate_rollouts(
    run_dir: Path,
    prompts: list[dict[str, Any]],
    rendered_prompts: list[str],
    prompt_token_ids: list[list[int]],
    *,
    model_id: str,
    model_revision: str,
    thinking_mode: str,
    vllm_language_model_only: bool,
    vllm_attention_backend: str,
    generation_chunk_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
) -> tuple[Any, list[dict[str, Any]]]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    LOG.info("Loading independent vLLM engine for fresh rollout generation")
    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "revision": model_revision,
        "tokenizer_revision": model_revision,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "seed": seed,
        "attention_backend": vllm_attention_backend,
    }
    if vllm_language_model_only:
        llm_kwargs["language_model_only"] = True
    llm = LLM(**llm_kwargs)
    params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    all_rows: list[dict[str, Any] | None] = [None] * len(prompts)
    for start in range(0, len(prompts), generation_chunk_size):
        end = min(start + generation_chunk_size, len(prompts))
        path = rollout_chunk_path(run_dir, start, end)
        if path.exists():
            rows = jsonl_load(path)
            if [row["index"] for row in rows] != list(range(start, end)):
                raise RuntimeError(f"invalid checkpoint indices in {path}")
            for row in rows:
                expected = prompts[row["index"]]["prompt_sha256"]
                if row["prompt_sha256"] != expected:
                    raise RuntimeError(f"prompt hash mismatch in {path}")
                all_rows[row["index"]] = row
            LOG.info("Rollout chunk [%d, %d) exists; skipping", start, end)
            continue

        chunk_started = time.monotonic()
        token_prompts = [
            TokensPrompt(prompt=rendered_prompts[index], prompt_token_ids=prompt_token_ids[index])
            for index in range(start, end)
        ]
        outputs = llm.generate(token_prompts, params, use_tqdm=False)
        if len(outputs) != end - start:
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {end - start} prompts")
        rows = []
        for offset, request_output in enumerate(outputs):
            index = start + offset
            if len(request_output.outputs) != 1:
                raise RuntimeError(
                    f"vLLM returned {len(request_output.outputs)} completions at row {index}"
                )
            if list(request_output.prompt_token_ids) != prompt_token_ids[index]:
                raise RuntimeError(f"vLLM prompt token mismatch at row {index}")
            row = {
                "index": index,
                "prompt_sha256": prompts[index]["prompt_sha256"],
                "rendered_prompt_sha256": sha256_text(rendered_prompts[index]),
                "generation_prompt_token_count": len(prompt_token_ids[index]),
                "generated_at": utc_now(),
                "generation_backend": "vllm",
                "thinking_mode": thinking_mode,
                "model_id": model_id,
                "model_revision": model_revision,
                **completion_record(request_output.outputs[0]),
            }
            rows.append(row)
            all_rows[index] = row
        jsonl_dump_atomic(path, rows)
        LOG.info(
            "Generated fresh rollout chunk [%d, %d) in %.1fs",
            start,
            end,
            time.monotonic() - chunk_started,
        )
    if any(row is None for row in all_rows):
        raise RuntimeError("rollout checkpoint coverage is incomplete")
    return llm, [row for row in all_rows if row is not None]
