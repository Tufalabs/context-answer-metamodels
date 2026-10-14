"""Continue exact saved token prefixes and recapture full-answer activations."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import continuation_budget, file_hash, read_rows, token_hash, write_json, write_rows


def generate(*, task, rows, output):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from cam.data import activation_collection as base

    config = task["source_config"]
    model = config["model_id"]
    profile = base.MODEL_PROFILES[model]
    revision = config["model_revision"]
    assert revision == profile["revision"]
    generation = config["generation"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model, revision=revision)
    engine = LLM(model=model, revision=revision, tokenizer_revision=revision,
                 dtype="bfloat16", trust_remote_code=True,
                 language_model_only=profile["vllm_language_model_only"],
                 attention_backend=profile["vllm_attention_backend"],
                 gpu_memory_utilization=.85, max_model_len=generation["max_model_len"],
                 max_num_seqs=8, seed=20261013)
    checkpoint = output / "continued_rows.jsonl"
    saved = read_rows(path=checkpoint) if checkpoint.exists() else []
    completed = {row["identity"]: row for row in saved}
    expected = {row["identity"]: row for row in rows}
    assert len(saved) == len(completed) and completed.keys() <= expected.keys()
    for identity, row in completed.items():
        original = expected[identity]
        prefix = original["rollout"]["generated_token_ids"]
        assert row["prefix_sha256"] == original["prefix_sha256"]
        assert row["rollout"]["generated_token_ids"][:len(prefix)] == prefix
        assert row["rollout"]["continuation_seed"] == original["continuation_seed"]
    for start in range(0, len(rows), 8):
        batch = [(index, row) for index, row in enumerate(rows[start:start + 8], start)
                 if row["identity"] not in completed]
        if not batch:
            continue
        prompts, parameters = [], []
        for index, row in batch:
            original = row["rollout"]
            budget = continuation_budget(rollout=original, prompt_tokens=row["context"]["prompt_token_ids"],
                                         source_cap=generation["max_new_tokens"], target_cap=task["target_cap"],
                                         context_limit=generation["max_model_len"])
            assert budget > 0 and token_hash(tokens=original["generated_token_ids"]) == row["prefix_sha256"]
            prompts.append({"prompt_token_ids": row["context"]["prompt_token_ids"] + original["generated_token_ids"]})
            parameters.append(SamplingParams(n=1, temperature=generation["temperature"], top_p=generation["top_p"],
                                             max_tokens=budget, seed=row["continuation_seed"]))
        outputs = engine.generate(prompts=prompts, sampling_params=parameters, use_tqdm=False)
        for (index, row), generated in zip(batch, outputs, strict=True):
            completion = generated.outputs[0]
            tokens = row["rollout"]["generated_token_ids"] + list(completion.token_ids)
            assert len(tokens) <= task["target_cap"] and completion.finish_reason in {"stop", "length"}
            if completion.finish_reason == "length":
                assert len(tokens) == task["target_cap"]
            response = tokenizer.decode(token_ids=tokens, skip_special_tokens=True)
            rollout = {**row["rollout"], "index": index, "generated_token_ids": tokens,
                       "generated_token_count": len(tokens), "response": response,
                       "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                       "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                       "cumulative_logprob": None, "generation_backend": "vllm_saved_prefix_continuation",
                       "continuation_seed": row["continuation_seed"], "source_identity": row["identity"],
                       "continuation_generated_at": datetime.now(UTC).isoformat(),
                       "continuation_prefix_tokens": len(row["rollout"]["generated_token_ids"]),
                       "original_response_sha256": row["rollout"]["response_sha256"]}
            completed[row["identity"]] = {"identity": row["identity"], "prefix_sha256": row["prefix_sha256"], "rollout": rollout}
        write_rows(path=checkpoint, rows=list(completed.values()))
        print(f"Continued {len(completed)}/{len(rows)}", flush=True)
    write_rows(path=output / "data/prompts.jsonl", rows=[{**row["prompt"], "index": index} for index, row in enumerate(rows)])
    write_rows(path=output / "data/context_tokens/context_000000.jsonl", rows=[{**row["context"], "index": index} for index, row in enumerate(rows)])
    write_rows(path=output / "data/rollouts/rollouts_000000.jsonl", rows=[completed[row["identity"]]["rollout"] for row in rows])
    layer = 19 if model == "Qwen/Qwen2.5-7B-Instruct" else 18
    assert layer in config["layers"]
    write_json(path=output / "run_config.json", value={**config, "n_contexts_requested": len(rows),
               "started_at": datetime.now(UTC).isoformat(), "run_dir": str(output), "durable_run_dir": str(output),
               "purpose": "Extend saved capped answers to a total 4,096-token budget and recapture activations",
               "source_prompt_range": {"start": 0, "end_exclusive": len(rows), "prompts_file": str(output / "data/prompts.jsonl")},
               "source_indices": [row["source_index"] for row in rows],
               "environment": base.record_environment(output), "project_git": None,
               "code_sha256": {"extension": file_hash(path=Path(__file__)), "activation_helpers": file_hash(path=Path(base.__file__))},
               "layers": [layer], "hidden_size": profile["expected_hidden"], "activation_storage_dtype": "bfloat16",
               "summary_storage_dtype": "float32", "activation_shard_size": 4,
               "upstream_rollout_artifacts_used": False,
               "generation": {**generation, "max_new_tokens": task["target_cap"], "seed": 20261013,
                              "original_seed": generation["seed"], "sampling_seed_source": "per-row continuation_seed",
                              "max_num_seqs": 8, "gpu_memory_utilization": .85, "chunk_size": 8},
               "continuation_protocol": "Exact saved prefixes; independent per-row continuation seeds. Original sampling policy and prompt tokenization retained.",
               "extension_task": task})
    write_json(path=output / "generation_complete.json", value={"status": "generation_complete", "n_contexts": len(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "generate"], default="all")
    args = parser.parse_args()
    task = json.loads((args.prepared / "tasks.json").read_text())[args.task]
    assert task["task"] == args.task
    input_path = args.prepared / task["input"]
    assert file_hash(path=input_path) == task["sha256"]
    rows = read_rows(path=input_path)
    assert len(rows) == task["rows"] and rows
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "task_identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == task
    else:
        write_json(path=identity_path, value=task)
    if args.stage == "generate":
        generate(task=task, rows=rows, output=args.output)
        return
    if not (args.output / "generation_complete.json").exists():
        subprocess.run(args=[sys.executable, "-m", __spec__.name, "--prepared", str(args.prepared),
                             "--task", str(args.task), "--output", str(args.output), "--stage", "generate"], check=True)
    config = json.loads((args.output / "run_config.json").read_text())
    assert config["extension_task"] == task
    subprocess.run(args=[sys.executable, "-m", "cam.data.rollouts", "--stage", "activations",
                         "--run-dir", str(args.output), "--durable-run-dir", str(args.output),
                         "--model-id", config["model_id"], "--layers", str(config["layers"][0]),
                         "--n-contexts", str(len(rows)), "--activation-shard-size", "4",
                         "--max-model-len", str(config["generation"]["max_model_len"])], check=True)
    subprocess.run(args=[sys.executable, "-m", "cam.data.verify_rollouts", "--run-dir", str(args.output)], check=True)
    write_json(path=args.output / "task_complete.json", value={"status": "verified_complete", "task": args.task,
               "input_sha256": task["sha256"], "verification_sha256": file_hash(path=args.output / "verification.json"),
               "artifact_manifest_sha256": file_hash(path=args.output / "artifact_manifest.json"),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID")})


if __name__ == "__main__":
    main()
