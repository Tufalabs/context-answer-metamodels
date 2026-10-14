"""Token budgets and immutable identities shared by extension jobs."""

import hashlib
import json
import os
from pathlib import Path


def read_rows(*, path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(*, path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(*, path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_hash(*, path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_hash(*, tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def continuation_budget(*, rollout, prompt_tokens, source_cap, target_cap, context_limit):
    tokens = rollout["generated_token_ids"]
    if len(tokens) != rollout["generated_token_count"] or len(tokens) > source_cap:
        raise ValueError("Invalid saved answer token count")
    if rollout["finish_reason"] == "stop":
        return 0
    if rollout["finish_reason"] != "length" or len(tokens) != source_cap:
        raise ValueError("Cannot interpret the saved truncation as an output-token cap")
    if target_cap <= source_cap:
        raise ValueError("The target cap must exceed the source cap")
    if len(prompt_tokens) + target_cap > context_limit:
        raise ValueError("The original context window cannot accommodate the target answer cap")
    return target_cap - len(tokens)


def continuation_seed(*, identity):
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
