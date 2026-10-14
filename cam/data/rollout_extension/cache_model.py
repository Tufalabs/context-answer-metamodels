"""Prepare a pinned node-local model once before running collectors offline."""

import argparse
import fcntl
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.tiktoken"]


def complete_snapshot(*, model, revision):
    try:
        path = Path(snapshot_download(repo_id=model, revision=revision, local_files_only=True,
                                      allow_patterns=MODEL_FILES))
    except LocalEntryNotFoundError:
        return None
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
    if not all((path / name).is_file() for name in required):
        return None
    weights = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    return path if all((path / name).is_file() for name in set(weights.values())) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.tasks.read_text())[args.task]["source_config"]
    model, revision = config["model_id"], config["model_revision"]
    lock_path = Path(os.environ["HF_HOME"]) / (model.replace("/", "--") + ".download.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = complete_snapshot(model=model, revision=revision)
        if path is None:
            snapshot_download(repo_id=model, revision=revision, max_workers=4,
                              allow_patterns=MODEL_FILES)
            path = complete_snapshot(model=model, revision=revision)
        assert path is not None, (model, revision)
    print(f"Using cached model {model} at {path}", flush=True)


if __name__ == "__main__":
    main()
