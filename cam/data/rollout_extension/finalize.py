"""Publish verified collection overlays without copying unchanged token tensors."""

import argparse
import json
from pathlib import Path

from .common import file_hash, read_rows, token_hash, write_json, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["all", "paper"], default="all")
    args = parser.parse_args()
    prepared = args.campaign / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert file_hash(path=prepared / "tasks.json") == manifest["tasks_sha256"]
    tasks = json.loads((prepared / "tasks.json").read_text())
    runs = manifest["runs"]
    if args.scope == "paper":
        runs = [run for run in runs if
                "/datasets/collections/weirdchat/Qwen3.5-9B_layer18/parts/" in run["path"] or
                "/datasets/collections/ifeval/Qwen3.5-9B_ifeval_layer18/" in run["path"]]
    selected_tasks = {row["task"] for run in runs for row in run["replacements"]}
    task_records = [None] * len(tasks)
    for task in tasks:
        if task["task"] not in selected_tasks:
            continue
        root = args.campaign / "collection" / f"task_{task['task']:05d}"
        complete = json.loads((root / "task_complete.json").read_text())
        assert complete["status"] == "verified_complete" and complete["input_sha256"] == task["sha256"]
        assert file_hash(path=root / "verification.json") == complete["verification_sha256"]
        assert file_hash(path=root / "artifact_manifest.json") == complete["artifact_manifest_sha256"]
        originals = read_rows(path=prepared / task["input"])
        rollouts = read_rows(path=root / "data/rollouts/rollouts_000000.jsonl")
        assert len(originals) == len(rollouts) == task["rows"]
        for original, rollout in zip(originals, rollouts, strict=True):
            prefix = original["rollout"]["generated_token_ids"]
            assert token_hash(tokens=prefix) == original["prefix_sha256"]
            assert rollout["generated_token_ids"][:len(prefix)] == prefix
            assert rollout["source_identity"] == original["identity"]
            assert rollout["continuation_seed"] == original["continuation_seed"]
            assert len(rollout["generated_token_ids"]) == rollout["generated_token_count"] <= manifest["target_cap"]
            assert rollout["finish_reason"] == "stop" or (rollout["finish_reason"] == "length" and rollout["generated_token_count"] == manifest["target_cap"])
        task_records[task["task"]] = {"task": task["task"], "path": str(root), "completion": complete}
    run_records = []
    for run in runs:
        source = Path(run["path"])
        for relative, expected in run["metadata_sha256"].items():
            assert file_hash(path=source / relative) == expected, source / relative
        replaced = {row["source_index"]: row for row in run["replacements"]}
        assert len(replaced) == len(run["replacements"])
        index_path = args.output / f"runs/run_{run['run']:04d}.jsonl"
        write_rows(path=index_path, rows=[{
            "source_index": index,
            "path": task_records[replaced[index]["task"]]["path"] if index in replaced else str(source),
            "index": replaced[index]["index"] if index in replaced else index,
            "continued": index in replaced,
        } for index in range(run["rows"])])
        run_records.append({**run, "row_index": str(index_path.relative_to(args.output)),
                            "row_index_sha256": file_hash(path=index_path)})
    count = sum(run["rows"] for run in runs)
    continued = sum(len(run["replacements"]) for run in runs)
    counts = {"rollouts": count, "continued": continued, "reused": count - continued}
    write_json(path=args.output / "manifest.json", value={
        **manifest, "status": "verified_complete", "runs": run_records, "tasks": task_records,
        "scope": args.scope, "counts": counts,
        "prepared_manifest_sha256": file_hash(path=prepared / "manifest.json"),
        "layout": "Row indices map to immutable source runs or verified continued runs; reuse metadata and tensors through this index.",
    })
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
