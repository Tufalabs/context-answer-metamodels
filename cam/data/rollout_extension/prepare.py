"""Audit source metadata and prepare only length-terminated answers for extension."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .common import continuation_budget, continuation_seed, file_hash, read_rows, token_hash, write_json, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path, required=True)
    parser.add_argument("--target-cap", type=int, default=4096)
    parser.add_argument("--task-size", type=int, default=64)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    tasks, runs, seen, totals = [], [], set(), Counter()
    for relative in sources:
        source = (args.project / relative).resolve()
        if source in seen:
            continue
        seen.add(source)
        run_number = len(runs)
        cache = args.metadata_cache / f"run_{run_number:04d}"
        files = [source / "run_config.json", source / "data/prompts.jsonl"]
        files += sorted((source / "data/context_tokens").glob("*.jsonl"))
        files += sorted((source / "data/rollouts").glob("*.jsonl"))
        hashes = {}
        for path in files:
            destination = cache / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src=path, dst=destination)
            hashes[str(path.relative_to(source))] = file_hash(path=destination)
        config = json.loads((cache / "run_config.json").read_text())
        generation = config["generation"]
        assert generation["max_new_tokens"] == 1024, source
        assert generation["temperature"] == 1.0 and generation["top_p"] == .95, source
        prompts = read_rows(path=cache / "data/prompts.jsonl")
        contexts = {row["index"]: row for path in sorted((cache / "data/context_tokens").glob("*.jsonl")) for row in read_rows(path=path)}
        rollouts = [row for path in sorted((cache / "data/rollouts").glob("*.jsonl")) for row in read_rows(path=path)]
        rollouts.sort(key=lambda row: row["index"])
        assert [row["index"] for row in prompts] == list(range(len(prompts)))
        assert [row["index"] for row in rollouts] == list(range(len(prompts)))
        assert set(contexts) == set(range(len(prompts)))
        pending, replacements = [], []
        for rollout in rollouts:
            index = rollout["index"]
            prompt, context = prompts[index], contexts[index]
            assert rollout["prompt_sha256"] == context["prompt_sha256"] == prompt["prompt_sha256"]
            assert rollout["rendered_prompt_sha256"] == context["rendered_prompt_sha256"]
            budget = continuation_budget(rollout=rollout, prompt_tokens=context["prompt_token_ids"], source_cap=1024, target_cap=args.target_cap, context_limit=generation["max_model_len"])
            if not budget:
                totals["reused"] += 1
                continue
            identity = f"{source.relative_to(args.project.resolve())}:{index}:{rollout['response_sha256']}"
            pending.append({"identity": identity, "source_run": run_number, "source_index": index,
                            "prompt": prompt, "context": context, "rollout": rollout,
                            "prefix_sha256": token_hash(tokens=rollout["generated_token_ids"]),
                            "continuation_seed": continuation_seed(identity=identity)})
        for start in range(0, len(pending), args.task_size):
            batch = pending[start:start + args.task_size]
            task_number = len(tasks)
            path = args.output / f"tasks/task_{task_number:05d}.jsonl"
            write_rows(path=path, rows=batch)
            tasks.append({"task": task_number, "input": str(path.relative_to(args.output)),
                          "sha256": file_hash(path=path), "rows": len(batch), "source_run": run_number,
                          "source_config": config, "target_cap": args.target_cap})
            replacements.extend({"source_index": row["source_index"], "task": task_number, "index": index}
                                for index, row in enumerate(batch))
        totals["continued"] += len(pending)
        totals["rollouts"] += len(rollouts)
        runs.append({"run": run_number, "path": str(source), "config": config, "metadata_sha256": hashes,
                     "rows": len(rollouts), "replacements": replacements})
        print(json.dumps({"source": str(source), "rows": len(rollouts), "continue": len(pending)}), flush=True)
    write_json(path=args.output / "tasks.json", value=tasks)
    write_json(path=args.output / "manifest.json", value={"status": "prepared", "schema_version": 1,
               "target_cap": args.target_cap, "source_cap": 1024, "runs": runs, "counts": dict(totals),
               "tasks_sha256": file_hash(path=args.output / "tasks.json")})
    print(json.dumps({"counts": dict(totals), "tasks": len(tasks)}), flush=True)


if __name__ == "__main__":
    main()
