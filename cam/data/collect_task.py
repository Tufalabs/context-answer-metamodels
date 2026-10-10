"""Execute one manifest-defined missing-trace task; resume saved generation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from cam.data.common import file_hash
from cam.data.common import read_rows
from cam.data.common import write_rows
from cam.data.common import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    task = json.loads((a.prepared / "collection_tasks.json").read_text())[a.task]
    assert task["task"] == a.task
    source = a.prepared / task["prompts"]
    assert file_hash(source) == task["prompt_file_sha256"]
    rows = read_rows(source)
    if a.smoke:
        rows = rows[:8]
    a.work.mkdir(parents=True, exist_ok=True)
    identity = {"task": task, "smoke": a.smoke, "rows": len(rows)}
    identity_path = a.work / "task_identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, (
            "attempted to reuse another task workspace"
        )
    else:
        write_json(identity_path, identity)
    promptfile = a.work / "input_prompts.jsonl"
    write_rows(promptfile, rows)
    model, layer = {
        "qwen35": ("Qwen/Qwen3.5-9B", 18),
        "qwen25": ("Qwen/Qwen2.5-7B-Instruct", 19),
        "gemma": ("google/gemma-4-12B-it", 47),
    }[task["model"]]
    existing_config = a.work / "run_config.json"
    if task["model"] != "qwen25" and existing_config.exists():
        config = json.loads(existing_config.read_text())
        assert config["model_id"] == model and config["generation"]["seed"] == task["seed"], (
            "saved traces have a different model or seed"
        )
    existing_prompts = a.work / "data/prompts.jsonl"
    if existing_prompts.exists():
        stored = read_rows(existing_prompts)
        assert [(r["source_global_index"], r["prompt_sha256"]) for r in stored] == [
            (r["source_global_index"], r["prompt_sha256"]) for r in rows
        ], "saved prompt identities do not match this task"
    if task["model"] == "qwen25":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cam.data.collect_contexts",
                "--model-id",
                model,
                "--layer",
                str(layer),
                "--prompts-file",
                str(promptfile),
                "--output-dir",
                str(a.work),
                "--max-prompt-tokens",
                "8192",
                "--max-sequences",
                "32",
                "--max-batch-tokens",
                "4096",
            ],
            check=True,
        )
        verification = json.loads((a.work / "verification.json").read_text())
    else:
        if not (a.work / "generation_complete.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cam.data.rollouts",
                    "--stage",
                    "generate",
                    "--model-id",
                    model,
                    "--layers",
                    str(layer),
                    "--run-dir",
                    str(a.work),
                    "--durable-run-dir",
                    str(a.work),
                    "--prompts-file",
                    str(promptfile),
                    "--n-contexts",
                    str(len(rows)),
                    "--generation-chunk-size",
                    "64",
                    "--activation-shard-size",
                    "8",
                    "--max-model-len",
                    "8192",
                    "--max-new-tokens",
                    "4096",
                    "--temperature",
                    "1.0",
                    "--top-p",
                    "0.95",
                    "--seed",
                    str(task["seed"]),
                    "--gpu-memory-utilization",
                    "0.85",
                    "--max-num-seqs",
                    "64",
                ],
                check=True,
            )
        args = [
            sys.executable,
            "-m",
            "cam.data.collect_compact",
            "--run-dir",
            str(a.work),
            "--model-id",
            model,
            "--layer",
            str(layer),
            "--n-contexts",
            str(len(rows)),
            "--max-full-tokens",
            "8192",
            "--max-sequences",
            "8",
            "--max-batch-tokens",
            "8192",
        ]
        if task["model"] == "qwen35" and task["seed"] == 43:
            args.append("--save-prompt-tokens")
        subprocess.run(args, check=True)
        verification = json.loads((a.work / "onpolicy_verification.json").read_text())
    assert verification["status"] == "verified_complete"
    assert verification["model_id"] == model and verification["layer"] == layer
    if task["model"] != "qwen25":
        config = json.loads((a.work / "run_config.json").read_text())
        assert config["generation"]["seed"] == task["seed"]
    write_json(
        a.work / "task_complete.json",
        {
            "status": "verified_complete",
            "task": task,
            "smoke": a.smoke,
            "rows": len(rows),
            "input_sha256": file_hash(promptfile),
            "verification": verification,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "scratch_dir": os.environ.get("SCRATCH_DIR"),
            "task_identity_sha256": file_hash(identity_path),
        },
    )


if __name__ == "__main__":
    main()
