"""Require complete Luna label coverage and preserve prompt/seed alignment."""

import argparse, json, shutil
from pathlib import Path
import numpy as np
from cam.data.common import read_rows
from cam.data.common import write_json
from cam.data.common import file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--judge", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    judged = {}
    hashes = {}
    for task in range(a.tasks):
        path = a.judge / f"task_{task:03d}.jsonl"
        m = json.loads(path.with_suffix(".json").read_text())
        assert m["status"] == "complete" and m["output_sha256"] == file_hash(path)
        hashes[path.name] = file_hash(path)
        for row in read_rows(path):
            key = (row["global_index"], row["seed"])
            assert key not in judged
            assert row["status"] == "ok" and isinstance(row["match"], bool)
            judged[key] = row
    assert len(judged) == 10640
    arrays = {}
    counts = {}
    for part in ("train", "validation", "test"):
        rows = read_rows(a.prepared / "splits" / f"{part}.jsonl")
        labels = []
        for pos, row in enumerate(rows):
            values = []
            for seed in (43, 44, 45, 46):
                j = judged[row["source_domain_index"], seed]
                assert (j["split"], j["partition_position"], j["behavior_id"]) == (
                    part,
                    pos,
                    row["behavior_id"],
                )
                values.append(j["match"])
            labels.append(values)
        arrays[part + "_match"] = np.asarray(labels, dtype=np.int8)
        arrays[part + "_behavior_id"] = np.asarray([r["behavior_id"] for r in rows])
        arrays[part + "_prompt_id"] = np.asarray([r["prompt_id"] for r in rows])
        counts[part] = {
            "prompts": len(rows),
            "positive_rollouts": int(arrays[part + "_match"].sum()),
        }
    np.savez(a.output / "labels.npz", **arrays)
    shutil.copy2(a.prepared / "event_profile.json", a.output / "event_profile.json")
    write_json(
        a.output / "verification.json",
        {
            "status": "complete_luna_judged",
            "judge_model": "gpt-6-luna",
            "reasoning_effort": "none",
            "judge_draws_per_rollout": 1,
            "rollouts_per_prompt": 4,
            "rollout_labels": 10640,
            "counts": counts,
            "labels_sha256": file_hash(a.output / "labels.npz"),
            "judge_output_sha256": hashes,
            "generation_cap": 4096,
        },
    )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
