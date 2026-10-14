"""Resumable standard Responses API judge, with strict binary labels and provenance."""

import argparse, json, hashlib, os, shlex, time, threading, random, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from experiments.behavior.judge_prompt import messages
from experiments.behavior.judge_prompt import parse
from cam.data.common import read_rows
from cam.data.common import write_json
from cam.data.common import file_hash


def key_from_file(path):
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "OPENAI_API_KEY":
            parts = shlex.split(value, comments=True)
            if parts and parts[0]:
                return parts[0]
    raise RuntimeError("OPENAI_API_KEY missing from specified credential file")


def body_for(row):
    return {
        "model": "gpt-6-luna",
        "input": messages(row),
        "reasoning": {"effort": "none"},
        "max_output_tokens": 1024,
        "store": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--tasks", type=int, default=8)
    p.add_argument("--key-file", type=Path, help="Defaults to OPENAI_API_KEY in the environment")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(a.input)
    assert len(rows) == 10640 and len({r["example_id"] for r in rows}) == 10640
    selected = [r for i, r in enumerate(rows) if i % a.tasks == a.task]
    path = a.output / f"task_{a.task:03d}.jsonl"
    done = {}
    for f in [a.output / "pilot.jsonl", path]:
        if f.exists():
            for r in read_rows(f):
                done[r["example_id"]] = r
    credential = key_from_file(a.key_file) if a.key_file else os.environ.get("OPENAI_API_KEY")
    if not credential:
        p.error("Set OPENAI_API_KEY or supply --key-file")
    lock = threading.Lock()
    next_time = [0.0]

    def execute(row):
        body = body_for(row)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        saved = done.get(row["example_id"])
        if saved:
            assert saved["body_sha256"] == digest
            return saved
        for attempt in range(8):
            with lock:
                delay = max(0, next_time[0] - time.monotonic())
                next_time[0] = max(time.monotonic(), next_time[0]) + 0.5
            time.sleep(delay)
            request = urllib.request.Request(
                "https://api.openai.com/v1/responses",
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": "Bearer " + credential,
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as handle:
                    response = json.load(handle)
                    request_id = handle.headers.get("x-request-id")
                assert response.get("model") == "gpt-6-luna", (
                    "Unexpected judge model; no substitution allowed"
                )
                assert (
                    response.get("usage", {})
                    .get("output_tokens_details", {})
                    .get("reasoning_tokens", 0)
                    == 0
                ), "Unexpected reasoning tokens"
                value, status, raw = parse(response)
                return {
                    "example_id": row["example_id"],
                    "global_index": row["global_index"],
                    "seed": row["seed"],
                    "split": row["split"],
                    "partition_position": row["partition_position"],
                    "behavior_id": row["behavior_id"],
                    "body_sha256": digest,
                    "judge_model": "gpt-6-luna",
                    "reasoning_effort": "none",
                    "status": status,
                    "match": value["match"] if value else None,
                    "explanation": value["explanation"] if value else None,
                    "raw_response": raw,
                    "response": response,
                    "request_id": request_id,
                }
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(
                        f"Judge HTTP {exc.code}; response body and credentials suppressed"
                    ) from None
                time.sleep(min(30, 2**attempt) + random.random())
            except (urllib.error.URLError, TimeoutError):
                time.sleep(min(30, 2**attempt))
        raise RuntimeError("API transient retries exhausted; saved rows can be resumed")

    # Fail fast on a real request, before launching concurrent paid work.
    if selected:
        pilot = execute(selected[0])
        assert pilot["status"] == "ok", (
            "First task judgment did not parse; stop before more requests"
        )
        done[pilot["example_id"]] = pilot
        if not path.exists() or pilot["example_id"] not in {
            r["example_id"] for r in read_rows(path)
        }:
            with path.open("a") as f:
                f.write(json.dumps(pilot) + "\n")
    with path.open("a") as handle, ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(execute, row): row for row in selected[1:]}
        saved_here = {r["example_id"] for r in read_rows(path)}
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["example_id"] not in saved_here:
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                saved_here.add(result["example_id"])
            if n % 100 == 0:
                print(f"judged {n + 1}/{len(selected)}", flush=True)
    complete = {r["example_id"]: r for r in read_rows(path)}
    assert set(complete) == {r["example_id"] for r in selected}
    invalid = [r["example_id"] for r in complete.values() if r["status"] != "ok"]
    write_json(
        path.with_suffix(".json"),
        {
            "status": "complete" if not invalid else "needs_label_review",
            "judgments": len(complete),
            "invalid": invalid,
            "input_sha256": file_hash(a.input),
            "output_sha256": file_hash(path),
            "judge_model": "gpt-6-luna",
            "reasoning_effort": "none",
            "draws_per_rollout": 1,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} missing/refused/incomplete judgments; no false labels substituted"
        )


if __name__ == "__main__":
    main()
