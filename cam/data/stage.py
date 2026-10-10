"""Stage only required immutable tensor shards under a per-node locked cache."""

import argparse
import fcntl
import os
import shutil
from pathlib import Path
from cam.data.common import file_hash


def stage_dataset(source, destination, limit):
    # A completed manifest fingerprints the dataset, avoiding mutable shared
    # cache directories. Each shard is published atomically under the lock.
    marker = source / "manifest.json"
    version = file_hash(marker)
    cache = destination / version
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for domain in ("lmsys", "weirdchat"):
            for partition in ("train", "validation", "test", "all"):
                for path in sorted((source / domain / partition).glob("*.safetensors")):
                    start = int(path.stem.split("_")[-2])
                    if domain == "lmsys" and partition == "train" and start >= limit:
                        continue
                    dest = cache / domain / partition / path.name
                    if dest.exists() and dest.stat().st_size == path.stat().st_size:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
                    shutil.copy2(path, temp)
                    os.replace(temp, dest)
        shutil.copy2(marker, cache / "manifest.json")
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    a = p.parse_args()
    print(stage_dataset(a.source, a.cache, a.limit))


if __name__ == "__main__":
    main()
