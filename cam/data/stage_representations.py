"""Stage only the Appendix A inputs needed by a fit into a locked node cache."""

import argparse
import fcntl
import shutil
from pathlib import Path
from cam.data.common import file_hash
from cam.data.tokens import cached_file
from cam.data.stage import stage_dataset


def main():
    p = argparse.ArgumentParser()
    for name in ("source", "cache"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    a = p.parse_args()
    vectors = stage_dataset(a.source / "vectors", a.cache / "vectors", a.limit)
    root = a.cache / file_hash(a.source / "manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for source in sorted((a.source / "ood").glob("*.safetensors")):
            cached_file(source, root / "ood" / source.name)
        if not (root / "vectors").exists():
            (root / "vectors").symlink_to(vectors, target_is_directory=True)
        shutil.copy2(a.source / "manifest.json", root / "manifest.json")
    print(root)


if __name__ == "__main__":
    main()
