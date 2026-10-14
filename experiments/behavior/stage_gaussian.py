"""Stage only train/validation tensors required for the Gaussian baseline."""

import argparse, fcntl, os, shutil
from cam.data.common import file_hash
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--include-test", action="store_true")
    a = p.parse_args()
    root = a.cache / file_hash(a.source / "manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for d in ("lmsys",):
            for part in (
                ("train", "validation", "test") if a.include_test else ("train", "validation")
            ):
                for src in sorted((a.source / d / part).glob("*.safetensors")):
                    dst = root / d / part / src.name
                    if dst.exists() and dst.stat().st_size == src.stat().st_size:
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
        shutil.copy2(a.source / "manifest.json", root / "manifest.json")
    print(root)


if __name__ == "__main__":
    main()
