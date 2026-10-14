"""Fit mixed-domain CAMs using the specified 4096-token generation budget."""

import sys, json
from pathlib import Path
from experiments import scaling as train_main
from cam.data.common import file_hash
from cam.data.common import write_json


def main():
    args = sys.argv
    if "--help" in args or "-h" in args:
        return train_main.main()

    def value(key):
        return args[args.index(key) + 1]

    data = Path(value("--data"))
    manifest = json.loads((data / "manifest.json").read_text())
    assert value("--regime") == "combined" and value("--n-train") == "500000"
    assert (
        manifest["status"] == "verified_complete"
        and manifest["generation"]["max_new_tokens"] == 4096
    )
    assert manifest["weirdchat_split_counts"] == {"train": 1596, "validation": 532, "test": 532}
    train_main.main()
    result_path = Path(value("--output")) / value("--family") / "combined/train_500000/results.json"
    result = json.loads(result_path.read_text())
    result.update(
        corrected_weirdchat_targets=True,
        weirdchat_generation_cap=4096,
        lmsys_generation_cap=4096,
        assembled_data_manifest_sha256=file_hash(data / "manifest.json"),
    )
    write_json(result_path, result)


if __name__ == "__main__":
    main()
