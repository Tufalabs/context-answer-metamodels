"""Create the 45 representation rows that reuse newly evaluated main models."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cam.data.common import MAIN_SIZES, file_hash, write_json
from cam.data.representation_spec import REUSED, cell, core_evaluations, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    run_root = Path(os.environ.get("CAM_RUN_ROOT", project / "runs"))
    parser.add_argument("--campaign", type=Path, default=run_root / "main")
    args = parser.parse_args()
    for family, representation in sorted(REUSED):
        for n in MAIN_SIZES:
            source = args.campaign / f"evaluations/main/train_{n:06d}/results.json"
            core = json.loads(source.read_text())
            value = result(
                family,
                representation,
                n,
                core_evaluations(core, family, n),
                reused=True,
                source_result=str(source),
                source_sha256=file_hash(source),
            )
            write_json(
                args.campaign
                / "appendix_a/fits"
                / cell(family, representation, n)
                / "results.json",
                value,
            )
    print("Exported 45 reused representation results from this rerun.")


if __name__ == "__main__":
    main()
