"""Link the published run's inputs into fresh, separate output directories."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = (args.run_root or project / "runs").resolve()
    config = project / "config/paths.json"
    if not config.is_file():
        parser.error("Copy config/paths.example.json to config/paths.json and set your input paths")
    paths = json.loads(config.read_text())
    main = Path(paths["main_campaign"])
    behavior = Path(paths["behavior_campaign"])
    evaluation = Path(paths["evaluation_campaign"])
    manifest = json.loads((evaluation / "manifest.json").read_text())
    if (
        manifest.get("status") != "verified_complete"
        or manifest.get("generation", {}).get("max_new_tokens") != 4096
    ):
        raise ValueError("Evaluation collections must be verified with a 4,096-token answer cap")
    if output == main or output == behavior or output.is_relative_to(Path(paths["data_project"])):
        raise ValueError("Choose an output directory outside the archived input project")
    if (output / "main/data").is_symlink():
        raise ValueError("Choose a fresh run root; data must contain separate input links")
    links = {
        output / "main" / name: main / name
        for name in [
            "data/cross",
            "prepared",
            "collection",
            "references",
            "layers/activations",
            "CORE_COMPLETE.json",
        ]
    }
    links.update(
        {
            output / "main" / name: evaluation / name
            for name in ["data/main_100000", "data/main_500000", "appendix_a/data"]
        }
    )
    links.update(
        {output / "behavior" / name: behavior / name for name in ["data", "labels", "prepared"]}
    )
    for target, source in links.items():
        if not source.exists():
            raise FileNotFoundError(source)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Existing input link points elsewhere: {target}")
        elif target.exists():
            raise FileExistsError(target)
    for target, source in links.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(source, target_is_directory=source.is_dir())
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "inputs.json").write_text(
        json.dumps({str(k.relative_to(output)): str(v) for k, v in links.items()}, indent=2) + "\n"
    )
    print(
        f"Prepared {output}; new fits and evaluations write here, input links are read-only by convention."
    )


if __name__ == "__main__":
    main()
