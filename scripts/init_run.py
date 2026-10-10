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
    if output == main or output == behavior or output.is_relative_to(Path(paths["data_project"])):
        raise ValueError("Choose an output directory outside the archived input project")
    links = {
        output / "main" / name: main / name
        for name in [
            "data",
            "prepared",
            "collection",
            "references",
            "appendix_a/data",
            "layers/activations",
        ]
    }
    links.update(
        {output / "behavior" / name: behavior / name for name in ["data", "labels", "prepared"]}
    )
    for target, source in links.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Existing input link points elsewhere: {target}")
        elif target.exists():
            raise FileExistsError(target)
    for target, source in links.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(source, target_is_directory=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "inputs.json").write_text(
        json.dumps({str(k.relative_to(output)): str(v) for k, v in links.items()}, indent=2) + "\n"
    )
    print(
        f"Prepared {output}; new fits and evaluations write here, input links are read-only by convention."
    )


if __name__ == "__main__":
    main()
