"""Print paper-scoped Slurm commands; add --submit to dispatch them."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


EXPERIMENTS = (
    "scaling",
    "evaluate",
    "cross-model",
    "representations",
    "rollouts",
    "layers",
    "solvers",
    "projection",
    "behavior-maps",
    "behavior-gaussian",
    "behavior-probes",
    "behavior-samples",
    "behavior-summary",
    "export",
)


def commands(project, experiment, dependency, run_root):
    arrays = {
        "scaling": "0-44",
        "evaluate": "0-14",
        "cross-model": "0-351",
        "layers": "0-31",
        "solvers": "0-1",
        "rollouts": ",".join(map(str, range(11, 540, 12))),
    }
    if experiment == "representations":
        tasks = [
            (family, representation, size)
            for family in ("linear", "mlp", "flow")
            for representation in ("32_bins", "every_token", "last_token", "mean_token")
            for size in range(15)
        ]
        reused = {("linear", "last_token"), ("mlp", "32_bins"), ("flow", "32_bins")}
        arrays[experiment] = ",".join(
            str(index)
            for index, (family, representation, _) in enumerate(tasks)
            if (family, representation) not in reused | {("linear", "every_token")}
        )
    prefix = ["sbatch", f"--output={run_root}/logs/%x-%A_%a.log"]
    if dependency:
        prefix.append("--dependency=afterok:" + dependency)
    if experiment == "behavior-maps":
        return [
            prefix
            + ["--export=ALL,FAMILY=" + family, str(project / "experiments/behavior/train.sbatch")]
            for family in ("linear", "mlp", "flow")
        ]
    if experiment.startswith("behavior-"):
        name = experiment.removeprefix("behavior-")
        filename = {"samples": "sample"}.get(name, name)
        array = {"samples": "0-27", "probes": "0-3"}.get(name)
        path = project / "experiments/behavior" / f"{filename}.sbatch"
    else:
        filename = {"cross-model": "cross_model"}.get(experiment, experiment)
        array = arrays.get(experiment)
        path = project / "slurm" / f"{filename}.sbatch"
    return [prefix + (["--array=" + array] if array else []) + [str(path)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--after", help="Colon-separated job IDs that must finish successfully")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run_root = Path(os.environ.get("CAM_RUN_ROOT", project / "runs")).resolve()
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("SLURM_") and key not in {"SLURM_CONF", "SLURM_CONF_SERVER", "SLURM_JWT"}:
            environment.pop(key, None)
    environment["CAM_PROJECT"] = str(project)
    environment["CAM_RUN_ROOT"] = str(run_root)
    if args.submit:
        if not (run_root / "inputs.json").is_file():
            parser.error("Initialize this output directory with scripts/init_run.py first")
        (run_root / "logs").mkdir(parents=True, exist_ok=True)
    for command in commands(
        project=project, experiment=args.experiment, dependency=args.after, run_root=run_root
    ):
        print(shlex.join(command), flush=True)
        if args.submit:
            subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
