"""Check companion source syntax, internal imports, and Slurm scripts."""

import ast
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    failures = []
    files = [
        path
        for group in ("cam", "experiments", "analysis", "scripts", "tests")
        for path in (root / group).rglob("*.py")
    ]
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
                if (root / Path(*node.module.split("."))).is_dir():
                    names = [node.module + "." + item.name for item in node.names]
            for name in names:
                if name.split(".")[0] in {"cam", "experiments", "tests"}:
                    target = root / Path(*name.split("."))
                    if not target.is_dir() and not target.with_suffix(".py").is_file():
                        failures.append(f"{path.relative_to(root)}: missing import {name}")
    shells = list((root / "slurm").glob("*.sh")) + list((root / "slurm").glob("*.sbatch"))
    shells += list((root / "experiments/behavior").glob("*.sbatch"))
    for path in shells:
        checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if checked.returncode:
            failures.append(checked.stderr)
    print(
        json.dumps(
            {"python_files": len(files), "shell_files": len(shells), "failures": failures}, indent=2
        )
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
