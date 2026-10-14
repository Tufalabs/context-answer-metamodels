"""Prompt-count grids, artifact I/O, and metric validation."""

import hashlib
import json
import os
from pathlib import Path

MAIN_SIZES = [
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    20000,
    28600,
    50000,
    75000,
    100000,
    200000,
    300000,
    500000,
]
CROSS_SIZES = [100, 300, 1000, 3000, 10000, 28600, 50000, 100000]
APPENDIX_REPRESENTATIONS = {
    "linear": ("32_bins", "last_token", "mean_token"),
    "mlp": ("32_bins", "every_token", "last_token", "mean_token"),
    "flow": ("32_bins", "every_token", "last_token", "mean_token"),
}


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(path, limit=None):
    from itertools import islice

    with Path(path).open() as handle:
        return [json.loads(line) for line in islice(handle, limit) if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def validate_r2_evaluations(evaluations, domains, allow_unrecorded_ci=False):
    """Validate point estimates and distinguish absent source CIs from bad CIs."""
    import math

    assert set(evaluations) == set(domains)
    for domain, count in domains.items():
        metrics = evaluations[domain]
        assert metrics["n_contexts"] == count and math.isfinite(metrics["r2"])
        low, high = metrics["r2_ci_low"], metrics["r2_ci_high"]
        if low is None or high is None:
            assert allow_unrecorded_ci and low is None and high is None
            assert metrics["r2_ci_status"] == "not_recorded_in_source"
        else:
            assert math.isfinite(low) and math.isfinite(high) and low <= high
