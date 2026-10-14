"""Train one CAM family and size on the prescribed prompt splits."""

import argparse
from pathlib import Path
import torch
from cam.training import scaling as base
from cam.data.common import MAIN_SIZES
from cam.data.common import file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--family", choices=("linear", "mlp", "flow"), required=True)
    p.add_argument("--regime", choices=("lmsys", "combined"), default="lmsys")
    p.add_argument("--n-train", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_float32_matmul_precision("high")
    if a.smoke:
        base.VALIDATION_INTERVAL = 2
        base.VALIDATION_PATIENCE = 1
        base.MAX_TRAINING_STEPS = 20
    train = base.load_partition(a.data, "lmsys", "train", a.family, a.n_train)
    validations = {
        d: base.load_partition(a.data, d, "validation", a.family, 16 if a.smoke else None)
        for d in ("lmsys", "weirdchat")
    }
    tests = {
        d: base.load_partition(
            a.data,
            d,
            "all" if d == "weirdchat" and a.regime == "lmsys" else "test",
            a.family,
            16 if a.smoke else None,
        )
        for d in ("lmsys", "weirdchat")
    }
    weird = (
        base.load_partition(a.data, "weirdchat", "train", a.family, None)
        if a.regime == "combined"
        else None
    )
    # Keep the established per-cell RNG convention across clean scaling runs.
    group = MAIN_SIZES.index(a.n_train) // 3
    old_task = base.FAMILIES.index(a.family) * 10 + base.REGIMES.index(a.regime) * 5 + group
    seed = 42 + old_task * 1000 + a.n_train
    base.run_size(a.family, a.regime, a.n_train, train, weird, validations, tests, a.output, seed)
    path = a.output / a.family / a.regime / f"train_{a.n_train:06d}" / "results.json"
    result = base.read_json(path)
    result.update(
        unique_prompt_manifest_sha256=file_hash(a.data / "manifest.json"),
        n_validation_contexts=len(validations["lmsys"].x),
        n_test_contexts=len(tests["lmsys"].x),
        weirdchat_evaluation_scope="full_unique_2660" if a.regime == "lmsys" else "heldout_532",
        smoke=a.smoke,
        campaign_train_sha256=file_hash(Path(__file__)),
    )
    base.write_json_atomic(path, result)


if __name__ == "__main__":
    main()
