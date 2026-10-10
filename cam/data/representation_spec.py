"""Context representation grid and evaluation schema."""

from cam.data.common import APPENDIX_REPRESENTATIONS
from cam.data.common import MAIN_SIZES
from cam.data.common import validate_r2_evaluations

FAMILIES = ("linear", "mlp", "flow")
REPRESENTATIONS = ("32_bins", "every_token", "last_token", "mean_token")
DOMAINS = {"LMSYS": 3000, "WeirdChat": 2660, "IFEval": 541}
TASKS = [
    (family, representation, n)
    for family in FAMILIES
    for representation in REPRESENTATIONS
    for n in MAIN_SIZES
]
ACTIVE_TASKS = [task for task in TASKS if task[1] in APPENDIX_REPRESENTATIONS[task[0]]]
REUSED = {("linear", "last_token"), ("mlp", "32_bins"), ("flow", "32_bins")}


def cell(family, representation, n):
    return f"{family}/{representation}/train_{n:06d}"


def validate_evaluations(evaluations, allow_unrecorded_ci=False):
    validate_r2_evaluations(evaluations, DOMAINS, allow_unrecorded_ci)


def core_evaluations(core, family, n):
    """Reuse recorded estimates; the core flow evaluator did not save R² CIs."""
    assert core["status"] == "complete" and core["smoke"] is False
    assert core["domains"] == DOMAINS and core["n_train_contexts"] == n
    evaluations = {}
    for domain, count in DOMAINS.items():
        point = core["point"][family][domain]
        keys = ("sample_mean_r2_ci_low", "sample_mean_r2_ci_high")
        present = [key in point for key in keys]
        assert all(present) or (family == "flow" and not any(present))
        evaluations[domain] = {
            "r2": point["sample_mean_r2"],
            "r2_ci_low": point.get(keys[0]),
            "r2_ci_high": point.get(keys[1]),
            "n_contexts": count,
            "r2_ci_status": "recorded" if all(present) else "not_recorded_in_source",
        }
    validate_evaluations(evaluations, allow_unrecorded_ci=family == "flow")
    return evaluations


def result(family, representation, n, evaluations, **extra):
    validate_evaluations(
        evaluations,
        allow_unrecorded_ci=bool(extra.get("reused"))
        and (family, representation) == ("flow", "32_bins"),
    )
    return {
        "status": "complete",
        "model_id": "Qwen/Qwen3.5-9B",
        "layer": 18,
        "family": family,
        "representation": representation,
        "n_train": n,
        "n_validation": 1000,
        "training_domain": "LMSYS only",
        "selection_domain": "LMSYS validation only",
        "evaluations": evaluations,
        "paper_updates": False,
        **extra,
    }


def r2_metrics(prediction, targets, seed):
    import numpy as np

    target = targets.float().mean(1) if targets.ndim == 3 else targets.float()
    prediction = prediction.float().cpu()
    target = target.cpu()
    sse = (prediction - target).double().square().sum(1).numpy()
    tss = (target.double() - target.double().mean(0)).square().sum(1).numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(500):
        ids = rng.integers(0, len(sse), len(sse))
        estimates.append(1 - sse[ids].sum() / tss[ids].sum())
    return {
        "r2": float(1 - sse.sum() / tss.sum()),
        "r2_ci_low": float(np.quantile(estimates, 0.025)),
        "r2_ci_high": float(np.quantile(estimates, 0.975)),
        "n_contexts": len(target),
        "r2_ci_status": "recorded",
    }
