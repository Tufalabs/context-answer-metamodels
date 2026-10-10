"""One new Appendix A fit, selected only on the common LMSYS validation set."""

import argparse
import time
from pathlib import Path
import torch
from safetensors.torch import save_file
from cam.training import scaling as base
from cam.training import tokens as exact
from cam.training import binned_ridge as linear
from cam.models import flow as flow_base
from cam.data.representations import load_compact
from cam.data.representations import load_exact
from cam.data.representation_spec import TASKS
from cam.data.representation_spec import REUSED
from cam.data.representation_spec import result
from cam.data.representation_spec import r2_metrics
from cam.data.common import file_hash
from cam.data.common import write_json


@torch.inference_mode()
def flow_mean(model, condition, values, device, seed, encoded=False):
    predictions = []
    for start in range(0, len(condition), 16):
        x = condition[start : start + 16].to(device=device, dtype=torch.float32)
        if not encoded:
            x = (x - values["x_mean"]) / values["x_scale"]
        samples = flow_base.sample_flow(model, x, 64, 32, seed + start)
        predictions.append((samples.mean(1) * values["y_scale"] + values["y_mean"]).cpu())
    return torch.cat(predictions)


def main():
    p = argparse.ArgumentParser()
    for name in ("project", "campaign", "main", "appendix", "cache", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    a = p.parse_args()
    family, representation, n = TASKS[a.task]
    assert (family, representation) not in REUSED
    assert (a.campaign / "CORE_COMPLETE.json").exists()
    if family == "linear" and representation == "every_token":
        raise ValueError("every-token linear fits were removed from Appendix A")
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "results.json").exists():
        return
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    started = time.monotonic()
    seed = 20261005 + a.task * 10000 + n
    if representation == "every_token":
        train, val, tests = load_exact(a.project, a.campaign, a.main, a.appendix, a.cache, n)
        values = exact.target_normalization(train, None, family, device)
        fitter = exact.fit_point if family == "mlp" else exact.fit_flow
        model, training = fitter("token_query", train, None, {"lmsys": val}, values, device, seed)
        base.save_tensors_atomic(a.output / "model.safetensors", model.state_dict())
        base.save_tensors_atomic(a.output / "normalization.safetensors", values)
        predictions = {}
        for i, (domain, data) in enumerate(tests.items()):
            if family == "mlp":
                pred = exact.point_predict(model, data, values, device, 128)
            else:
                condition = exact.encode_dataset(model, data, device, 128)
                pred = flow_mean(
                    exact.EncodedFlowView(model),
                    condition,
                    values,
                    device,
                    seed + 100000 * (i + 1),
                    encoded=True,
                )
            predictions[domain] = pred
    else:
        train, val, tests = load_compact(a.main, a.appendix, representation, n)
        predictions = {}
        if family == "linear":
            if representation == "32_bins" and n > 10000:
                as_point = lambda d: linear.PointData(d.x, d.y.float().mean(1))
                weight, values, training = linear.fit_binned_linear(
                    as_point(train),
                    as_point(val),
                    [316.0, 1000.0, 3162.0, 10000.0],
                    device,
                    256,
                    512,
                    10,
                    1e-5,
                    3,
                    seed,
                )
                base.save_tensors_atomic(
                    a.output / "model.safetensors",
                    {"weight": weight, **{f"normalization.{k}": v for k, v in values.items()}},
                )
                weight = weight.to(device)
                for domain, data in tests.items():
                    parts = []
                    for x in data.x.split(128):
                        x = (x.to(device).float() - values["x_mean"].to(device)) / values[
                            "x_scale"
                        ].to(device)
                        parts.append(
                            (
                                torch.nn.functional.linear(x.flatten(1), weight)
                                + values["y_mean"].to(device)
                            )
                            .detach()
                            .cpu()
                        )
                    predictions[domain] = torch.cat(parts)
            else:
                # Solve the underdetermined small binned problems exactly;
                # ten epochs would otherwise mean very few optimizer updates.
                original_hidden = base.HIDDEN
                if representation == "32_bins":
                    for data in [train, val, *tests.values()]:
                        data.x = data.x.flatten(1)
                    base.HIDDEN = train.x.shape[1]
                try:
                    state, training = base.fit_linear(train, None, {"lmsys": val}, device)
                finally:
                    base.HIDDEN = original_hidden
                base.save_tensors_atomic(a.output / "model.safetensors", state)
                predictions = {
                    domain: base.linear_predict(state, data, device)
                    for domain, data in tests.items()
                }
        else:
            # A one-token sequence makes learned-query pooling exactly identity.
            # The prediction head, normalization and optimizer remain the main recipe.
            for data in [train, val, *tests.values()]:
                if data.x.ndim == 2:
                    data.x = data.x[:, None, :]
            values = base.normalization(train, None, family, device)
            fitter = base.fit_mlp if family == "mlp" else base.fit_flow
            model, training = fitter(train, None, {"lmsys": val}, values, device, seed)
            base.save_tensors_atomic(a.output / "model.safetensors", model.state_dict())
            base.save_tensors_atomic(a.output / "normalization.safetensors", values)
            for i, (domain, data) in enumerate(tests.items()):
                predictions[domain] = (
                    base.mlp_predict(model, data, values, device)
                    if family == "mlp"
                    else flow_mean(model, data.x, values, device, seed + 100000 * (i + 1))
                )
    evaluations = {
        domain: r2_metrics(pred, tests[domain].y, seed + i)
        for i, (domain, pred) in enumerate(predictions.items())
    }
    save_file(
        {d: p.contiguous() for d, p in predictions.items()}, a.output / "predictions.safetensors"
    )
    write_json(
        a.output / "results.json",
        result(
            family,
            representation,
            n,
            evaluations,
            training=training,
            elapsed_seconds=time.monotonic() - started,
            fit_seed=seed,
            reused=False,
            data_manifest_sha256=file_hash(a.appendix / "manifest.json"),
            wrapper_sha256=file_hash(Path(__file__)),
            flow_mean_estimator={"samples": 64, "euler_steps": 32} if family == "flow" else None,
        ),
    )


if __name__ == "__main__":
    main()
