"""Re-extract all-layer summary activations from existing, fixed responses."""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForMultimodalLM
from safetensors.torch import save_file
from cam.data import activation_collection as base
from cam.data.collect_compact import make_batches
from cam.data.common import file_hash
from cam.data.common import read_rows
from cam.data.common import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(a.tokens)
    profile = base.MODEL_PROFILES["Qwen/Qwen3.5-9B"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", revision=profile["revision"])
    _, suffix = base.assistant_suffix(tokenizer, profile["chat_template_kwargs"])
    samples = []
    for pos, row in enumerate(rows):
        _, expected = base.render_prompt(tokenizer, row["prompt"], profile["chat_template_kwargs"])
        for seedpos, r in enumerate(row["rollouts"]):
            assert r["seed"] == 43 + seedpos and r["prompt_token_ids"] == expected
            generated = r["generated_token_ids"]
            assert generated
            overlap = base.suffix_overlap(generated, suffix) if r["finish_reason"] == "stop" else 0
            appended = suffix[overlap:] if r["finish_reason"] == "stop" else []
            full = expected + generated + appended
            assert len(full) <= 8192
            samples.append((pos, seedpos, len(expected), full))
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.5-9B",
        revision=profile["revision"],
        dtype=torch.bfloat16,
        device_map={"": torch.device("cuda:0")},
    ).eval()
    blocks = base.resolve_decoder_blocks(model)
    assert len(blocks) == 32
    last = torch.empty((len(rows), 32, 4096), dtype=torch.bfloat16)
    targets = torch.empty((len(rows), 4, 32, 4096), dtype=torch.bfloat16)
    lengths = [len(v[3]) for v in samples]
    batches = make_batches(lengths, 8, 8192)
    for bi, indices in enumerate(batches):
        ids = torch.full(
            (len(indices), max(lengths[i] for i in indices)),
            tokenizer.pad_token_id or tokenizer.eos_token_id,
            dtype=torch.long,
            device="cuda",
        )
        mask = torch.zeros_like(ids)
        for j, i in enumerate(indices):
            ids[j, : lengths[i]] = torch.tensor(samples[i][3], device="cuda")
            mask[j, : lengths[i]] = 1
        captured = base.capture_block_outputs(model, blocks, ids, mask, list(range(32)))
        for layer, states in captured.items():
            for j, i in enumerate(indices):
                pos, seedpos, start, _ = samples[i]
                if seedpos == 0:
                    last[pos, layer] = states[j, start - 1].cpu().bfloat16()
                targets[pos, seedpos, layer] = (
                    states[j, start : lengths[i]].float().mean(0).cpu().bfloat16()
                )
        if bi % 50 == 0:
            print(f"all-layer batches {bi + 1}/{len(batches)}", flush=True)
        del ids, mask, captured
    assert torch.isfinite(last).all() and torch.isfinite(targets).all()
    hashes = {}
    for layer in range(32):
        path = a.output / f"layer_{layer:03d}.safetensors"
        save_file(
            {
                "x_last": last[:, layer].contiguous(),
                "y_rollouts": targets[:, :, layer].contiguous(),
                "source_global_indices": torch.tensor([r["source_global_index"] for r in rows]),
            },
            path,
        )
        hashes[path.name] = file_hash(path)
    write_json(
        a.output / "verification.json",
        {
            "status": "verified_complete",
            "tokens_sha256": file_hash(a.tokens),
            "output_files_sha256": hashes,
            "rows": len(rows),
            "response_generation": False,
        },
    )


if __name__ == "__main__":
    main()
