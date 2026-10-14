"""Replace continued OOD targets while preserving existing contexts and row order."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from .common import file_hash, write_json
from .reader import answer_tensor_location


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    overlay_path = (args.overlay or args.campaign / "overlay") / "manifest.json"
    overlay = json.loads(overlay_path.read_text())
    assert overlay["status"] == "verified_complete" and overlay["target_cap"] == 4096
    args.scratch.mkdir(parents=True, exist_ok=True)
    cache = {}

    def vector(*, root, index):
        path, offset = answer_tensor_location(row={"path": str(root), "index": index})
        if path not in cache:
            local = args.scratch / "answer.safetensors"
            shutil.copyfile(src=path, dst=local)
            cache[path] = load_file(filename=local)["v_answer_mean"].clone()
        values = cache[path]
        assert values.shape[1:] == (1, 4096)
        return values[offset, 0]

    replacements = {"weirdchat": {}, "ifeval": {}}
    for run in overlay["runs"]:
        source = Path(run["path"])
        relative = source.relative_to(args.project)
        if relative.parts[:4] == ("datasets", "collections", "weirdchat", "Qwen3.5-9B_layer18"):
            domain = "weirdchat"
            offset = run["config"]["source_prompt_range"]["start"]
        elif relative.parts[:4] == ("datasets", "collections", "ifeval", "Qwen3.5-9B_ifeval_layer18"):
            domain, offset = "ifeval", 0
        else:
            continue
        seed = run["config"]["generation"]["seed"]
        for row in run["replacements"]:
            key = (offset + row["source_index"], seed - 43)
            assert key not in replacements[domain]
            new_root = Path(overlay["tasks"][row["task"]]["path"])
            replacements[domain][key] = (
                vector(root=source, index=row["source_index"]),
                vector(root=new_root, index=row["index"]),
            )

    def patch(*, tensors, domain, indices):
        count = 0
        for position, global_index in enumerate(indices):
            for seed_index in range(4):
                pair = replacements[domain].get((global_index, seed_index))
                if pair is None:
                    continue
                old, new = pair
                target = tensors["y_rollouts"][position, seed_index]
                assert torch.equal(input=target, other=old.to(dtype=target.dtype)), (domain, global_index, seed_index)
                target.copy_(new)
                count += 1
        return count

    def save(*, tensors, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        for value in tensors.values():
            assert not value.is_floating_point() or torch.isfinite(value).all()
        save_file(tensors={key: value.contiguous() for key, value in tensors.items()}, filename=path,
                  metadata={"answer_max_new_tokens": "4096", "continuation_campaign": str(args.campaign)})
        return file_hash(path=path)

    def stage_load(*, path):
        local = args.scratch / "input.safetensors"
        shutil.copyfile(src=path, dst=local)
        return load_file(filename=local)

    provenance = {"status": "verified_complete", "generation": {"max_new_tokens": 4096, "temperature": 1.0, "top_p": .95},
                  "overlay_manifest_sha256": file_hash(path=overlay_path)}
    weird_all_indices = []
    for size in (100000, 500000):
        source = args.source_campaign / f"data/main_{size}"
        destination = args.output / f"data/main_{size}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "lmsys").symlink_to(target=source / "lmsys", target_is_directory=True)
        output_hashes = {}
        for path in sorted((source / "weirdchat").glob("*/*.safetensors")):
            tensors = stage_load(path=path)
            indices = tensors["source_global_indices"].tolist()
            patch(tensors=tensors, domain="weirdchat", indices=indices)
            if size == 500000 and path.parent.name == "all":
                weird_all_indices.extend(indices)
            relative = path.relative_to(source)
            output_hashes[str(relative)] = save(tensors=tensors, path=destination / relative)
        original = json.loads((source / "manifest.json").read_text())
        write_json(path=destination / "manifest.json", value={**original, **provenance,
                   "source_manifest_sha256": file_hash(path=source / "manifest.json"), "output_files_sha256": output_hashes})
    assert len(weird_all_indices) == len(set(weird_all_indices)) == 2660
    source_appendix = args.source_campaign / "appendix_a/data"
    output_appendix = args.output / "appendix_a/data"
    output_appendix.mkdir(parents=True, exist_ok=True)
    for path in source_appendix.iterdir():
        if path.name not in {"ood", "manifest.json"}:
            (output_appendix / path.name).symlink_to(target=path, target_is_directory=path.is_dir())
    (output_appendix / "ood").mkdir(exist_ok=True)
    appendix_hashes = {}
    for path in (source_appendix / "ood").iterdir():
        destination = output_appendix / "ood" / path.name
        if path.name not in {"weirdchat.safetensors", "ifeval.safetensors"}:
            destination.symlink_to(target=path, target_is_directory=path.is_dir())
            continue
        domain = path.stem
        tensors = stage_load(path=path)
        indices = weird_all_indices if domain == "weirdchat" else list(range(541))
        assert len(tensors["y_rollouts"]) == len(indices)
        patched = patch(tensors=tensors, domain=domain, indices=indices)
        expected = sum(index in set(indices) for index, seed in replacements[domain])
        assert patched == expected
        appendix_hashes[str(destination.relative_to(output_appendix))] = save(tensors=tensors, path=destination)
    write_json(path=output_appendix / "manifest.json", value={**provenance, "output_files_sha256": appendix_hashes,
               "source_manifest_sha256": file_hash(path=source_appendix / "manifest.json")})
    write_json(path=args.output / "manifest.json", value={**provenance, "source_campaign": str(args.source_campaign),
               "replaced_rollouts": {domain: len(values) for domain, values in replacements.items()}})


if __name__ == "__main__":
    main()
