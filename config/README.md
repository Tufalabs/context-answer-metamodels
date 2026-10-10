# Prepared inputs

Copy `paths.example.json` to `paths.json` and supply absolute paths.
`paths.json` is machine-specific and ignored by Git. No credentials belong in it.

The main input directory has this layout:

```text
data/
  main_100000/{lmsys,weirdchat}/
  main_500000/{lmsys,weirdchat}/
  cross/conditions/{qwen25,gemma}/lmsys/
  cross/targets/{qwen35,gemma}/lmsys/
prepared/
  lmsys/{train,validation,test}.jsonl
  weirdchat/{train,validation,test}.jsonl
  collection_tasks.json
  manifest.json
  validation.json
collection/task_*/prompt_tokens.safetensors
references/{main,qwen35,gemma}/reference.safetensors
appendix_a/data/{vectors,ood}/
layers/activations/task_*/layer_*.safetensors
CORE_COMPLETE.json
```

The compact datasets use partition directories (`train`, `validation`, `test`;
WeirdChat additionally has `all`) containing safetensors shards. Main tensors
include `prompt_bins` (32 × 4,096), `x_last` (4,096), `y_rollouts`
(4 × 4,096), and `source_global_indices` per prompt. Cross-model conditions and
targets use their model-specific widths and the prescribed rollout bank.
Manifests preserve exact prompt identities and shard ordering. References are
fit on training data only.

The behavior input directory contains `data/` (LMSYS-compatible WeirdChat
partitions and its manifest), `prepared/` (split/labeling manifests), and
`labels/` (`labels.npz`, `event_profile.json`, and `verification.json`). Labels
must cover all four seeds for every prompt; missing or refused judge responses
are never silently treated as negatives.

The additional asset root supplies:

- `datasets/Qwen3.5-9B_ifeval_layer18/compact/ifeval_layer18.safetensors`
- `datasets/Qwen3.5-9B_layer18_full_prompt_tokens_500k/{lmsys,weirdchat}/parts/tokens_*.safetensors`

Exact-token inputs combine that token bank with the main input directory's
collection shards through explicit prompt indices. The data-loading code
checks those indices before training.
