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

Set `main_campaign` to this directory. Set `evaluation_campaign` to the inputs
used for main and representation evaluation:

```text
manifest.json
data/main_100000/{manifest.json,lmsys/,weirdchat/}
data/main_500000/{manifest.json,lmsys/,weirdchat/}
appendix_a/data/{manifest.json,vectors/,ood/}
```

The root, both main datasets, and appendix manifests must report
`status: verified_complete` and `generation.max_new_tokens: 4096`. OOD tensor
metadata records `answer_max_new_tokens: "4096"`. IFEval comes from
`evaluation_campaign/appendix_a/data/ood/ifeval.safetensors`.
`init_run.py` links these inputs separately from the main campaign's cross-model
inputs, prompt identities, and references. Use a fresh run root when changing
collections to keep each run's inputs and results consistent.

The compact datasets use partition directories (`train`, `validation`, `test`;
WeirdChat additionally has `all`) containing safetensors shards. Main tensors
include `prompt_bins` (32 × 4,096), `x_last` (4,096), `y_rollouts`
(4 × 4,096), and `source_global_indices` per prompt. Cross-model conditions and
targets use their model-specific widths and the prescribed rollout bank.
Manifests preserve exact prompt identities and shard ordering. References are
fit on training data only.

The paper uses the following names. CLI values and tensor keys retain their
existing names for compatibility with saved datasets and checkpoints.

| Paper name | Representation key | Input |
| --- | --- | --- |
| Final context-token activation | `last_token` | `x_last`, the raw final context-token activation |
| Mean context activation | `mean_token` | `x_mean`, the unweighted mean over context tokens |
| 32-bin context trace | `32_bins` | `prompt_bins`, 32 context-token bin averages |
| Full context trace | `every_token` | All context-token activations from the token bank |

Bin construction uses adaptive average pooling over contiguous, approximately
equal-sized token intervals. Intervals can overlap; contexts shorter than 32
tokens reuse tokens across bins. Linear ridge concatenates the bin averages;
the MLP and flow use learned-query attention pooling. The Gaussian uses the
MLP mean and its pooled context features.

All layer indices are zero-indexed raw decoder-block outputs:

| Model | Block | Activation width |
| --- | ---: | ---: |
| Qwen3.5-9B | 18 | 4,096 |
| Qwen2.5-7B-Instruct | 19 | 3,584 |
| Gemma-4-12B-it | 47 | 3,840 |

The family key `linear` denotes the linear ridge metamodel. The MLP has three
linear layers (two hidden layers). The flow uses conditional flow matching
with a linear path. Answer targets are mean-pooled answer-token activations.

The behavior input directory contains `data/` (LMSYS-compatible WeirdChat
partitions and its manifest), `prepared/` (split/labeling manifests), and
`labels/` (`labels.npz`, `event_profile.json`, and `verification.json`). Labels
must cover all four seeds for every prompt; missing or refused judge responses
are never silently treated as negatives.

The additional asset root, `data_project`, supplies
`datasets/Qwen3.5-9B_layer18_full_prompt_tokens_500k/{lmsys,weirdchat}/parts/tokens_*.safetensors`.

Exact-token inputs combine that token bank with the main input directory's
collection shards through explicit prompt indices. The data-loading code
checks those indices before training.
