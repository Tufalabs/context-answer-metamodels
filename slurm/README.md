# Reproducing the experiments

Run these commands from the repository root.

## Inputs

Reproduction requires prepared activation datasets and behavior labels. Set
machine-specific paths in an ignored configuration file:

```bash
cp config/paths.example.json config/paths.json
# Set the four paths in config/paths.json to your prepared input directories.
python3 scripts/init_run.py
```

`main_campaign` contains the training and cross-model inputs, prompt manifests,
and common metric references. `evaluation_campaign` supplies the main and
appendix evaluation inputs, including IFEval.
`behavior_campaign` contains WeirdChat tensors, split manifests, and Luna labels.
`data_project` supplies full context-trace activation assets.
See [the input layout](../config/README.md).
These inputs are not bundled or automatically downloaded. Collection code is
provided for supplied prompt manifests; exact reproduction requires the same
prompt identities and ordering.

Initialization links inputs into `runs/main/` and `runs/behavior/`. Training
writes new fits and evaluations alongside those links. Treat linked inputs as
read-only. To use a different output location, initialize with
`--run-root /absolute/path` and set `CAM_RUN_ROOT` to that path when submitting.

## Run directly

After installing the pinned environment, run an experiment on a CUDA host:

```bash
python -m experiments.scaling \
  --data runs/main/data/main_100000 --output runs/main/fits/main \
  --family linear --n-train 100
```

This produces `runs/main/fits/main/linear/lmsys/train_000100/`. Change `--family`
and `--n-train` to select other cells. Use `main_500000` for training sizes above
100,000. Every experiment module exposes `--help` for its paths and task indices;
the Python entry points can also be launched by another scheduler.

## Run on Slurm

The batch scripts require Bash, Python 3, `rsync`, and `uv` on the compute node,
plus CUDA for GPU jobs. They stage code and selected inputs into a temporary
workspace, create the pinned Python environment there, and copy results to the
run directory. No site-provided environment script is required.

```bash
python3 scripts/submit.py scaling
python3 scripts/submit.py scaling --submit
python3 scripts/submit.py evaluate --after SCALING_JOB_ID --submit
```

Without `--submit`, the submission script prints commands only. Replace
`SCALING_JOB_ID` with Slurm's returned ID. Use colon-separated IDs for multiple
dependencies. The printed `sbatch` command can be restricted to a smaller array.

Account, partition, QoS, and GPU type are left to your scheduler configuration.
The scripts request one GPU for training, with CPU, memory, and time defaults
in each `.sbatch` file. Supply site-specific options through Slurm's standard
environment variables or the submission helper:

```bash
export SBATCH_ACCOUNT=my_account
export SBATCH_PARTITION=my_gpu_partition
python3 scripts/submit.py scaling --sbatch-arg=--gres=gpu:my_gpu_type:1
python3 scripts/submit.py scaling --sbatch-arg=--array=0 --sbatch-arg=--time=01:00:00 --submit
```

Replace the example names with values from your scheduler. Set `SBATCH_QOS` if
required. Resource and array overrides follow the generated defaults; the
helper passes each `--sbatch-arg` through to `sbatch`. When calling `sbatch`
directly, submit from the repository root or export `CAM_PROJECT` with its
absolute path. See the [Slurm option reference](https://slurm.schedmd.com/sbatch.html).

Storage and runtime settings are environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAM_RUN_ROOT` | Repository's `runs/` | Durable inputs and outputs initialized by `init_run.py` |
| `CAM_SCRATCH_ROOT` | `SLURM_TMPDIR`, then `TMPDIR`, then `/tmp` | Parent directory for a unique workspace per job or array task |
| `CAM_CACHE_DIR` | Workspace's `cache/` | Tensor, model, package, and compiler caches |
| `CAM_PYTHON` | Python installed from `uv.lock` in the workspace | Existing compatible interpreter to use instead of installing an environment |

Use absolute paths. Choose node-local scratch with enough space for the staged
activation tensors and environment. For persistent, rebuildable caches, point
`CAM_CACHE_DIR` at a writable node-local cache directory; standard cache variables
such as `UV_CACHE_DIR` and `HF_HOME` take precedence when already set. Workspaces
are printed in the job log and left for inspection or your site's temporary-file
cleanup. Completed outputs are copied to `CAM_RUN_ROOT`.

| Experiment argument | Grid | Requires |
| --- | --- | --- |
| `scaling` | Linear ridge/MLP/flow × 15 sizes, 100–500,000 prompts | Inputs |
| `evaluate` | 15 metric suites, including validation-selected Gaussian | Scaling fits |
| `cross-model` | 352 Qwen/Gemma fits and evaluations | Inputs |
| `representations` | 120 additional representation fits | Inputs |
| `rollouts` | 45 fits at 100,000 prompts: 2/4/8/12/16 rollouts × three replicates × three families | Inputs |
| `layers` | 32 decoder blocks at 10,000 prompts | Layer activations |
| `solvers` | LMSYS validation/test inference-step ablation | Endpoint flow fit |
| `projection` | Gaussian and flow PCA trajectory | Endpoint evaluation |
| `behavior-maps` | Three mixed-domain CAMs | Behavior inputs |
| `behavior-gaussian` | Gaussian around the mixed MLP | Behavior maps |
| `behavior-probes` | Four event-specific readout bundles | Behavior maps |
| `behavior-samples` | 28 tasks, 1,024 samples per prompt | Maps, Gaussian, probes |
| `behavior-summary` | Validation-selected held-out metrics | Behavior samples |
| `export` | CSV metrics and diagnostic plotting inputs | Completed results |

After main evaluation finishes, run `python3 scripts/reuse_main_results.py`.
This supplies the 45 representation cells that reuse main results, giving 165
cells together with the 120 additional fits. Task numbering determines random
seeds; nonconsecutive array indices preserve the reported protocol.

The main LMSYS split is 500,000/1,000/3,000 train/validation/test prompts.
Zero-shot evaluation uses 2,660 deduplicated WeirdChat prompts and 541 IFEval
prompts. The behavior study uses a separate WeirdChat split of 1,596/532/532,
with four rollout seeds per prompt. All three datasets use a 4,096-token answer
cap, temperature 1.0, and top-p 0.95. The behavior study's model, generation,
judge, readout, and selection settings are in `experiments/behavior/protocol.json`.

The three rollout-count replicates vary the selected training-rollout subsets
and neural training seeds. The subsets coincide at 12 and 16 rollouts; neural
initialization, minibatch sampling, and flow training noise still vary. All
cells are evaluated against the same 16-rollout bank. Cross-model error bars
instead summarize three rollout partitions for the Qwen3.5 target and three
initialization replicates for the Gemma 4 target.

## Behavior readouts

The behavior study trains new metamodels on 500,000 LMSYS prompts and the 1,596
WeirdChat training prompts with equal domain weighting. These differ from the
LMSYS-only metamodels used for zero-shot evaluation. Answer readouts are logistic
regression or a 256-unit, one-hidden-layer SiLU MLP, trained on observed answer
activations and selected by validation log loss for each forecast and behavior group.

The direct-context baseline selects among logistic or MLP readouts of the final
context-token activation or the mean of the 32 context-bin averages, a
learned-query attention MLP over the bins, and global or behavior-specific
prevalence baselines. Every context-only method assigns one probability to all
four answers for answer-level AUPRC and AUROC. The actual-answer reference scores
each answer separately; Gaussian and flow forecasts average readout probabilities
over 1,024 samples per forecast.

## Gaussian scale

For the primary Qwen3.5 model, the Gaussian mean is the MLP forecast. Its covariance is
`kappa² * gamma(z)² * Sigma`, where `kappa` is the validation-selected sample-scale
multiplier. In heteroscedastic mode, the 4,096→256→1 SiLU head predicts log
within-prompt variance relative to the training mean. If `g(z)` is that prediction
clipped to `[-6, 6]`, then
`gamma(z) = clip(exp(g(z)/2) / c, 0.1, 10)`, with
`c = sqrt(mean_train(exp(g(z))))`. Global mode uses `gamma(z) = 1`.
The implementation is in [`cam/models/gaussian.py`](../cam/models/gaussian.py).

## Collection and judging

For supplied prompt manifests, `cam.data.rollouts` generates answers and saves
token activations; `cam.data.collect_task` and `cam.data.collect_compact` collect
the task-based compact representations. `cam.data.collect_contexts` collects
conditioner traces, and `cam.data.collect_layers` collects the layer sweep.
Each module exposes `--help`. Run collection on a CUDA host or in a GPU allocation.
These entry points do not replace the prepared split manifests and data layouts
listed in the [input guide](../config/README.md).

`experiments.behavior.judge` labels prepared answer rows using the fixed Luna
protocol, and `experiments.behavior.merge_labels` validates and merges the
responses. Judging requires `OPENAI_API_KEY` or an explicit `--key-file` and
incurs API usage; it is unnecessary when the prepared labels are available.

## Analyze outputs

After `export` completes, use its CSVs and diagnostic data:

```bash
Rscript analysis/plot_results.R runs/export runs/figures
Rscript analysis/plot_layers.R runs/export runs/figures
python -m cam.plotting.solvers --experiment-root runs/export/diagnostics/solvers --output-dir runs/figures
python -m cam.plotting.trajectory --trajectory-root runs/export/diagnostics/projection --output-dir runs/figures
```

The CSV exporter also accepts explicit run locations:

```bash
python analysis/export.py --campaign runs/main --behavior runs/behavior --output runs/export
```

Completed main and representation results record
`evaluation_generation_max_new_tokens: 4096`. Cached results, reused representation
rows, and exports reject missing or mismatched caps. To load main and
representation results from a separate directory, pass
`--source-overlay /path/to/evaluation-results`; it must contain all
15 main and 165 representation result files.

To evaluate saved checkpoints with a fixed Gaussian configuration, pass
`--gaussian-selection /path/to/results.json` to `experiments.evaluate`.
For Slurm, set `CAM_GAUSSIAN_SELECTION_DIR` to a directory containing
`train_NNNNNN/results.json` for each size. By default, evaluations select the Gaussian
using LMSYS validation. The published result CSVs and plots can also be used
directly; see [results/paper](../results/paper/README.md).

Run data aggregation on a machine with sufficient memory, or inside a compute allocation.
