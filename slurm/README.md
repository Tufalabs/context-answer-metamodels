# Reproducing the experiments

Run these commands from the repository root.

## Inputs

Reproduction requires prepared activation datasets and behavior labels. Set
machine-specific paths in an ignored configuration file:

```bash
cp config/paths.example.json config/paths.json
# Set the three paths in config/paths.json to your prepared input directories.
python3 scripts/init_run.py
```

`main_campaign` contains the prepared LMSYS/WeirdChat/IFEval inputs and common
metric references. `behavior_campaign` contains the four-rollout, 4,096-token
WeirdChat tensors, split manifests, and Luna labels. `data_project` supplies
additional IFEval and exact-token activation assets. See [the input layout](../config/README.md).
These inputs are not bundled or automatically downloaded. Collection code is
provided for supplied prompt manifests; exact reproduction requires the same
prompt identities and ordering.

Initialization links inputs into `runs/main/` and `runs/behavior/`. Training
writes new fits and evaluations alongside those links. Treat linked inputs as
read-only. To use a different output location, initialize with
`--run-root /absolute/path` and set `CAM_RUN_ROOT` to that path when submitting.

## Run experiments

On Kander, the batch scripts create a lockfile-based environment in job scratch,
stage inputs on node-local storage, and copy outputs back before the job ends.
They use the managed `uv` and `/etc/kander/job-env.sh` environment.

```bash
python3 scripts/check_project.py
python3 scripts/submit.py scaling
python3 scripts/submit.py scaling --submit
python3 scripts/submit.py evaluate --after SCALING_JOB_ID --submit
```

Without `--submit`, the submission script prints commands only. Replace
`SCALING_JOB_ID` with Slurm's returned ID. Use colon-separated IDs for multiple
dependencies. The printed `sbatch` command can be restricted to a smaller array.

| Experiment argument | Grid | Requires |
| --- | --- | --- |
| `scaling` | Linear/MLP/flow × 15 sizes, 100–500,000 prompts | Inputs |
| `evaluate` | 15 metric suites, including validation-selected Gaussian | Scaling fits |
| `cross-model` | 352 Qwen/Gemma fits and evaluations | Inputs |
| `representations` | 120 additional representation fits | Inputs |
| `rollouts` | 45 fits at 100,000 prompts: 2/4/8/12/16 rollouts × 3 seeds × 3 families | Inputs |
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
with four rollout seeds per prompt. Its model, generation, judge, readout, and
selection settings are in `experiments/behavior/protocol.json`.

Every experiment exposes `--help` and can be run directly with
`python -m experiments.<name>` in the pinned environment on another CUDA host;
the supplied Slurm environment setup is specific to Kander.

## Collection and judging

For supplied prompt manifests, `cam.data.rollouts` generates answers and saves
token activations; `cam.data.collect_task` and `cam.data.collect_compact` collect
the task-based compact representations. `cam.data.collect_contexts` collects
conditioner traces, and `cam.data.collect_layers` collects the layer sweep.
Each module exposes `--help`. Run collection inside a GPU allocation.
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

Run data aggregation and ML tests inside a compute allocation. With the pinned
Python environment active, run `python -m unittest discover -s tests -v`.
