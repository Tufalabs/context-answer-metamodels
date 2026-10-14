# Context–Answer Metamodels

Code for **Context–Answer Metamodels: Forecasting LLM Activations Before the First Answer Token**.

Context–Answer Metamodels (CAMs) predict answer activations from context activations,
before generating an answer. We compare linear ridge, MLP, Gaussian, and flow
metamodels, including behavior readouts and transfer across language models.
The flow uses conditional flow matching with a linear path.

## Main results

The current manuscript trains on up to **500,000 LMSYS-Chat-1M prompts** with four
answer rollouts per prompt. For Qwen3.5-9B at decoder block 18 (zero-indexed),
endpoint R² against the empirical mean answer activation is shown below.
This is a pooled multivariate R², not an average of per-dimension R² values;
the flow forecast is its sample mean.

| CAM | Context representation | LMSYS | WeirdChat | IFEval |
| --- | --- | ---: | ---: | ---: |
| Linear ridge | Final context-token activation | 0.7844 | 0.6095 | 0.4996 |
| MLP | 32-bin context trace | 0.8739 | 0.7309 | 0.6839 |
| Flow | 32-bin context trace | 0.8517 | 0.7286 | 0.6711 |
| MLP | Full context trace | 0.8808 | 0.7591 | 0.7183 |
| Flow | Full context trace | 0.8599 | 0.7520 | 0.7119 |

LMSYS uses 1,000 validation and 3,000 test prompts. WeirdChat (2,660 prompts) and
IFEval (541 prompts) are zero-shot evaluations. The neural models pool the context
bins or tokens with learned-query attention.
LMSYS, WeirdChat, and IFEval rollouts use a 4,096-token generation cap,
temperature 1.0, and top-p 0.95.

WeirdChat top-1 retrieval at 500,000 training prompts is **38.9%** for the linear ridge,
**53.1%** for the MLP, and **47.9%** for the flow. The current numerical results
and their provenance are included in [results/paper](results/paper/README.md).
Regenerate the main and supplementary empirical plots without activation inputs:

```bash
Rscript analysis/plot_results.R results/paper outputs/paper-figures
Rscript analysis/plot_layers.R results/paper outputs/paper-figures
```

- **Distributions:** the Gaussian has slightly lower in-domain energy score;
  the flow has lower energy scores on WeirdChat and IFEval. Neither dominates all metrics.
- **Behavior:** MLP forecasts improve held-out WeirdChat AUPRC over direct-context
  probes in all four behavior groups, although some gains are small. Readouts are
  trained on observed answer activations; labels come from GPT-6 Luna. These are
  new mixed-domain metamodel fits on 500,000 LMSYS prompts and the 1,596-prompt
  WeirdChat training split, with equal domain weighting. See the
  [behavior protocol](slurm/README.md#behavior-readouts) for baseline and readout definitions.
- **Cross-model transfer:** at 100,000 training prompts, an MLP conditioned on
  Qwen2.5-7B-Instruct context traces predicts Gemma-4-12B-it answer means with
  R² **0.819 ± 0.002**, compared with **0.846 ± 0.002** using Gemma 4's own
  context traces (mean ± SD, three initialization replicates).

## Install

Python 3.11, CUDA for training, and base R for the main plots. Render plots in a
UTF-8 locale so mathematical labels and hyphens display correctly.

```bash
uv sync --frozen --extra plots
source .venv/bin/activate
```

The Python entry points run in this environment on a CUDA host. Slurm is optional.

## Run

Prepared activations, split manifests, and labels are required and are not bundled.
See [input formats](config/README.md) and the [experiment guide](slurm/README.md)
for all grids, dependencies, collection entry points, and plotting commands.

```bash
cp config/paths.example.json config/paths.json
# Set your input paths in config/paths.json.
python3 scripts/init_run.py
python -m experiments.scaling \
  --data runs/main/data/main_100000 --output runs/main/fits/main \
  --family linear --n-train 100
```

This runs one scaling experiment. The modules expose `--help` for their input
paths and experiment settings. Results go under `runs/`; machine-specific input
paths and generated outputs are ignored by Git.

For the full grids on Slurm:

```bash
python3 scripts/submit.py scaling                 # preview commands
python3 scripts/submit.py scaling --submit
python3 scripts/submit.py evaluate --after JOB_ID --submit
```

Replace `JOB_ID` with the scaling job ID. Set your scheduler's account, partition,
and GPU requirements through `SBATCH_*` environment variables or repeatable
`--sbatch-arg` options. The [experiment guide](slurm/README.md) describes resource
overrides, scratch storage, and all grids.

## Code

`cam/` contains collection, models, training, and metrics; `experiments/` contains
the paper's experiment entry points; `analysis/` exports metrics and plots results.
