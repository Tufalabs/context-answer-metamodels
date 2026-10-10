# Context–Answer Metamodels

Code for **Context–Answer Metamodels: Forecasting LLM Activations Before the First Answer Token**.

Context–Answer Metamodels (CAMs) predict answer activations from prompt activations,
before generating an answer. We compare linear maps, MLPs, Gaussian models, and
conditional flows, including behavior readouts and transfer across language models.

## Main results

The current manuscript trains on up to **500,000 LMSYS prompts** with four answer
rollouts per prompt. For Qwen3.5-9B at decoder block 18, endpoint R² against the
empirical mean answer activation is:

| CAM | Prompt representation | LMSYS | WeirdChat | IFEval |
| --- | --- | ---: | ---: | ---: |
| Linear | Last token | 0.7844 | 0.6031 | 0.5005 |
| MLP | 32 bins | 0.8739 | 0.7259 | 0.6840 |
| Flow mean | 32 bins | 0.8517 | 0.7234 | 0.6713 |
| MLP | Every token | 0.8808 | 0.7537 | 0.7187 |
| Flow mean | Every token | 0.8599 | 0.7460 | 0.7127 |

LMSYS uses 1,000 validation and 3,000 test prompts. WeirdChat (2,660 prompts) and
IFEval (541 prompts) are zero-shot evaluations. Neural models pool the prompt
bins or tokens with learned-query attention.

- **Distributions:** the Gaussian has slightly lower in-domain energy score;
  flow has lower energy scores on WeirdChat and IFEval. Neither dominates all metrics.
- **Behavior:** MLP forecasts improve held-out WeirdChat AUPRC over direct-context
  probes in all four behavior groups, although some gains are small. Readouts are
  trained on real answer activations; labels come from GPT-6 Luna.
- **Cross-model transfer:** at 100,000 training prompts, an MLP conditioned on
  Qwen2.5 traces predicts Gemma 4 answer means with R² **0.819 ± 0.002**, compared
  with **0.846 ± 0.002** using Gemma 4's own prompt traces (mean ± SD, three fits).

## Install

Python 3.11, CUDA for training, and base R for the main plots:

```bash
uv sync --frozen --extra plots
```

On Kander, batch scripts install the pinned environment in job scratch automatically.

## Run

Prepared activations, split manifests, and labels are required and are not bundled.
See [input formats](config/README.md) and the [experiment guide](slurm/README.md)
for all grids, dependencies, collection entry points, and plotting commands.

```bash
cp config/paths.example.json config/paths.json
# Set your input paths in config/paths.json.
python3 scripts/init_run.py
python3 scripts/submit.py scaling                 # preview commands
python3 scripts/submit.py scaling --submit        # launch on Kander
python3 scripts/submit.py evaluate --after JOB_ID --submit
```

Replace `JOB_ID` with the scaling job ID. Results go under `runs/`; input paths
and generated outputs are ignored by Git. The launchers request B200s and stage
work onto node-local storage. Python experiment modules also expose `--help` for
running in an existing environment on another CUDA host.

## Code

`cam/` contains collection, models, training, and metrics; `experiments/` contains
the paper's experiment entry points; `analysis/` exports metrics and plots results.
Use `python3 scripts/check_project.py` for static checks and
`python -m unittest discover -s tests -v` in the pinned environment for tests.
