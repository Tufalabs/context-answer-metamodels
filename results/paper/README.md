# Paper result snapshot

These 18 CSVs contain the numerical results used by the current manuscript:
15 main scaling evaluations, 165 representation cells, cross-model experiments,
rollout and layer ablations, and behavior readouts. `manifest.json` records file
hashes and the source campaigns. No prompt text, rollout text, or activations are
included.

LMSYS, WeirdChat, and IFEval use a 4,096-token answer cap, temperature 1.0, and
top-p 0.95. The main zero-shot results evaluate all 2,660 WeirdChat and 541 IFEval
prompts, with four rollouts each. At 500,000 LMSYS training prompts, WeirdChat
retrieval is 38.9% (linear ridge), 53.1% (MLP), and 47.9% (flow).
Point metrics use the linear ridge and MLP predictions and the flow sample mean.
Mean R² denotes pooled multivariate R² against empirical rollout means.

The paper's merged cross-model endpoint table draws on both
`unique_cross_qwen35.csv` and `unique_cross_gemma.csv`; these files remain separate
by target. Qwen3.5 replicates vary rollout partitions, while Gemma 4 replicates
vary initialization. `unique_probes.csv` reports separate mixed-domain metamodel
fits, as described in the [behavior protocol](../../slurm/README.md#behavior-readouts).

From the repository root:

```bash
Rscript analysis/plot_results.R results/paper outputs/paper-figures
Rscript analysis/plot_layers.R results/paper outputs/paper-figures
```

To export a new run, use `analysis/export.py` as described in the
[experiment guide](../../slurm/README.md#analyze-outputs). The `source` columns
identify the experiment records; the archived campaigns are needed to inspect
those records, but are not needed to plot these CSVs.
