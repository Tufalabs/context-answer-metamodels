#!/bin/bash
#SBATCH --job-name=cam-layerfit
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
unset SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
. /etc/kander/job-env.sh
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
layer=$(printf 'layer_%03d' "${SLURM_ARRAY_TASK_ID:?}")
mkdir -p "$SCRATCH_DIR/data" "$SCRATCH_DIR/output" "$durable/layers/fits/$layer"
for task in {0..13}; do
  name=$(printf 'task_%03d' "$task")
  mkdir -p "$SCRATCH_DIR/data/$name"
  cp "$durable/layers/activations/$name/$layer.safetensors" "$SCRATCH_DIR/data/$name/"
done
rsync -a "$durable/layers/fits/$layer/" "$SCRATCH_DIR/output/"
trap 'rsync -a "$SCRATCH_DIR/output/" "$durable/layers/fits/$layer/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.layers --data "$SCRATCH_DIR/data" --layer "$SLURM_ARRAY_TASK_ID" --output "$SCRATCH_DIR/output"
