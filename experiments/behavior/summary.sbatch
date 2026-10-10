#!/bin/bash
#SBATCH --job-name=cam-behavior-summary
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
mkdir -p "$SCRATCH_DIR/input" "$SCRATCH_DIR/output" "$campaign/summary"
rsync -a "$campaign/predictions" "$campaign/labels" "$campaign/probes" "$SCRATCH_DIR/input/" --exclude '*.safetensors'
trap 'rsync -a "$SCRATCH_DIR/output/" "$campaign/summary/"' EXIT
"$CAM_PYTHON" -m experiments.behavior.summarize \
 --prediction-dir "$SCRATCH_DIR/input/predictions" --label-dir "$SCRATCH_DIR/input/labels" \
 --probe-dir "$SCRATCH_DIR/input/probes" --output-dir "$SCRATCH_DIR/output" \
 --flow-checkpoint-label "$campaign/metamodels/flow/combined/train_500000" \
 --mlp-checkpoint-label "$campaign/metamodels/mlp/combined/train_500000" \
 --probe-normalization-label "$campaign/metamodels/flow/combined/train_500000"
