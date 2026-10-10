#!/bin/bash
#SBATCH --job-name=cam-wl-predict
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
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
mkdir -p "$SCRATCH_DIR/output"
rsync -a "$campaign/data/" "$SCRATCH_DIR/input/" --exclude train/
for family in mlp flow; do
 mkdir -p "$SCRATCH_DIR/$family"
 rsync -a "$campaign/metamodels/$family/combined/train_500000/" "$SCRATCH_DIR/$family/" --include '*.safetensors' --exclude '*'
done
mkdir -p "$SCRATCH_DIR/linear" "$SCRATCH_DIR/gaussian"
rsync -a "$campaign/metamodels/linear/combined/train_500000/" "$SCRATCH_DIR/linear/" --include '*.safetensors' --exclude '*'
rsync -a "$campaign/metamodels/gaussian/" "$SCRATCH_DIR/gaussian/" --include '*.safetensors' --include results.json --exclude '*'
rsync -a "$campaign/probes/" "$SCRATCH_DIR/probes/"
mkdir -p "$campaign/predictions"
trap 'rsync -a "$SCRATCH_DIR/output/" "$campaign/predictions/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.behavior.sample --input-dir "$SCRATCH_DIR/input" --flow-dir "$SCRATCH_DIR/flow" --mlp-dir "$SCRATCH_DIR/mlp" --linear-dir "$SCRATCH_DIR/linear" --gaussian-dir "$SCRATCH_DIR/gaussian" --probe-dir "$SCRATCH_DIR/probes" --output-dir "$SCRATCH_DIR/output" --task-index "$SLURM_ARRAY_TASK_ID" --num-tasks 28 --n-samples 1024 --n-steps 32 --context-batch 2 \
 --flow-checkpoint-label "$campaign/metamodels/flow/combined/train_500000" \
 --mlp-checkpoint-label "$campaign/metamodels/mlp/combined/train_500000" \
 --linear-checkpoint-label "$campaign/metamodels/linear/combined/train_500000" \
 --probe-normalization-label "$campaign/metamodels/flow/combined/train_500000"
