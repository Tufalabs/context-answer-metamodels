#!/bin/bash
#SBATCH --job-name=cam-wl-gaussian
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
cached=$("$CAM_PYTHON" -m experiments.behavior.stage_gaussian --source "$source_campaign/data/main_500000" --cache "/cache/users/$UID/context-answer-metamodels/main")
mkdir -p "$SCRATCH_DIR/input" "$SCRATCH_DIR/mlp" "$SCRATCH_DIR/output" "$campaign/metamodels/gaussian"
ln -s "$cached/lmsys" "$SCRATCH_DIR/input/lmsys"
rsync -a "$campaign/data/" "$SCRATCH_DIR/input/" --exclude test/
cp "$campaign/metamodels/mlp/combined/train_500000/model.safetensors" "$SCRATCH_DIR/mlp/"
rsync -a "$campaign/metamodels/gaussian/" "$SCRATCH_DIR/output/"
trap 'rsync -a "$SCRATCH_DIR/output/" "$campaign/metamodels/gaussian/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.behavior.fit_gaussian --input-dir "$SCRATCH_DIR/input" --mlp-checkpoint "$SCRATCH_DIR/mlp/model.safetensors" --output-dir "$SCRATCH_DIR/output" \
 --mean-checkpoint-label "$campaign/metamodels/mlp/combined/train_500000"
