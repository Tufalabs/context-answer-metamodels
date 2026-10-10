#!/bin/bash
#SBATCH --job-name=cam-wl-metamodel
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
family=${FAMILY:?}
cached=$("$CAM_PYTHON" -m experiments.behavior.stage_gaussian --source "$source_campaign/data/main_500000" --cache "/cache/users/$UID/context-answer-metamodels/main" --include-test)
mkdir -p "$SCRATCH_DIR/input" "$SCRATCH_DIR/output/$family/combined/train_500000" "$campaign/metamodels/$family/combined/train_500000"
ln -s "$cached/lmsys" "$SCRATCH_DIR/input/lmsys"
rsync -a "$campaign/data/" "$SCRATCH_DIR/input/"
rsync -a "$campaign/metamodels/$family/combined/train_500000/" "$SCRATCH_DIR/output/$family/combined/train_500000/"
trap 'rsync -a "$SCRATCH_DIR/output/$family/combined/train_500000/" "$campaign/metamodels/$family/combined/train_500000/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.behavior.train_mixed --data "$SCRATCH_DIR/input" --output "$SCRATCH_DIR/output" --family "$family" --regime combined --n-train 500000
