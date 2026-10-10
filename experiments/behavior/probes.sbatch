#!/bin/bash
#SBATCH --job-name=cam-wl-probes
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
mkdir -p "$SCRATCH_DIR/flow" "$SCRATCH_DIR/output"
rsync -a "$campaign/data/" "$SCRATCH_DIR/input/"
rsync -a "$campaign/labels/" "$SCRATCH_DIR/labels/"
cp "$campaign/metamodels/flow/combined/train_500000/normalization.safetensors" "$SCRATCH_DIR/flow/"
events=(any_target_behavior any_safety_failure self_harm_or_eating_disorder deception_or_false_capability)
event=${events[${SLURM_ARRAY_TASK_ID:?}]}
mkdir -p "$campaign/probes/$event" "$SCRATCH_DIR/output/$event"
rsync -a "$campaign/probes/$event/" "$SCRATCH_DIR/output/$event/"
trap 'rsync -a "$SCRATCH_DIR/output/$event/" "$campaign/probes/$event/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.behavior.train_event --input-dir "$SCRATCH_DIR/input" --label-dir "$SCRATCH_DIR/labels" --flow-dir "$SCRATCH_DIR/flow" --output-dir "$SCRATCH_DIR/output" --event-index "$SLURM_ARRAY_TASK_ID"
