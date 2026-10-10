#!/bin/bash
#SBATCH --job-name=cam-projection
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
unset SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
. /etc/kander/job-env.sh
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
input=$("$CAM_PYTHON" -m cam.data.stage --source "$durable/data/main_500000" --cache "/cache/users/$UID/context-answer-metamodels/main" --limit 500000)
mkdir -p "$SCRATCH_DIR/checkpoints/flow" "$SCRATCH_DIR/checkpoints/mlp" "$SCRATCH_DIR/output" "$durable/diagnostics/projection"
for name in flow.safetensors normalization.safetensors; do cp "$durable/fits/main/flow/lmsys/train_500000/$name" "$SCRATCH_DIR/checkpoints/flow/"; done
cp "$durable/fits/main/mlp/lmsys/train_500000/model.safetensors" "$SCRATCH_DIR/checkpoints/mlp/"
cp "$durable/evaluations/main/train_500000/results.json" "$SCRATCH_DIR/metrics.json"
trap 'rsync -a "$SCRATCH_DIR/output/" "$durable/diagnostics/projection/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.projection --input-dir "$input" --mlp-checkpoint "$SCRATCH_DIR/checkpoints/mlp/model.safetensors" --flow-checkpoint-dir "$SCRATCH_DIR/checkpoints/flow" --domain-method-result "$SCRATCH_DIR/metrics.json" --output-dir "$SCRATCH_DIR/output" --n-train 500000 --projection-contexts 8192 --total-steps 1000 --n-samples 64 --context-batch 16 --sampling-seed 20260901 --projection-seed 20260902
