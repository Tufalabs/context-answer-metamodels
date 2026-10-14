#!/bin/bash
#SBATCH --job-name=cam-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
source "${CAM_PROJECT:-$PWD}/slurm/runtime.sh"
sizes=(100 250 500 1000 2500 5000 10000 20000 28600 50000 75000 100000 200000 300000 500000)
n=${sizes[${SLURM_ARRAY_TASK_ID:?}]}
tag=$(printf 'train_%06d' "$n")
data=main_500000
if (( n <= 100000 )); then data=main_100000; fi
input=$("$CAM_PYTHON" -m cam.data.stage \
 --source "$durable/data/$data" --cache "$CAM_CACHE_DIR/main" --limit "$n")
mkdir -p "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/output" "$durable/evaluations/main/$tag"
for family in linear mlp flow; do
  mkdir -p "$SCRATCH_DIR/checkpoints/$family"
  rsync -a "$durable/fits/main/$family/lmsys/$tag/" "$SCRATCH_DIR/checkpoints/$family/"
done
cp "$durable/appendix_a/data/ood/ifeval.safetensors" "$SCRATCH_DIR/ifeval.safetensors"
cp "$durable/references/main/reference.safetensors" "$SCRATCH_DIR/reference.safetensors"
rsync -a "$durable/evaluations/main/$tag/" "$SCRATCH_DIR/output/"
trap 'rsync -a "$SCRATCH_DIR/output/" "$durable/evaluations/main/$tag/" --exclude ".*.tmp-*"' EXIT
selection_args=()
if [[ -n ${CAM_GAUSSIAN_SELECTION_DIR:-} ]]; then
    cp "$CAM_GAUSSIAN_SELECTION_DIR/$tag/results.json" "$SCRATCH_DIR/gaussian_selection.json"
    selection_args=(--gaussian-selection "$SCRATCH_DIR/gaussian_selection.json")
fi
"$CAM_PYTHON" -m experiments.evaluate \
 --input-dir "$input" --ifeval-compact "$SCRATCH_DIR/ifeval.safetensors" \
 --linear-checkpoint "$SCRATCH_DIR/checkpoints/linear/model.safetensors" \
 --mlp-checkpoint "$SCRATCH_DIR/checkpoints/mlp/model.safetensors" \
 --flow-checkpoint-dir "$SCRATCH_DIR/checkpoints/flow" --reference "$SCRATCH_DIR/reference.safetensors" \
 --output-dir "$SCRATCH_DIR/output" --n-train "$n" "${selection_args[@]}"
