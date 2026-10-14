#!/bin/bash
#SBATCH --job-name=cam-cross
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=112G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
source "${CAM_PROJECT:-$PWD}/slurm/runtime.sh"
task=${SLURM_ARRAY_TASK_ID:?}
read -r target conditioner family replicate n < <("$CAM_PYTHON" -c 'from experiments.cross_model import TASKS; import sys; print(*TASKS[int(sys.argv[1])])' "$task")
condition_source="$durable/data/cross/conditions/$conditioner"
if [[ $conditioner == qwen35 ]]; then condition_source="$durable/data/main_100000"; fi
condition=$("$CAM_PYTHON" -m cam.data.stage \
 --source "$condition_source" --cache "$CAM_CACHE_DIR/conditions-$conditioner" --limit "$n")
targets=$("$CAM_PYTHON" -m cam.data.stage \
 --source "$durable/data/cross/targets/$target" --cache "$CAM_CACHE_DIR/targets-$target" --limit "$n")
cell="$target/$conditioner/$family/replicate_$replicate/$(printf 'train_%06d' "$n")"
mkdir -p "$SCRATCH_DIR/output" "$durable/fits/cross/$cell"
rsync -a "$durable/fits/cross/$cell/" "$SCRATCH_DIR/output/" --exclude '.*.tmp-*'
cp "$durable/references/$target/reference.safetensors" "$SCRATCH_DIR/reference.safetensors"
sync_output() { rsync -a "$SCRATCH_DIR/output/" "$durable/fits/cross/$cell/" --exclude '.*.tmp-*'; }
trap sync_output EXIT
"$CAM_PYTHON" -m experiments.cross_model \
 --condition "$condition" --targets "$targets" --output "$SCRATCH_DIR/output" \
 --task "$task" --reference "$SCRATCH_DIR/reference.safetensors"
