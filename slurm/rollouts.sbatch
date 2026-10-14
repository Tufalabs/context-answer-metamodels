#!/bin/bash
#SBATCH --job-name=cam-rollouts
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
families=(linear mlp flow)
family=${families[$((task / 180))]}
within=$((task % 180))
input=$("$CAM_PYTHON" -m cam.data.stage \
 --source "$durable/data/main_100000" --cache "$CAM_CACHE_DIR/main" --limit 100000)
targets=$("$CAM_PYTHON" -m cam.data.stage \
 --source "$durable/data/cross/targets/qwen35" --cache "$CAM_CACHE_DIR/targets-qwen35" --limit 100000)
mkdir -p "$SCRATCH_DIR/output" "$durable/fits/rollout_count"
# Each task owns one cell, so parallel stage-out cannot overwrite another fit.
read -r replicate seed_count n < <("$CAM_PYTHON" -c 'from experiments.rollouts import task_specification; import sys; r,s,ns=task_specification(sys.argv[1],int(sys.argv[2])); print(r,len(s),ns[0])' "$family" "$within")
cell="$family/replicate_$replicate/$(printf 'seeds_%02d' "$seed_count")/$(printf 'train_%06d' "$n")"
mkdir -p "$SCRATCH_DIR/output/$cell" "$durable/fits/rollout_count/$cell"
rsync -a "$durable/fits/rollout_count/$cell/" "$SCRATCH_DIR/output/$cell/"
trap 'rsync -a "$SCRATCH_DIR/output/$cell/" "$durable/fits/rollout_count/$cell/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.rollouts \
 --input-dir "$input" --target-dir "$targets" --output-dir "$SCRATCH_DIR/output" \
 --family "$family" --task-index "$within"
