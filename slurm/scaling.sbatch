#!/bin/bash
#SBATCH --job-name=cam-main
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
sizes=(100 250 500 1000 2500 5000 10000 20000 28600 50000 75000 100000 200000 300000 500000)
families=(linear mlp flow)
task=${SLURM_ARRAY_TASK_ID:?}
n=${sizes[$((task % 15))]}
family=${families[$((task / 15))]}
regime=${REGIME:-lmsys}
data=main_500000
if (( n <= 100000 )); then data=main_100000; fi
input=$("$CAM_PYTHON" -m cam.data.stage \
 --source "$durable/data/$data" --cache "/cache/users/$UID/context-answer-metamodels/main" --limit "$n")
cell="$family/$regime/$(printf 'train_%06d' "$n")"
mkdir -p "$SCRATCH_DIR/output/$cell" "$durable/fits/main/$cell"
rsync -a "$durable/fits/main/$cell/" "$SCRATCH_DIR/output/$cell/" --exclude '.*.tmp-*'
sync_output() { rsync -a "$SCRATCH_DIR/output/$cell/" "$durable/fits/main/$cell/" --exclude '.*.tmp-*'; }
trap sync_output EXIT
"$CAM_PYTHON" -m experiments.scaling \
 --data "$input" --output "$SCRATCH_DIR/output" --family "$family" --regime "$regime" --n-train "$n"
