#!/bin/bash
#SBATCH --job-name=cam-solvers
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
unset SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
. /etc/kander/job-env.sh
source "${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}/slurm/runtime.sh"
case "${SLURM_ARRAY_TASK_ID:?}" in
0) regime=lmsys; domain=lmsys; split=validation ;;
1) regime=lmsys; domain=lmsys; split=test ;;
*) echo "Paper solver tasks are 0 (validation) and 1 (test)" >&2; exit 1 ;;
esac
input=$("$CAM_PYTHON" -m cam.data.stage --source "$durable/data/main_100000" --cache "/cache/users/$UID/context-answer-metamodels/main" --limit 0)
cell="$regime/$domain/$split"
mkdir -p "$SCRATCH_DIR/checkpoint" "$SCRATCH_DIR/output" "$durable/diagnostics/solvers/$cell"
for name in flow.safetensors normalization.safetensors results.json; do cp "$durable/fits/main/flow/$regime/train_500000/$name" "$SCRATCH_DIR/checkpoint/"; done
trap 'rsync -a "$SCRATCH_DIR/output/" "$durable/diagnostics/solvers/$cell/" --exclude ".*.tmp-*"' EXIT
"$CAM_PYTHON" -m experiments.solvers --checkpoint-dir "$SCRATCH_DIR/checkpoint" --input-dir "$input" --regime "$regime" --domain "$domain" --split "$split" --output-dir "$SCRATCH_DIR/output" --n-samples 64 --context-batch 16 --bootstrap-draws 2000 --sampling-seed 20260901
