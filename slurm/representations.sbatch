#!/bin/bash
#SBATCH --job-name=cam-appendix-fit
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
test -f "$data_campaign/CORE_COMPLETE.json"
task=${SLURM_ARRAY_TASK_ID:?}
read -r family representation n < <("$CAM_PYTHON" -c 'from cam.data.representation_spec import TASKS; import sys; print(*TASKS[int(sys.argv[1])])' "$task")
input=$("$CAM_PYTHON" -m cam.data.stage --source "$durable/data/main_500000" --cache "/cache/users/$UID/context-answer-metamodels/main" --limit "$n")
appendix=$("$CAM_PYTHON" -m cam.data.stage_representations --source "$durable/appendix_a/data" --cache "/cache/users/$UID/context-answer-metamodels/appendix" --limit "$n")
cell="$family/$representation/$(printf 'train_%06d' "$n")"
mkdir -p "$SCRATCH_DIR/output" "$durable/appendix_a/fits/$cell"
rsync -a "$durable/appendix_a/fits/$cell/" "$SCRATCH_DIR/output/" --exclude '.*.tmp-*'
trap 'rsync -a "$SCRATCH_DIR/output/" "$durable/appendix_a/fits/$cell/" --exclude ".*.tmp-*"' EXIT
args=(--project "$project_dir" --campaign "$data_campaign" --main "$input" --appendix "$appendix" --cache "/cache/users/$UID/context-answer-metamodels/exact_tokens" --output "$SCRATCH_DIR/output" --task "$task")
"$CAM_PYTHON" -m experiments.representations "${args[@]}"
