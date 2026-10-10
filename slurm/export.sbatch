#!/bin/bash
#SBATCH --job-name=cam-export
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail
unset SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
. /etc/kander/job-env.sh
project="${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}"
runs="${CAM_RUN_ROOT:-$project/runs}"
work="$SCRATCH_DIR/export"
mkdir -p "$work/main" "$work/behavior/summary" "$work/output" "$runs/export"
for directory in evaluations fits appendix_a/fits layers/fits; do
    mkdir -p "$work/main/$directory"
    rsync -a "$runs/main/$directory/" "$work/main/$directory/" --include '*/' --include results.json --exclude '*'
done
mkdir -p "$work/main/prepared"
cp "$runs/main/prepared/validation.json" "$work/main/prepared/"
cp "$runs/behavior/summary/selected_test_metrics.csv" "$work/behavior/summary/"
python3 "$project/analysis/export.py" \
 --campaign "$work/main" --behavior "$work/behavior" --output "$work/output"
mkdir -p "$work/output/diagnostics/projection" "$work/output/diagnostics/solvers"
rsync -a "$runs/main/diagnostics/projection/" "$work/output/diagnostics/projection/" --include results.json --include projection_points.npz --exclude '*'
rsync -a "$runs/main/diagnostics/solvers/" "$work/output/diagnostics/solvers/" --include '*/' --include results.json --exclude '*'
rsync -a "$work/output/" "$runs/export/"
