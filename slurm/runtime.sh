#!/bin/bash
set -euo pipefail
unset SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR VIRTUAL_ENV CAM_PYTHON
. /etc/kander/job-env.sh
source_root="${CAM_PROJECT:-/data/personal/$USER/context-answer-metamodels}"
run_root="${CAM_RUN_ROOT:-$source_root/runs}"
read -r project_dir data_campaign behavior_archive < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["data_project"],p["main_campaign"],p["behavior_campaign"])' "$source_root/config/paths.json")
durable="$run_root/main"
source_campaign="$durable"
campaign="$run_root/behavior"
if [[ ! -f "$run_root/inputs.json" ]]; then
    echo "Run python3 scripts/init_run.py in $source_root before submitting jobs" >&2
    exit 1
fi
code_dir="$SCRATCH_DIR/code"
mkdir -p "$code_dir" "$TMPDIR" "$UV_CACHE_DIR"
for directory in cam experiments scripts config; do
    rsync -a "$source_root/$directory" "$code_dir/" --exclude __pycache__/
done
cp "$source_root/pyproject.toml" "$source_root/uv.lock" "$code_dir/"
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/venv" UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/python"
export UV_LINK_MODE=copy PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="$OMP_NUM_THREADS"
export VLLM_CACHE_ROOT="/cache/users/$UID/context-answer-metamodels/vllm"
export TORCHINDUCTOR_CACHE_DIR="/cache/users/$UID/context-answer-metamodels/inductor"
export TRITON_CACHE_DIR="/cache/users/$UID/context-answer-metamodels/triton"
export XDG_CACHE_HOME="/cache/users/$UID/context-answer-metamodels/xdg"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
uv sync --project "$code_dir" --frozen --extra plots --no-dev --no-install-project --managed-python --python 3.11.15
export CAM_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
if [[ -n ${HF_TOKEN_FILE:-} ]]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi
cd "$code_dir"
