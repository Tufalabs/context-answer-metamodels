#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
project_dir=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_project"])' "$source_root/config/paths.json")
data_campaign=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["main_campaign"])' "$source_root/config/paths.json")
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
export UV_LINK_MODE=copy PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CMAKE_BUILD_PARALLEL_LEVEL="$OMP_NUM_THREADS"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -z ${CAM_PYTHON:-} ]]; then
    unset VIRTUAL_ENV
    export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/venv"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CAM_CACHE_DIR/python}"
    uv sync --project "$code_dir" --frozen --extra plots --no-dev --no-install-project --managed-python --python 3.11.15
    export CAM_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
else
    export CAM_PYTHON
    CAM_PYTHON=$("$CAM_PYTHON" -c 'import sys; print(sys.executable)')
fi
if [[ -n ${HF_TOKEN_FILE:-} ]]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi
cd "$code_dir"
