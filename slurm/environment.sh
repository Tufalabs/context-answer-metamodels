#!/bin/bash
set -euo pipefail
source_root=$(cd -- "${CAM_PROJECT:-$(dirname -- "${BASH_SOURCE[0]}")/..}" && pwd)
run_root="${CAM_RUN_ROOT:-$source_root/runs}"
scratch_root="${CAM_SCRATCH_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
mkdir -p "$scratch_root"
export SCRATCH_DIR
SCRATCH_DIR=$(mktemp -d "$scratch_root/cam-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}.XXXXXX")
export TMPDIR="$SCRATCH_DIR/tmp" TEMPDIR="$SCRATCH_DIR/tmp"
export CAM_CACHE_DIR="${CAM_CACHE_DIR:-$SCRATCH_DIR/cache}"
mkdir -p "$TMPDIR" "$CAM_CACHE_DIR"
CAM_CACHE_DIR=$(cd -- "$CAM_CACHE_DIR" && pwd)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CAM_CACHE_DIR/uv}"
export HF_HOME="${HF_HOME:-$CAM_CACHE_DIR/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CAM_CACHE_DIR/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CAM_CACHE_DIR/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CAM_CACHE_DIR/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CAM_CACHE_DIR/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CAM_CACHE_DIR/triton}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" \
    "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
printf 'Job workspace: %s\n' "$SCRATCH_DIR" >&2
