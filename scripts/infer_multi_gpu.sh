#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Set CUDA_VISIBLE_DEVICES to select GPUs and NPROC_PER_NODE to match their count.
exec torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-6}" \
  inference_multi_gpu.py "$@"
