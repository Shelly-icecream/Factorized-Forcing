#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Default: one machine with six RTX4090 GPUs and the low-memory LoRA profile.
exec torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-6}" train.py \
  --config_path configs/factorized_dmd_framewise.yaml \
  --logdir logs/factorized_dmd --disable-wandb --no_visualize --export-on-complete "$@"
