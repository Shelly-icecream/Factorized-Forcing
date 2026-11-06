#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# One process per GPU, one machine with eight visible A100 GPUs.
# Extra train.py arguments can override logdir, max steps, resume, etc.
exec torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/factorized_dmd_framewise.yaml \
  --disable-wandb --no_visualize "$@"
