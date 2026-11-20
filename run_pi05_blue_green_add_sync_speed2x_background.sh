#!/usr/bin/env bash
set -euo pipefail
OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$OPENPI_ROOT/../.." && pwd)"
export DATASET_NAME=pap_tube_blue_green_add_sync_speed2x
export CONFIG_NAME=pi05_ur7e_blue_green_add_sync_speed2x_lora
export NORM_STATS_SCRIPT=scripts/compute_ur7e_norm_stats.py
export OPENPI_TRAIN_PYTHON="${OPENPI_TRAIN_PYTHON:-$OPENPI_ROOT/.venv/bin/python}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$VLA_ROOT/.local/openpi/checkpoints}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Leave room for the existing inference service on the 32 GB GPU.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.65}"
exec bash "$OPENPI_ROOT/run_pi05_tube1102_background.sh" "${1:-pap_tube_blue_green_add_sync_speed2x_50stp}" "${2:-fresh}"
