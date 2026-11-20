#!/usr/bin/env bash
set -euo pipefail
OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$OPENPI_ROOT/../.." && pwd)"
export DATASET_NAME=pap_tube_blue_green
export CONFIG_NAME=pi05_ur7e_tube_colors_lora
export NORM_STATS_SCRIPT=scripts/compute_ur7e_norm_stats.py
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$VLA_ROOT/.local/openpi/checkpoints}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
# New composition gets its own run. Resume is explicit, avoiding an old 29999 checkpoint.
exec bash "$OPENPI_ROOT/run_pi05_tube1102_background.sh" "${1:-pap_tube_blue_green_50stp}" "${2:-fresh}"
