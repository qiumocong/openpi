#!/usr/bin/env bash
set -euo pipefail

# Run from the OpenPI repository root, regardless of the caller's directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
# A VLA checkout keeps machine-local resources outside its submodules.
VLA_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
if [[ -f "$VLA_ROOT/.gitmodules" && "$PROJECT_ROOT" == "$VLA_ROOT/third_party/openpi" ]]; then
    RESOURCE_ROOT="$VLA_ROOT/.local"
else
    RESOURCE_ROOT="$PROJECT_ROOT/.local"
fi
ASSETS_BASE_DIR="${OPENPI_ASSETS_BASE_DIR:-$RESOURCE_ROOT/openpi/assets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$RESOURCE_ROOT/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/qmc/openpi_data/checkpoints}"

# Usage: ./run_pi05_training_background.sh [run_name] [resume|overwrite|fresh]
CONFIG_NAME="${OPENPI_CONFIG:-pi05_ur7e_base_lora}"
RUN_NAME="${1:-dataset1012_50stp}"
MODE="${2:-resume}"

LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/${CONFIG_NAME}_${RUN_NAME}.log"
PID_FILE="$LOG_DIR/${CONFIG_NAME}_${RUN_NAME}.pid"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$CONFIG_NAME/$RUN_NAME"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"

case "$MODE" in
    resume|overwrite|fresh) ;;
    *)
        echo "Usage: $0 [run_name] [resume|overwrite|fresh]" >&2
        exit 2
        ;;
esac

# Validate local resources before train.py can initialize or overwrite a run.
uv run --no-sync python - "$CONFIG_NAME" "$ASSETS_BASE_DIR" <<'PY'
import json
import os
from pathlib import Path
import sys

from openpi.shared import normalize
from openpi.training.config import get_config

config_name, assets_base = sys.argv[1:]
config = get_config(config_name)
repo_id = config.data.repo_id
dataset = Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
info = dataset / "meta/info.json"
if not info.is_file():
    raise SystemExit(
        f"Local dataset metadata is missing: {info}\n"
        "Place the dataset in VLA/.local/datasets/<repo_id>, or set HF_LEROBOT_HOME to its parent."
    )
json.loads(info.read_text())
asset_id = config.data.assets.asset_id or repo_id
assets = Path(config.data.assets.assets_dir or Path(assets_base) / config_name) / asset_id
if not (assets / "norm_stats.json").is_file():
    raise SystemExit(
        f"Normalization statistics are missing: {assets / 'norm_stats.json'}\n"
        "Copy the matching dataset statistics here, or set OPENPI_ASSETS_BASE_DIR to the config assets parent."
    )
stats = normalize.load(assets)
if not {"state", "actions"}.issubset(stats):
    raise SystemExit(f"Normalization statistics must contain state and actions: {assets}")
print(f"dataset={dataset}\nassets={assets}", flush=True)
PY

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Training is already running: pid=$old_pid" >&2
        echo "log=$LOG_FILE" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

if [[ "$MODE" == "fresh" && -e "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory already exists: $CHECKPOINT_DIR" >&2
    echo "Choose another run name, or use 'overwrite' explicitly." >&2
    exit 2
fi

extra_args=(--checkpoint-base-dir="$CHECKPOINT_ROOT" --assets-base-dir="$ASSETS_BASE_DIR")
case "$MODE" in
    resume)
        if [[ -e "$CHECKPOINT_DIR" ]]; then
            extra_args+=(--resume)
        fi
        ;;
    overwrite)
        extra_args+=(--overwrite)
        ;;
    fresh)
        # The directory was checked above and does not exist, so train.py will
        # create it without needing --overwrite or --resume.
        ;;
esac

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1

nohup uv run --no-sync scripts/train.py "$CONFIG_NAME" \
    --exp-name="$RUN_NAME" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --save-interval="$SAVE_INTERVAL" \
    --log-interval="$LOG_INTERVAL" \
    --num-workers="$NUM_WORKERS" \
    --no-wandb-enabled \
    "${extra_args[@]}" \
    > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
printf 'started pid=%s\nconfig=%s\nrun=%s\nlog=%s\ncheckpoint=%s\n' \
    "$(cat "$PID_FILE")" "$CONFIG_NAME" "$RUN_NAME" "$LOG_FILE" "$CHECKPOINT_DIR"
