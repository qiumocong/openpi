#!/usr/bin/env bash
set -euo pipefail

# Run from the OpenPI repository root, regardless of the caller's directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage: ./run_pi05_training_background.sh [run_name] [resume|overwrite|fresh]
CONFIG_NAME="${OPENPI_CONFIG:-pi05_ur7e_lora}"
RUN_NAME="${1:-dataset0924_50stp}"
MODE="${2:-overwrite}"

LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/${CONFIG_NAME}_${RUN_NAME}.log"
PID_FILE="$LOG_DIR/${CONFIG_NAME}_${RUN_NAME}.pid"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/$CONFIG_NAME/$RUN_NAME"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"

case "$MODE" in
    resume|overwrite|fresh) ;;
    *)
        echo "Usage: $0 [run_name] [resume|overwrite|fresh]" >&2
        exit 2
        ;;
esac

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

if [[ "$MODE" == "fresh" && -e "$CHECKPOINT_DIR" && \
      -n "$(find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Checkpoint directory is not empty: $CHECKPOINT_DIR" >&2
    echo "Use 'overwrite' to delete this experiment, or choose another run name." >&2
    exit 2
fi

mkdir -p "$CHECKPOINT_DIR"

extra_args=()
case "$MODE" in
    resume)
        if [[ -n "$(find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            extra_args+=(--resume)
        fi
        ;;
    overwrite)
        extra_args+=(--overwrite)
        ;;
esac

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT}"
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
