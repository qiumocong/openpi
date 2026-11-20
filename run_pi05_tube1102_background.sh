#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$OPENPI_ROOT/../.." && pwd)"
RESOURCE_ROOT="$VLA_ROOT/.local"
DATASET_NAME="${DATASET_NAME:-dataset_tube}"
CONFIG_NAME="${CONFIG_NAME:-pi05_ur7e_tube1102_lora}"
RUN_NAME="${1:-tube1102_50stp}"
MODE="${2:-fresh}"
case "$MODE" in
    resume|overwrite|fresh) ;;
    *) echo "Usage: $0 [run_name] [resume|overwrite|fresh]" >&2; exit 2 ;;
esac

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$RESOURCE_ROOT/datasets_openpi_v21}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

DATASET_PATH="$HF_LEROBOT_HOME/$DATASET_NAME"
SOURCE_DATASET="${SOURCE_DATASET:-$RESOURCE_ROOT/datasets/$DATASET_NAME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$OPENPI_ROOT"
uv run --no-sync python scripts/prepare_lerobot_v21.py \
    --source "$SOURCE_DATASET" --target "$DATASET_PATH"

ASSETS_BASE_DIR="$RESOURCE_ROOT/openpi/assets"
NORM_STATS="$ASSETS_BASE_DIR/$CONFIG_NAME/$DATASET_NAME/norm_stats.json"
NORM_STATS_SCRIPT="${NORM_STATS_SCRIPT:-scripts/compute_norm_stats.py}"
FINGERPRINT="$(cat "$DATASET_PATH/.dataset_fingerprint"):$(sha256sum "$NORM_STATS_SCRIPT" | cut -d' ' -f1)"
STATS_FINGERPRINT="$(cat "$(dirname "$NORM_STATS")/.dataset_fingerprint" 2>/dev/null || true)"
if [[ ! -f "$NORM_STATS" || "$STATS_FINGERPRINT" != "$FINGERPRINT" ]]; then
    echo "Dataset changed or statistics are missing; recomputing OpenPI normalization."
    cd "$OPENPI_ROOT"
    JAX_PLATFORMS=cpu uv run --no-sync python "$NORM_STATS_SCRIPT" \
        --config-name "$CONFIG_NAME"
    COMPUTED_STATS="$OPENPI_ROOT/assets/$CONFIG_NAME/$DATASET_NAME/norm_stats.json"
    if [[ ! -f "$COMPUTED_STATS" ]]; then
        echo "OpenPI did not produce normalization statistics: $COMPUTED_STATS" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$NORM_STATS")"
    cp "$COMPUTED_STATS" "$NORM_STATS.tmp"
    mv "$NORM_STATS.tmp" "$NORM_STATS"
    if [[ -f "$(dirname "$COMPUTED_STATS")/norm_stats_report.json" ]]; then
        cp "$(dirname "$COMPUTED_STATS")/norm_stats_report.json" "$(dirname "$NORM_STATS")/norm_stats_report.json"
    fi
    printf '%s\n' "$FINGERPRINT" > "$(dirname "$NORM_STATS")/.dataset_fingerprint"
    rm "$COMPUTED_STATS"
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    printf 'dataset=%s\nnorm_stats=%s\nconfig=%s\n' "$DATASET_PATH" "$NORM_STATS" "$CONFIG_NAME"
    exit 0
fi

cd "$OPENPI_ROOT"
OPENPI_CONFIG="$CONFIG_NAME" bash run_pi05_training_background.sh "$RUN_NAME" "$MODE"
