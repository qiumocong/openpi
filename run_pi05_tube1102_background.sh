#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$OPENPI_ROOT/../.." && pwd)"
RESOURCE_ROOT="$VLA_ROOT/.local"
DATASET_NAME="dataset_tube"
CONFIG_NAME="pi05_ur7e_tube1102_lora"
RUN_NAME="${1:-tube1102_50stp}"
MODE="${2:-overwrite}"
case "$MODE" in
    resume|overwrite|fresh) ;;
    *) echo "Usage: $0 [run_name] [resume|overwrite|fresh]" >&2; exit 2 ;;
esac

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$RESOURCE_ROOT/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

DATASET_PATH="$HF_LEROBOT_HOME/$DATASET_NAME"
SOURCE_DATASET="$VLA_ROOT/$DATASET_NAME"
if [[ ! -f "$SOURCE_DATASET/meta/info.json" ]]; then
    echo "Dataset metadata is missing: $SOURCE_DATASET/meta/info.json" >&2
    exit 1
fi
mkdir -p "$(dirname "$DATASET_PATH")"
if [[ -L "$DATASET_PATH" ]]; then
    rm "$DATASET_PATH"
fi
mkdir -p "$DATASET_PATH/meta"
if [[ -L "$DATASET_PATH/data" ]]; then
    rm "$DATASET_PATH/data"
fi
mkdir -p "$DATASET_PATH/data"
if [[ ! -e "$DATASET_PATH/videos" ]]; then
    ln -s "$SOURCE_DATASET/videos" "$DATASET_PATH/videos"
fi

cd "$OPENPI_ROOT"
uv run --no-sync python - "$SOURCE_DATASET" "$DATASET_PATH" <<'PY'
import json
import hashlib
from pathlib import Path
import sys

import pyarrow.parquet as pq

source, target = map(Path, sys.argv[1:])
meta = target / "meta"
fingerprint = hashlib.sha256()
for path in sorted([source / 'meta/info.json', source / 'meta/stats.json',
                    *source.glob('data/chunk-*/*.parquet'),
                    *source.glob('meta/episodes/chunk-*/*.parquet')]):
    fingerprint.update(path.relative_to(source).as_posix().encode())
    fingerprint.update(path.read_bytes())
info = json.loads((source / "meta/info.json").read_text())
info["data_path"] = "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4"
(meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
(meta / "stats.json").write_bytes((source / "meta/stats.json").read_bytes())

for path in sorted((source / "data").glob("chunk-*/*.parquet")):
    output = target / path.relative_to(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"huggingface", None)
    pq.write_table(table.replace_schema_metadata(metadata), output, compression="zstd")

tasks = pq.read_table(source / "meta/tasks.parquet").to_pylist()
with (meta / "tasks.jsonl").open("w") as stream:
    for item in tasks:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")

episodes = []
episode_root = source / "meta/episodes"
for path in sorted(episode_root.glob("chunk-*/*.parquet")):
    episodes.extend(pq.read_table(path).to_pylist())
episodes.sort(key=lambda item: item["episode_index"])
with (meta / "episodes.jsonl").open("w") as stream:
    for item in episodes:
        stream.write(json.dumps({key: item[key] for key in ("episode_index", "tasks", "length")}, ensure_ascii=False) + "\n")
with (meta / "episodes_stats.jsonl").open("w") as stream:
    for item in episodes:
        stats = {}
        for key, value in item.items():
            if key.startswith("stats/"):
                _, feature, metric = key.split("/", 2)
                stats.setdefault(feature, {})[metric] = value
        stream.write(json.dumps({"episode_index": item["episode_index"], "stats": stats}) + "\n")

if len(episodes) != info["total_episodes"]:
    raise SystemExit(f"Episode metadata count mismatch: {len(episodes)} != {info['total_episodes']}")
if sum(item['length'] for item in episodes) != info['total_frames']:
    raise SystemExit('Episode frame counts do not match dataset metadata')
(target / '.dataset_fingerprint').write_text(fingerprint.hexdigest() + '\n')
print(f"Prepared {len(episodes)} episodes and {info['total_frames']} frames at {target}")
PY

ASSETS_BASE_DIR="$RESOURCE_ROOT/openpi/assets"
NORM_STATS="$ASSETS_BASE_DIR/$CONFIG_NAME/$DATASET_NAME/norm_stats.json"
FINGERPRINT="$(cat "$DATASET_PATH/.dataset_fingerprint")"
STATS_FINGERPRINT="$(cat "$(dirname "$NORM_STATS")/.dataset_fingerprint" 2>/dev/null || true)"
if [[ ! -f "$NORM_STATS" || "$STATS_FINGERPRINT" != "$FINGERPRINT" ]]; then
    echo "Dataset changed or statistics are missing; recomputing OpenPI normalization."
    cd "$OPENPI_ROOT"
    uv run --no-sync python scripts/compute_norm_stats.py \
        --config-name "$CONFIG_NAME"
    COMPUTED_STATS="$OPENPI_ROOT/assets/$CONFIG_NAME/$DATASET_NAME/norm_stats.json"
    if [[ ! -f "$COMPUTED_STATS" ]]; then
        echo "OpenPI did not produce normalization statistics: $COMPUTED_STATS" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$NORM_STATS")"
    cp "$COMPUTED_STATS" "$NORM_STATS.tmp"
    mv "$NORM_STATS.tmp" "$NORM_STATS"
    printf '%s\n' "$FINGERPRINT" > "$(dirname "$NORM_STATS")/.dataset_fingerprint"
    rm "$COMPUTED_STATS"
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    printf 'dataset=%s\nnorm_stats=%s\nconfig=%s\n' "$DATASET_PATH" "$NORM_STATS" "$CONFIG_NAME"
    exit 0
fi

cd "$OPENPI_ROOT"
OPENPI_CONFIG="$CONFIG_NAME" bash run_pi05_training_background.sh "$RUN_NAME" "$MODE"
