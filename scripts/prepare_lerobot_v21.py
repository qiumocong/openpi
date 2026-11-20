"""Create the local v2.1 view required by this OpenPI checkout from v3 metadata.

Only independent, zero-offset episode videos are supported; ambiguous/shared
video segments fail explicitly. Data/video addresses come from metadata, never
from an assumed relationship between episode number and source file number.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CONVERTER_VERSION = 'metadata-addressed-v21-1'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare(source: Path, target: Path):
    source, target = source.resolve(), target.absolute()
    if target.resolve() == source or source in target.resolve().parents or target.resolve() in source.parents:
        raise ValueError('Source and output must be separate, non-nested directories')
    info = json.loads((source / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0':
        raise ValueError('Expected a v3.0 source dataset')
    cameras = [k for k,v in info['features'].items() if v['dtype']=='video']
    rows = []
    dependencies = {source / 'meta/info.json', source / 'meta/stats.json', source / 'meta/tasks.parquet'}
    for path in sorted((source / 'meta/episodes').glob('chunk-*/*.parquet')):
        dependencies.add(path)
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda r:r['episode_index'])
    if [r['episode_index'] for r in rows] != list(range(info['total_episodes'])):
        raise ValueError('Episode metadata is incomplete, duplicated or non-contiguous')
    if sum(r['length'] for r in rows) != info['total_frames']:
        raise ValueError('Metadata frame count mismatch')
    addresses = []
    used_videos = set()
    for row in rows:
        data = source / info['data_path'].format(chunk_index=row['data/chunk_index'],file_index=row['data/file_index'])
        videos = {}
        for key in cameras:
            if abs(row[f'videos/{key}/from_timestamp']) > 1e-7:
                raise ValueError(f'Video requires explicit trimming: episode {row["episode_index"]}, {key}')
            if abs(row[f'videos/{key}/to_timestamp'] - row['length'] / info['fps']) > 1e-3:
                raise ValueError(f'Video metadata duration mismatch: episode {row["episode_index"]}, {key}')
            video = source / info['video_path'].format(video_key=key,chunk_index=row[f'videos/{key}/chunk_index'],file_index=row[f'videos/{key}/file_index'])
            if video in used_videos:
                raise ValueError(f'Shared video requires explicit trimming: {video}')
            used_videos.add(video)
            videos[key] = video
            dependencies.add(video)
        dependencies.add(data)
        addresses.append((data,videos))
    content = {str(p.relative_to(source)):sha256(p) for p in sorted(dependencies)}
    fingerprint = hashlib.sha256(json.dumps([CONVERTER_VERSION,content],sort_keys=True).encode()).hexdigest()
    marker = target / '.dataset_fingerprint'
    if marker.is_file() and marker.read_text().strip()==fingerprint and (target/'conversion.json').is_file():
        # Cached views must still point at live source videos.
        cached_videos = list((target/'videos').rglob('*.mp4'))
        cached_data = list((target/'data').rglob('*.parquet'))
        if len(cached_videos)==len(rows)*len(cameras) and len(cached_data)==len(rows) and all(p.is_file() for p in cached_videos):
            print(f'Unchanged dataset: {target}',flush=True)
            return
    if target.exists() and not (target/'conversion.json').is_file():
        raise FileExistsError(f'Refusing to replace an unmarked directory: {target}')
    stage = target.with_name('.'+target.name+'.converting-'+uuid.uuid4().hex[:8])
    stage.mkdir(parents=True)
    old_info=copy.deepcopy(info)
    old_info.update(codebase_version='v2.1',data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',total_videos=len(rows)*len(cameras),total_chunks=(len(rows)+info['chunks_size']-1)//info['chunks_size'])
    episodes, stats = [], []
    offset=0
    records=[]
    for row,(data_path,videos) in zip(rows,addresses,strict=True):
        ep=int(row['episode_index']); chunk=ep//info['chunks_size']; n=int(row['length'])
        table=pq.read_table(data_path)
        table=table.filter(pa.compute.equal(table['episode_index'],ep))
        if table.num_rows!=n or table['index'].to_pylist()!=list(range(offset,offset+n)) or table['frame_index'].to_pylist()!=list(range(n)):
            raise ValueError(f'Bad numeric indexing for episode {ep}')
        timestamps=np.asarray(table['timestamp'].to_pylist())
        if not np.allclose(timestamps,np.arange(n)/info['fps'],atol=1e-4,rtol=0):
            raise ValueError(f'Bad timestamps in episode {ep}')
        for key in ('observation.state','action'):
            if not np.isfinite(np.asarray(table[key].to_pylist())).all():
                raise ValueError(f'Nonfinite {key} in episode {ep}')
        output=stage/old_info['data_path'].format(episode_chunk=chunk,episode_index=ep)
        output.parent.mkdir(parents=True,exist_ok=True)
        schema_meta=dict(table.schema.metadata or {}); schema_meta.pop(b'huggingface',None)
        pq.write_table(table.replace_schema_metadata(schema_meta),output,compression='zstd')
        if not pq.read_table(output).equals(table,check_metadata=False):
            raise ValueError('Numeric values changed during conversion')
        for key,video in videos.items():
            with av.open(str(video)) as container:
                stream=container.streams.video[0]
                if stream.frames!=n or abs(float(stream.average_rate)-info['fps'])>.001:
                    raise ValueError(f'Video length/FPS mismatch: {video}')
            dest=stage/old_info['video_path'].format(episode_chunk=chunk,video_key=key,episode_index=ep)
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Relative links survive moving the VLA checkout as a unit.
            try:
                dest.symlink_to(os.path.relpath(video,dest.parent))
            except OSError:
                shutil.copy2(video,dest)
        episodes.append({k:row[k] for k in ('episode_index','tasks','length')})
        ep_stats={}
        for key,value in row.items():
            if key.startswith('stats/'):
                _,feature,metric=key.split('/',2)
                if feature in cameras and metric!='count' and np.asarray(value).size==3:
                    value=np.asarray(value).reshape(3,1,1).tolist()
                ep_stats.setdefault(feature,{})[metric]=value
        stats.append({'episode_index':ep,'stats':ep_stats})
        records.append({'episode_index':ep,'data':str(data_path.relative_to(source)),'videos':{k:str(p.relative_to(source)) for k,p in videos.items()}})
        offset+=n
    dump(stage/'meta/info.json',old_info)
    shutil.copy2(source/'meta/stats.json',stage/'meta/stats.json')
    for name,items in [('tasks',pq.read_table(source/'meta/tasks.parquet').to_pylist()),('episodes',episodes),('episodes_stats',stats)]:
        (stage/f'meta/{name}.jsonl').write_text(''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in items),encoding='utf-8')
    dump(stage/'conversion.json',{'converter':CONVERTER_VERSION,'source':str(source),'fingerprint':fingerprint,'content_sha256':content,'episodes':records})
    (stage/'.dataset_fingerprint').write_text(fingerprint+'\n')
    if target.exists():
        target.rename(target.with_name(target.name+'.backup-'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    stage.rename(target)
    print(f'Prepared {len(rows)} episodes / {offset} frames: {target}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    args=p.parse_args(); prepare(args.source,args.target)
