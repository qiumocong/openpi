"""Exact numeric normalization for the joint-space UR7E config.

Uses all frames and all horizon positions, clamping at each episode boundary
exactly as LeRobot does. Applies the config's actual numeric transforms, checks
them against normal video-loading LeRobot samples, and computes exact quantiles
in float64. Images have no effect on these two supported numeric transforms.
Unknown transform pipelines are rejected; use compute_norm_stats.py for those.
"""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi import transforms
from openpi.policies.ur7e_policy import UR7EInputs
from openpi.shared import normalize
from openpi.training import config as configs, data_loader


def main(config_name: str):
    cfg = configs.get_config(config_name)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    if dc.rlds_data_dir is not None or tuple(dc.action_sequence_keys) != ('action',):
        raise ValueError('Only local UR7E action sequences are supported')
    if [type(t) for t in dc.repack_transforms.inputs] != [transforms.RepackTransform] or [type(t) for t in dc.data_transforms.inputs] != [transforms.DeltaActions, UR7EInputs]:
        raise ValueError('Unknown pipeline: run scripts/compute_norm_stats.py instead')
    fn = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
    root = Path(os.environ['HF_LEROBOT_HOME']) / dc.repo_id
    info = json.loads((root/'meta/info.json').read_text())
    if info['codebase_version'] != 'v2.1':
        raise ValueError('Run prepare_lerobot_v21.py first')
    episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
    regular = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    samples = data_loader.TransformedDataset(regular,[*dc.repack_transforms.inputs,*dc.data_transforms.inputs])
    pieces = {'state':[], 'actions':[]}
    horizon = cfg.model.action_horizon
    checked, offset = 0, 0
    for ep in episodes:
        index = ep['episode_index']
        path = root / info['data_path'].format(episode_chunk=index//info['chunks_size'], episode_index=index)
        table = pq.read_table(path)
        state, action = [np.asarray(table[k].to_pylist(),dtype=np.float32) for k in ('observation.state','action')]
        n = len(state)
        if n != ep['length'] or state.shape != (n,7) or action.shape != state.shape:
            raise ValueError('Unexpected UR7E numeric layout')
        future = np.minimum(np.arange(n)[:,None] + np.arange(horizon)[None,:], n-1)
        raw = {'observation.state':state, 'action':action[future], 'observation.images.camera_0':np.zeros((3,1,1),np.float32), 'observation.images.camera_1':np.zeros((3,1,1),np.float32), 'prompt':'UR7E pick and place'}
        batch = fn(raw)
        # Exercise every episode, including tail padding, through real video decoding.
        for frame in sorted({0,n//2,n-1}):
            original = samples[offset+frame]
            for key in pieces:
                np.testing.assert_array_equal(batch[key][frame],np.asarray(original[key]))
            checked += 1
        pieces['state'].append(batch['state'])
        pieces['actions'].append(batch['actions'].reshape(-1,7))
        offset += n
        if (index+1)%25 == 0:
            print(f'Numeric stats verified: {index+1}/{len(episodes)} episodes',flush=True)
    if offset != info['total_frames'] or len(regular) != offset:
        raise ValueError('Incomplete numeric coverage')
    stats={}
    for key, parts in pieces.items():
        values=np.concatenate(parts).astype(np.float64)
        if not np.isfinite(values).all():raise ValueError(f'Nonfinite {key}')
        q01,q99=np.quantile(values,[.01,.99],axis=0)
        stats[key]=normalize.NormStats(mean=values.mean(axis=0),std=values.std(axis=0),q01=q01,q99=q99)
    out=cfg.assets_dirs/dc.repo_id
    normalize.save(out,stats)
    report={'algorithm':'exact-float64-quantiles-v1','frames':offset,'episodes':len(episodes),'action_horizon':horizon,'action_vectors':offset*horizon,'regular_loader_comparison_samples':checked,'all_comparisons_exact':True,'includes_final_partial_batch':True,'dataset_fingerprint':(root/'.dataset_fingerprint').read_text().strip()}
    (out/'norm_stats_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Writing stats to: {out}\n{json.dumps(report)}',flush=True)


if __name__=='__main__':
    tyro.cli(main)
