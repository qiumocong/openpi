"""Regression: video/data file addresses may differ from episode numbers."""
import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_lerobot_v21 import prepare, sha256


class PrepareTest(unittest.TestCase):
    def test_shuffled_source_files_and_immutable_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); src=root/'source'; dst=root/'converted'
            (src/'meta/episodes/chunk-000').mkdir(parents=True)
            (src/'data/chunk-000').mkdir(parents=True)
            (src/'videos/camera/chunk-000').mkdir(parents=True)
            info={'codebase_version':'v3.0','fps':30,'total_episodes':2,'total_frames':6,'chunks_size':1000,'features':{'camera':{'dtype':'video'}},'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet','video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
            (src/'meta/info.json').write_text(json.dumps(info))
            (src/'meta/stats.json').write_text('{}')
            pq.write_table(pa.Table.from_pylist([{'task_index':0,'task':'test'}]),src/'meta/tasks.parquet')
            rows=[]
            for ep,fi,vi in [(0,17,8),(1,4,2)]:
                table=pa.table({'episode_index':[ep]*3,'frame_index':[0,1,2],'index':list(range(ep*3,ep*3+3)),'timestamp':[0.,1/30,2/30],'observation.state':[[ep]*7]*3,'action':[[ep+2]*7]*3})
                pq.write_table(table,src/f'data/chunk-000/file-{fi:03d}.parquet')
                video=src/f'videos/camera/chunk-000/file-{vi:03d}.mp4'
                with av.open(str(video),'w') as out:
                    stream=out.add_stream('libx264',rate=30); stream.width=16;stream.height=16;stream.pix_fmt='yuv420p'
                    for _ in range(3):
                        frame=av.VideoFrame.from_ndarray(np.full((16,16,3),ep*180,dtype=np.uint8),format='rgb24')
                        for packet in stream.encode(frame):out.mux(packet)
                    for packet in stream.encode():out.mux(packet)
                rows.append({'episode_index':ep,'tasks':['test'],'length':3,'data/chunk_index':0,'data/file_index':fi,'videos/camera/chunk_index':0,'videos/camera/file_index':vi,'videos/camera/from_timestamp':0.0,'videos/camera/to_timestamp':.1})
            pq.write_table(pa.Table.from_pylist(rows),src/'meta/episodes/chunk-000/file-000.parquet')
            before={str(p.relative_to(src)):sha256(p) for p in src.rglob('*') if p.is_file()}
            prepare(src,dst)
            for ep,vi in [(0,8),(1,2)]:
                self.assertEqual(sha256(dst/f'videos/chunk-000/camera/episode_{ep:06d}.mp4'),sha256(src/f'videos/camera/chunk-000/file-{vi:03d}.mp4'))
                self.assertEqual(pq.read_table(dst/f'data/chunk-000/episode_{ep:06d}.parquet')['observation.state'].to_pylist(),[[ep]*7]*3)
            self.assertEqual(json.loads((dst/'meta/info.json').read_text())['codebase_version'],'v2.1')
            self.assertEqual(before,{str(p.relative_to(src)):sha256(p) for p in src.rglob('*') if p.is_file()})
            prepare(src,dst)
            rows[0]['videos/camera/from_timestamp']=1.0
            pq.write_table(pa.Table.from_pylist(rows),src/'meta/episodes/chunk-000/file-000.parquet')
            with self.assertRaisesRegex(ValueError,'trimming'):
                prepare(src,dst)


if __name__=='__main__':
    unittest.main()
