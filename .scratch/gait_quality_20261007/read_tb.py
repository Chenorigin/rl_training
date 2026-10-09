"""Bounded TensorBoard tail extraction, with CRC checked records; no training writes."""
import json, struct
from collections import deque, defaultdict
from pathlib import Path
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

root = Path.cwd()
runs = {
    'initial': ('2026-10-02_00-12-17', 80000),
    'intermediate': ('2026-10-03_15-12-45_gait_resume', 149999),
    'latest': ('2026-10-06_22-06-56_stair_resume', 196600),
}
result = {}
for label, (run, cutoff) in runs.items():
    records = deque(maxlen=250000)
    parsed = defaultdict(dict)
    paths = sorted((root/'logs/rsl_rl/deeprobotics_m20_stair_teacher'/run).glob('events*'))
    count = 0
    for path in paths:
        with path.open('rb') as f:
            size = path.stat().st_size
            while f.tell()+12 <= size:
                start = f.tell()
                header = f.read(12)
                length, crc = struct.unpack('<QI', header)
                assert masked_crc32c(header[:8]) == crc, (path, start)
                if start + 16 + length > size: break  # live writer's incomplete tail
                records.append((path, start+12, length))
                f.seek(length+4, 1)
                count += 1
    handles = {path:path.open('rb') for path in paths}
    for path, offset, length in records:
        f = handles[path]; f.seek(offset)
        payload=f.read(length); crc=struct.unpack('<I',f.read(4))[0]
        assert masked_crc32c(payload) == crc, (path, offset)
        event=Event.FromString(payload)
        if event.step > cutoff: continue
        for value in event.summary.value:
            tag=value.tag
            if any(k in tag for k in ('Episode_Reward/', 'Curriculum/gait', 'Train/mean_reward', 'Train/mean_episode_length')):
                if value.HasField('simple_value'):
                    parsed[tag][event.step]=float(value.simple_value)
    for f in handles.values(): f.close()
    selected={}
    for tag, values in parsed.items():
        points=sorted(values.items())[-500:]
        selected[tag]={'count':len(points),'first_step':points[0][0],'last_step':points[-1][0],
                       'mean':float(np.mean([v for _,v in points])), 'last':points[-1][1]}
    result[label]={'cutoff':cutoff,'records_indexed':count,'records_parsed':len(records),'tags':selected}
    print(label, 'tags', len(selected), flush=True)
(root/'docs/gait_quality_20261007/tensorboard_rewards.json').write_text(json.dumps(result,indent=2)+'\n')
