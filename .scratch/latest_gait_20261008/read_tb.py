"""Bounded TensorBoard tail extraction, with CRC checked records; no training writes."""
import json, struct
from collections import deque, defaultdict
from pathlib import Path
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

root = Path.cwd()
runs = {'baseline': ('2026-10-06_22-06-56_stair_resume',196600), 'latest': ('2026-10-07_23-47-00_gait_safety_v2',22000)}
result = {}
for label, (run, cutoff) in runs.items():
    records = deque(maxlen=3000000 if label=="latest" else 1000000)
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
            if True:
                if value.HasField('simple_value'):
                    parsed[tag][event.step]=float(value.simple_value)
    for f in handles.values(): f.close()
    selected={}
    for tag, values in parsed.items():
        points=sorted(values.items())[-500:]
        selected[tag]={'min_point':list(min(values.items(),key=lambda p:p[1])), 'max_point':list(max(values.items(),key=lambda p:p[1])), 'bins_1000':[[i,float(np.mean([v for st,v in values.items() if i<=st<i+1000]))] for i in range(min(values)//1000*1000,max(values)+1,1000)], 'trend':[[int(st),float(val)] for st,val in sorted(values.items())[::max(1,len(values)//200)]], 'count':len(points),'first_step':points[0][0],'last_step':points[-1][0],
                       'mean':float(np.mean([v for _,v in points])), 'last':points[-1][1]}
    result[label]={'cutoff':cutoff,'records_indexed':count,'records_parsed':len(records),'tags':selected}
    print(label, 'tags', len(selected), flush=True)
(root/'docs/latest_gait_20261008/tensorboard.json').write_text(json.dumps(result,indent=2)+'\n')
