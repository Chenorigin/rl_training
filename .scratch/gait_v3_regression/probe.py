from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for p in [Path('logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt'),*sorted(Path('logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-08_16-51-12_gait_v3').glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))[::5]]:
 d=torch.load(p,map_location='cpu',weights_only=False);s=d['actor_state_dict']; print(p.name,[(k,v.flatten().tolist()) for k,v in s.items() if 'std' in k])
e=EventAccumulator('logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-08_16-51-12_gait_v3');e.Reload()
for t in e.Tags()['scalars']:
 if 'Loss' in t or 'Policy' in t:
  s=e.Scalars(t);print(t,[(v.step,round(v.value,5)) for v in [s[0],s[len(s)//2],s[-1]]])
