from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=Path('docs/gait_v3_regression')
e=EventAccumulator('logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-08_16-51-12_gait_v3',size_guidance={'scalars':0});e.Reload()
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,t in zip(axs.flat,['Train/mean_reward','Metrics/base_velocity/error_vel_xy','Curriculum/terrain_level_pyramid_stairs','Curriculum/gait_quality/front_fold_support_fraction']):
 s=e.Scalars(t);x=np.array([v.step for v in s]);y=np.array([v.value for v in s]);ax.plot(x,y,alpha=.2);win=50;ax.plot(x[win-1:],np.convolve(y,np.ones(win)/win,'valid'));ax.set_title(t);ax.set_xlabel('PPO iteration');ax.grid(alpha=.2)
fig.savefig(p/'tensorboard_trends.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,case in zip(axs,['ascent','flat']):
 for label in ['baseline','latest','guarded_smoke']:
  d=np.load(p/label/(case+'.npz'));pos=d['root_pos'];ax.plot(pos[:,0],pos[:,1],label=label)
 ax.set_title(case+' actual physics trajectory');ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)');ax.grid(alpha=.2);ax.legend()
fig.savefig(p/'mujoco_trajectories.png',dpi=140)
