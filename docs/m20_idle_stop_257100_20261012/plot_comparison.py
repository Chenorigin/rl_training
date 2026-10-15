from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for label,name,color in [('baseline','Original 257100','#6b7280'),('distilled_round5','Idle-stop fitted','#047857')]:
 for case,col,start in [('零命令',0,10),('平地启停',1,14.02)]:
  f=np.load(B/'evaluations'/label/(case+'_full.npz'));t=f['time'];xy=f['qpos'][:,:2];i=np.searchsorted(t,start);w,x,y,z=f['qpos'][:,3:7].T;yaw=np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
  displacement=np.linalg.norm(xy[i:]-xy[i],axis=1)*100
  axes[0,col].plot(t[i:]-t[i],displacement,label=name,color=color,lw=2)
  axes[1,col].plot(t[i:]-t[i],yaw[i:]-yaw[i],label=name,color=color,lw=2)
for col,title,limit in [(0,'Stationary: t=10-36 s',5),(1,'Release W: t=14.02-36 s',10)]:
 axes[0,col].set_title(title);axes[0,col].axhline(limit,color='#b91c1c',ls='--',lw=1,label=f'{limit} cm limit');axes[0,col].set_ylabel('Displacement from start (cm)');axes[0,col].legend(frameon=False)
 axes[1,col].set_ylabel('Yaw change (degrees)');axes[1,col].set_xlabel('Elapsed time (s)')
for ax in axes.flat:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Original MuJoCo interactive_rollout | 50 Hz policy / 200 Hz physics\nSame model, gains and terrain; no runtime action clamp',fontsize=12)
fig.savefig(B/'idle_stop_comparison.png',dpi=180);plt.close(fig)
