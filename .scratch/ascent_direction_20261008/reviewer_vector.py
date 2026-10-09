import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
import numpy as np,torch
from check_ascent_direction import load
from check_stair_rewards import Entity,make_env,Scene
ns=load();torch.set_num_threads(1);N=8
params=dict(asset_cfg=Entity('robot'),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
rng=np.random.default_rng(8753)
def new(n):
    e=make_env();e.num_envs=n
    for item in e.scene.values():
        for k,v in vars(item.data).items():
            if isinstance(v,torch.Tensor):setattr(item.data,k,v.repeat((n,)+(1,)*(v.ndim-1)))
    e.scene['robot'].data.root_lin_vel_w=torch.zeros(n,3)
    e.episode_length_buf=e.episode_length_buf.repeat(n)
    cmd=torch.tensor([[.5,0,0]]).repeat(n,1)
    e.command_manager.get_command=lambda _:cmd
    return e
bat=new(N);singles=[new(1) for _ in range(N)]
peak=0.
for t in range(120):
    yaw=torch.tensor(rng.uniform(-2,2,N),dtype=torch.float32)
    cmds=torch.tensor(rng.uniform(-.7,.7,(N,3)),dtype=torch.float32)
    cmds[:,0]=torch.tensor(rng.choice([0.,.2,.5,-.5],N))
    shift=torch.tensor(rng.uniform(-.015,.015,(N,2)),dtype=torch.float32)
    velocity=torch.tensor(rng.uniform(-.5,.5,(N,2)),dtype=torch.float32)
    seen=torch.tensor(rng.uniform(0,1,(N,2))>.5)
    upper=torch.tensor(rng.choice([.15,.30,.45],(N,2)),dtype=torch.float32)
    reset=torch.tensor(rng.uniform(0,1,N)<.08)
    q=torch.zeros(N,4);q[:,0]=torch.cos(yaw/2);q[:,3]=torch.sin(yaw/2)
    for i,e in enumerate([bat]+singles):
        idx=slice(None) if i==0 else slice(i-1,i)
        e.common_step_counter+=1;e.episode_length_buf+=1;e.episode_length_buf[reset[idx]]=0
        e.scene['robot'].data.root_pos_w[:,:2]+=shift[idx]
        e.scene['robot'].data.root_lin_vel_w[:,:2]=velocity[idx]
        e.scene['robot'].data.root_quat_w=q[idx].clone()
        e.scene['height_scanner'].data.quat_w=q[idx].clone()
        e.command_manager.get_command('base_velocity')[:]=cmds[idx]
        e._m20_stair_scan_context=dict(counter=e.common_step_counter,sensor_name='height_scanner',context=dict(ascending_edges_valid=seen[idx],ascending_edges_upper_z=upper[idx],ascending_edges_x=torch.tensor([.15,.35]),down_gate=torch.zeros(e.num_envs)))
        ns['_ascent_direction_context'](e,**params)
    for k in ('cost','active','cross_track','lateral_speed','heading_error'):
        a=bat._m20_ascent_direction_cache[k]
        b=torch.cat([e._m20_ascent_direction_cache[k] for e in singles])
        assert torch.equal(a,b),(t,k,a,b)
    peak=max(peak,float(bat._m20_ascent_direction_cache['cost'].max()))
print(json.dumps({'parallel_envs':N,'ticks':120,'batched_equals_individual':True,'max_cost':peak}))
