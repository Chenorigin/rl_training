"""Read-only telemetry through original deployment FSM with headless viewer."""
import argparse, ast, contextlib, json, math, sys
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deploy
from gait_metrics import GaitMetrics
p=argparse.ArgumentParser(); p.add_argument('--label',required=True); p.add_argument('--case',choices=['xml20','up30','down30','turn_q','turn_e'],required=True); p.add_argument('--video',action='store_true'); args=p.parse_args()
OUT=ROOT/'docs/latest_gait_20261008'/args.label/(args.case+'_continued'); OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'.scratch/latest_gait_20261008/checkpoints.json').read_text()); cp=Path(manifest[args.label]['copy'])
active=None; samples=[]; physics=[]; static_force=[]; writer=None; renderer=None; clocksteps=0; collision={}; reasons=[]
tuning=next(eval(compile(ast.Expression(n.value), '<TUNING>', 'eval'), {'np':np,'__builtins__':{}}) for n in ast.parse((ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TUNING' for t in n.targets))

class Contract(deploy.M20Contract):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        global active
        active=self; self.metrics=GaitMetrics(self); self.policy=False; self.done=False; self.command=np.zeros(3); self.action=np.zeros(16); self.substeps=0; self.contact_time=np.zeros(4); self.credited=set(); self.landings=[]; self.first_front_down=None; self.first_rear_down=None; self.started=False
        self.geom_data=mujoco.MjData(self.model); self.hips=self.metrics.hips; self.kneebodies=np.array([self.model.body(f'{leg}_knee').id for leg in ('fl','fr','hl','hr')])
    def reset(self,data,*a,**kw): self.data=data; return super().reset(data,*a,**kw)
    def observe(self,data,command,previous): self.policy=True; self.command=command.copy(); return super().observe(data,command,previous)
    def apply_pd(self,data,action): self.action=action.copy(); return super().apply_pd(data,action)

class Viewer:
    def __init__(self): self.cam=mujoco.MjvCamera()
    def lock(self): return contextlib.nullcontext()
    def is_running(self): return not active.done
    def sync(self): pass
    def close(self): pass
class Terminal(deploy.TerminalInput):
    def __init__(self,k): super().__init__(k); self.sent=set()
    def start(self): return True
    def close(self): pass
    def poll(self):
        t=active.data.time
        for when,key in ((.25,'z'),(5.8,'c')):
            if t>=when and key not in self.sent: self.keyboard.on_char(key); self.sent.add(key)
        if t>=6.5: self.keyboard.on_char('q' if args.case=='turn_q' else 'e' if args.case=='turn_e' else 'w')

real_step=mujoco.mj_step

def measure_step(model,d,*a,**kw):
    global writer,renderer
    real_step(model,d,*a,**kw)
    if active is None: return
    forces=active.metrics.forces(d)
    physical_data=d
    mujoco.mj_copyData(active.geom_data,model,d); mujoco.mj_kinematics(model,active.geom_data); d=active.geom_data
    if 5.4<d.time<5.8: static_force.append(float(forces[:,2].sum()))
    if not active.policy: return
    active.substeps+=1
    wheels=d.xpos[active.wheel_body_ids].copy(); R=d.xmat[active.base_id].reshape(3,3); hips=d.xpos[active.hips].copy()
    torque=d.actuator_force[active.actuator_ids].copy(); speed=d.qvel[active.qvel_ids].copy()
    power=np.abs(torque*speed).sum(); positive=np.maximum(torque*speed,0).sum()
    x=float(d.xpos[active.base_id,0]); y=float(d.xpos[active.base_id,1])
    start=active.terrain.parameters.get('start_x',[-100])[0]; goal=active.terrain.goal_x
    in_course=bool((np.abs(wheels[:,1])<=active.terrain.course_width/2).all())
    completed=bool(not args.case.startswith('turn') and (wheels[:,0]>=goal+.09).all() and in_course)
    phase=bool(d.time>=6.5 and (args.case.startswith('turn') or (active.command[0]>.1 and wheels[:,0].max()>=start-.10 and not completed)))
    # Physical-rate traces include force transients. Ground queries/landing state at 50Hz below.
    physics.append(dict(time=float(d.time),root_x=x,root_y=y,forces=forces,power=power,positive_power=positive,torque=torque,velocity=speed,phase=phase))
    for i in range(physical_data.ncon):
        ct=physical_data.contact[i]; bodies=model.geom_bodyid[[ct.geom1,ct.geom2]]
        if 0 not in bodies or bodies[0]==bodies[1]: continue
        body=int(bodies[1] if bodies[0]==0 else bodies[0])
        if body in active.wheel_body_ids: continue
        f=np.zeros(6); mujoco.mj_contactForce(model,physical_data,i,f); norm=float(np.linalg.norm(f[:3]))
        if norm<=1: continue
        name=model.body(body).name; item=collision.setdefault(name,dict(first_time=float(d.time),peak_n=0,samples=0)); item['peak_n']=max(norm,item['peak_n']); item['samples']+=1
    if active.substeps%4: return
    ground=np.array([active.ground_height(d,*v[:2]) for v in wheels]); clearance=wheels[:,2]-ground-.09
    supporting=(forces[:,2]>5)&(forces[:,2]>.5*np.linalg.norm(forces,axis=1))&(np.abs(clearance)<=.05)
    active.contact_time=np.where(supporting,active.contact_time+.02,0)
    stable=active.contact_time>=.06
    levels=np.rint(ground/.15).astype(int)
    if args.case in ('up30','xml20'):
        for leg in range(4):
            key=(leg,int(levels[leg]))
            if stable[leg] and 0<levels[leg]<int(active.terrain.parameters['step_count'][0]) and key not in active.credited:
                active.credited.add(key); active.landings.append(dict(time=float(d.time),leg=('fl','fr','hl','hr')[leg],level=int(levels[leg]),x=float(wheels[leg,0]),y=float(wheels[leg,1])))
    elif args.case=='down30':
        if (ground[:2]<1.2-.075).any() and active.first_front_down is None: active.first_front_down=float(d.time)
        if (ground[2:]<1.2-.075).any() and active.first_rear_down is None: active.first_rear_down=float(d.time)
    legvec=(wheels-hips)@R
    yaw=math.atan2(R[1,0],R[0,0]); heading=np.array([math.cos(yaw),math.sin(yaw),0])
    wx=(wheels-hips)@heading; wz=(wheels-hips)[:,2]
    tilt=np.degrees(np.arctan2(legvec[:,0],-legvec[:,2])); worldtilt=np.degrees(np.arctan2(wx,-wz))
    bodypts=np.array([[x,y,tuning['body_underside_z']] for x in tuning['front_body_x'] for y in tuning['front_body_y']])@R.T+d.xpos[active.base_id]
    bodygap=min(float(pt[2]-active.ground_height(d,*pt[:2])) for pt in bodypts)
    row=dict(body_gap=bodygap,command=active.command.copy(),yaw_velocity=float((R@d.qvel[3:6])[2]),wheel_clearance=clearance,time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),root_pos=d.xpos[active.base_id].copy(),wheel_pos=wheels,hip_pos=hips,ground=ground,forces=forces,
        knee=d.qpos[active.metrics.knees].copy(),extension=np.linalg.norm(wheels-hips,axis=1),support=supporting,contact_time=active.contact_time.copy(),
        tilt_body_deg=tilt,tilt_yaw_deg=worldtilt,leg_forward_body=legvec[:,0],pitch_deg=math.degrees(math.atan2(-R[2,0],math.hypot(R[2,1],R[2,2]))),phase=phase)
    samples.append(row)
    if args.video and len(samples)%5==0:
        if renderer is None: renderer=mujoco.Renderer(model,width=640,height=480)
        cam=mujoco.MjvCamera(); cam.lookat[:]=d.xpos[active.base_id]; cam.distance=3; cam.azimuth=100; cam.elevation=-15
        renderer.update_scene(d,camera=cam); renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        if writer is None: writer=imageio.get_writer(OUT/'closed_loop.mp4',fps=10,codec='libx264',quality=7)
        writer.append_data(renderer.render())
    if completed: reasons.append('all_wheels_crossed'); active.done=True
    elif phase and not in_course and 'left_course' not in reasons: reasons.append('left_course')

def sequence(events):
    pairs=[]
    for a,b in zip(events,events[1:]): pairs.append(dict(prev=a['leg']+str(a['level']),next=b['leg']+str(b['level']),correct=a['leg']!=b['leg'] and b['level']==a['level']+1,same_tread=a['level']==b['level']))
    return dict(events=events,checks=len(pairs),correct=sum(v['correct'] for v in pairs),rate=sum(v['correct'] for v in pairs)/len(pairs) if pairs else None,sequential_same_tread=sum(v['same_tread'] for v in pairs),transitions=pairs)
# Independent measurement positive/negative controls.
assert sequence([dict(leg='fl',level=1),dict(leg='fr',level=2),dict(leg='fl',level=3)])['rate']==1
assert sequence([dict(leg='fl',level=1),dict(leg='fr',level=1),dict(leg='fl',level=2),dict(leg='fr',level=2)])['sequential_same_tread']==2

deploy.M20Contract=Contract; deploy.TerminalInput=Terminal; deploy.launch_viewer=lambda *a,**kw:(Viewer(),[]); deploy.draw_height_scan=lambda *a,**kw:None
mujoco.mj_step=measure_step; deploy.time.sleep=lambda _:None; deploy.TELEOP_VELOCITIES['w']=(.7 if args.case=='xml20' else .5,0,0)
run=argparse.Namespace(model=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml'),checkpoint=cp,
    terrain='flat' if args.case.startswith('turn') else 'descent' if args.case=='down30' else 'ascent',terrain_xml=None,
    stair_height=None if args.case.startswith('turn') or args.case=='xml20' else .15,tread_depth=None if args.case.startswith('turn') or args.case=='xml20' else .30,
    stair_count=None if args.case.startswith('turn') or args.case=='xml20' else 8,stair_start=None,viewer=True,autoplay=False,
    steps=800 if args.case.startswith('turn') else 2200 if args.case=='xml20' else 1600,output=None,print_every=800,stop_on_fall=True)
try: result=deploy.interactive_rollout(run)
finally:
    if writer: writer.close()
    if renderer: renderer.close()
assert samples and physics
S={k:np.asarray([v[k] for v in samples]) for k in samples[0]}; P={k:np.asarray([v[k] for v in physics]) for k in physics[0]}
np.savez_compressed(OUT/'telemetry.npz',**S); np.savez_compressed(OUT/'physics.npz',**P)
mask=P['phase']; norms=np.linalg.norm(P['forces'][mask],axis=-1); loaded=norms[norms>5]
phase_duration=mask.sum()*.005; distance=float(P['root_x'][mask][-1]-P['root_x'][mask][0]) if mask.any() else 0
force_summary=dict(contact_samples=int(len(loaded)),mean_n=float(loaded.mean()) if loaded.size else None,p95_n=float(np.quantile(loaded,.95)) if loaded.size else None,p99_n=float(np.quantile(loaded,.99)) if loaded.size else None,max_n=float(loaded.max()) if loaded.size else None)
stair_ground=(S['ground'][:,:2]>.075)&(S['ground'][:,:2]<active.terrain.parameters.get('step_count',[0])[0]*.15-.075)
front_loaded=S['support'][:,:2]&S['phase'][:,None]&stair_ground
front_support=front_loaded&(S['contact_time'][:,:2]>=.20)
front_air=(np.linalg.norm(S['forces'][:,:2],axis=-1)<5)&S['phase'][:,None]&stair_ground
def stats(ext,knee,mask):
    return dict(samples=int(mask.sum()),extension_min_m=float(ext[mask].min()) if mask.any() else None,extension_p05_m=float(np.quantile(ext[mask],.05)) if mask.any() else None,knee_abs_max_deg=float(np.degrees(np.abs(knee[mask])).max()) if mask.any() else None,knee_abs_p95_deg=float(np.quantile(np.degrees(np.abs(knee[mask])),.95)) if mask.any() else None)
first=active.first_front_down; last=active.first_rear_down
entry=(S['time']>=first)&(S['time']<=last+1 if last is not None else np.ones(len(samples),bool)) if first is not None else np.zeros(len(samples),bool)
rear_support=S['support'][:,2:]&(S['contact_time'][:,2:]>=.12)&entry[:,None]
result.update(checkpoint_original=manifest[args.label]['source'],checkpoint_sha256=manifest[args.label]['sha256'],control='original prone/Z/C/FSM, CPU display headless',command=deploy.TELEOP_VELOCITIES['q' if args.case=='turn_q' else 'e' if args.case=='turn_e' else 'w'],stop_reasons=reasons,physics_sampling_hz=200,
    static_wheel_force_mean_n=float(np.mean(static_force)),robot_weight_n=float(active.model.body_mass.sum()*9.81),nonwheel_collisions=collision,
    phase_duration_s=phase_duration,phase_forward_distance_m=distance,force=force_summary,mechanical_power_mean_w=float(P['power'][mask].mean()) if mask.any() else None,
    absolute_mechanical_energy_j=float(P['power'][mask].sum()*.005),positive_mechanical_energy_j=float(P['positive_power'][mask].sum()*.005),mechanical_energy_j_per_m=float(P['power'][mask].sum()*.005/distance) if distance>0 else None,
    front_loaded=stats(S['extension'][:,:2],S['knee'][:,:2],front_loaded),
    body_gap_min_m=float(S['body_gap'][S['phase']].min()) if S['phase'].any() else None,body_gap_low_fraction=float((S['body_gap'][S['phase']]<tuning['body_gap_target']).mean()) if S['phase'].any() else None,
    front_support=stats(S['extension'][:,:2],S['knee'][:,:2],front_support),front_swing=stats(S['extension'][:,:2],S['knee'][:,:2],front_air),
    front_support_envelope_excess_fraction=float(((S['extension'][:,:2]<tuning['front_min_extension'])|(np.abs(S['knee'][:,:2])>tuning['front_max_knee']))[front_support].mean()) if front_support.any() else None,
    front_swing_envelope_excess_fraction=float(((S['extension'][:,:2]<tuning['front_swing_min_extension'])|(np.abs(S['knee'][:,:2])>tuning['front_swing_max_knee']))[front_air].mean()) if front_air.any() else None,
    front_sequence=sequence([e for e in active.landings if e['leg'] in ('fl','fr')]),rear_sequence=sequence([e for e in active.landings if e['leg'] in ('hl','hr')]),
    descent_entry=dict(first_front_down_s=first,first_rear_down_s=last,stable_rear_samples=int(rear_support.sum()),rear_tilt_body_p95_deg=float(np.quantile(S['tilt_body_deg'][:,2:][rear_support],.95)) if rear_support.any() else None,rear_tilt_yaw_p95_deg=float(np.quantile(S['tilt_yaw_deg'][:,2:][rear_support],.95)) if rear_support.any() else None,rear_forward_body_max_m=float(S['leg_forward_body'][:,2:][rear_support].max()) if rear_support.any() else None,pitch_min_deg=float(S['pitch_deg'][entry].min()) if entry.any() else None,rear_fold=stats(S['extension'][:,2:],S['knee'][:,2:],rear_support)))
turnmask=S['phase']
if args.case.startswith('turn'):
    result['turn']=dict(actual_yaw_mean_rad_s=float(S['yaw_velocity'][turnmask].mean()),yaw_mae_rad_s=float(np.abs(S['yaw_velocity'][turnmask]-S['command'][turnmask,2]).mean()),wheel_clearance_max_m=float(S['wheel_clearance'][turnmask].max()),translation_drift_m=float(np.linalg.norm(S['root_pos'][turnmask][-1,:2]-S['root_pos'][turnmask][0,:2])))
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('RESULT',json.dumps({k:result[k] for k in ('stop_reasons','fell_in_policy','phase_duration_s','force','mechanical_power_mean_w','front_support','front_swing','descent_entry')}),flush=True)
# Measurement tool scale check: stable stance support must match gravity.
assert abs(result['static_wheel_force_mean_n']/result['robot_weight_n']-1)<.15,result['static_wheel_force_mean_n']
