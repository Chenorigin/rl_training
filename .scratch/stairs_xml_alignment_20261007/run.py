"""Run the original FSM in the official MuJoCo viewer; native angle annotations."""
import argparse, ctypes, hashlib, json, math, os, re, subprocess, sys, time, threading
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deploy
p=argparse.ArgumentParser(); p.add_argument('--case',choices=['up','down'],required=True); args=p.parse_args()
OUT=ROOT/'docs/stairs_xml_alignment_20261007'; OUT.mkdir(exist_ok=True)
checkpoint=ROOT/'.scratch/gait_quality_20261007/latest.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='765228e6499dd8af9fd3eb7d41d8289072b4478ad20e19ce95b23ab8f7862fe8'
active=None; captured=False; window=None

def normalized(v): return v/np.linalg.norm(v)
def angle(u,v): return math.degrees(math.acos(float(np.clip(np.dot(normalized(u),normalized(v)),-1,1))))
def tilt(v,forward,down): return math.degrees(math.atan2(np.dot(v,forward),np.dot(v,down)))
assert abs(angle(np.array([0,0,1.]),np.array([0,0,-1.]))-180)<1e-8
assert abs(angle(np.array([1.,0,0]),np.array([0,0,-1.]))-90)<1e-8
assert abs(tilt(np.array([1.,0,-1]),np.array([1.,0,0]),np.array([0,0,-1.]))-45)<1e-8
assert tilt(np.array([0.,0,-1]),np.array([1.,0,0]),np.array([0,0,-1.]))==0

class Contract(deploy.M20Contract):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        global active; active=self
    def reset(self,data,*a,**kw): self.data=data; return super().reset(data,*a,**kw)

class Terminal(deploy.TerminalInput):
    def __init__(self,k): super().__init__(k); self.sent=set()
    def start(self): return True
    def close(self): pass
    def poll(self):
        for when,key in ((.25,'z'),(5.8,'c')):
            if active.data.time>=when and key not in self.sent: self.keyboard.on_char(key); self.sent.add(key)
        if active.data.time>=6.5: self.keyboard.on_char('w')

def own_window():
    lines=subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines()
    for line in lines:
        if 'mujoco' not in line.lower(): continue
        match=re.match(r'\s*(0x[0-9a-fA-F]+)',line)
        if not match: continue
        xid=match.group(1)
        result=subprocess.run(['xprop','-id',xid,'_NET_WM_PID'],capture_output=True,text=True)
        pid=re.search(r'=\s*(\d+)',result.stdout)
        if pid and int(pid.group(1))==os.getpid(): return xid
    raise RuntimeError('Own MuJoCo X11 window not found by PID; no whole-desktop capture')

class Handle:
    def __init__(self,h): self.h=h
    def __getattr__(self,n): return getattr(self.h,n)
    def is_running(self): return self.h.is_running() and not captured
def launch(model,data,key_callback=None):
    global window
    from mujoco import viewer as mj_viewer
    existing=set(threading.enumerate())
    h=mj_viewer.launch_passive(model,data,key_callback=key_callback,show_left_ui=False,show_right_ui=False)
    threads=[t for t in threading.enumerate() if t not in existing]
    for _ in range(20):
        try: window=own_window(); break
        except RuntimeError: time.sleep(.05)
    if window is None: raise RuntimeError('Own viewer unavailable')
    x=ctypes.CDLL('libX11.so.6'); x.XOpenDisplay.restype=ctypes.c_void_p
    x.XResizeWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_uint,ctypes.c_uint]
    x.XFlush.argtypes=[ctypes.c_void_p]; x.XCloseDisplay.argtypes=[ctypes.c_void_p]
    display=x.XOpenDisplay(None); assert display
    x.XResizeWindow(display,int(window,16),1280,900); x.XFlush(display); x.XCloseDisplay(display)
    return Handle(h),threads

def geom(scene,kind,pos,size,rgba,label=None):
    g=scene.geoms[scene.ngeom]; mujoco.mjv_initGeom(g,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),np.eye(3).ravel(),np.array([*rgba,1.],dtype=np.float32))
    if label: g.label=label
    scene.ngeom+=1; return g
def line(scene,a,b,color,width=.004):
    g=geom(scene,mujoco.mjtGeom.mjGEOM_CAPSULE,(a+b)/2,[width,0,0],color)
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.array(a,dtype=float),np.array(b,dtype=float))
def text(scene,p,label,color): geom(scene,mujoco.mjtGeom.mjGEOM_LABEL,p,[0,0,0],color,label)
def arc(scene,center,u,v,radius,color,label):
    u=normalized(u); v=normalized(v); a=math.acos(float(np.clip(np.dot(u,v),-1,1)))
    transverse=normalized(v-np.dot(v,u)*u)
    positions=[center+radius*(math.cos(t)*u+math.sin(t)*transverse) for t in np.linspace(0,a,40)]
    for start,end in zip(positions,positions[1:]): line(scene,start,end,color,.0025)
    middle=center+radius*1.25*(math.cos(a/2)*u+math.sin(a/2)*transverse)
    text(scene,middle,label,color)

YELLOW=(1.,.9,.05); PINK=(1.,.25,.7); CYAN=(.1,1.,1.); ORANGE=(1.,.5,.1); WHITE=(1.,1.,1.)

def annotate(viewer,c,tag,selected_leg):
    d=mujoco.MjData(c.model)
    mujoco.mj_copyData(d,c.model,c.data)
    # mj_step integrates qpos after solving; refresh geometry for that exact qpos
    # before comparing raw joint rotation with rendered-pivot angles. No step/replay.
    mujoco.mj_kinematics(c.model,d)
    R=d.xmat[c.base_id].reshape(3,3).copy(); forward=R[:,0]; body_down=-R[:,2]
    yaw=math.atan2(R[1,0],R[0,0]); world_forward=np.array([math.cos(yaw),math.sin(yaw),0]); world_down=np.array([0.,0.,-1.])
    if args.case=='up': leg=selected_leg
    else:
        v=[(d.xpos[c.model.body(l+'_wheel').id]-d.xpos[c.model.body(l+'_hipy').id])@forward for l in ('hl','hr')]
        leg=('hl','hr')[int(np.argmax(v))]
    H,K,W=[d.xpos[c.model.body(leg+'_'+b).id].copy() for b in ('hipy','knee','wheel')]
    joint=c.model.joint(leg+'_knee_joint'); q=float(d.qpos[joint.qposadr[0]])
    axis=d.xaxis[joint.id].copy(); u=K-H; v=W-K
    u_plane=u-np.dot(u,axis)*axis; v_plane=v-np.dot(v,axis)*axis
    inner=angle(-u_plane,v_plane); geometric=angle(H-K,W-K)
    assert abs(inner+abs(math.degrees(q))-180)<1e-6
    beta=tilt(W-H,forward,body_down); phi=tilt(W-H,world_forward,world_down); thigh=tilt(K-H,forward,body_down)
    captured_state={'case':args.case,'leg':leg,'time_s':float(d.time),'checkpoint':'model_196600.pt','physical_closed_loop':True,'phase_tag':tag,'terrain':c.terrain.summary(),'command_x':deploy.TELEOP_VELOCITIES['w'][0],
        'H_hip_joint_world':H.tolist(),'K_knee_joint_world':K.tolist(),'W_wheel_center_world':W.tolist(),
        'knee_q_signed_deg':math.degrees(q),'knee_inner_hinge_plane_deg':inner,'knee_inner_3d_deg':geometric,
        'whole_leg_forward_tilt_body_deg':beta,'whole_leg_forward_tilt_gravity_deg':phi,'thigh_forward_tilt_body_deg':thigh,
        'body_pitch_positive_nose_down_deg':math.degrees(math.atan2(-R[2,0],math.hypot(R[2,1],R[2,2])))}
    with viewer.lock():
        scene=viewer.user_scn; scene.ngeom=0
        left=leg.endswith('l'); side=R[:,1]*(1 if left else -1)
        offset=side*.16
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:]=d.xpos[c.base_id]+np.array([0,0,-.10]); viewer.cam.distance=1.8
        viewer.cam.azimuth=math.degrees(yaw)+(-90 if left else 90); viewer.cam.elevation=-5; viewer.cam.orthographic=1
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        if args.case=='up':
            # H*/W* are projections onto the knee hinge plane. This avoids drawing
            # raw joint q as the different 3D pivot-to-pivot internal angle.
            A=K-u_plane+offset; B=K+offset; C=K+v_plane+offset
            for pos,name in [(A,'H*'),(B,'K'),(C,'W*')]:
                geom(scene,mujoco.mjtGeom.mjGEOM_SPHERE,pos,[.010,0,0],YELLOW); text(scene,pos+side*.025,name,WHITE)
            line(scene,A,B,YELLOW);line(scene,B,C,YELLOW)
            line(scene,B,B+normalized(u_plane)*.20,PINK,.002)
            arc(scene,B,-u_plane,v_plane,.065,YELLOW,f'a={inner:.1f}')
            arc(scene,B,u_plane,v_plane,.12,PINK,f'q={abs(math.degrees(q)):.1f}')
            labels='Model / frame\nSelected leg\nYELLOW a: knee inner angle\nPINK q: knee hinge rotation\nRelation in hinge plane\nH*, W*: plane projections\n3D pivot inner angle'
            values=f"196600 / t={d.time:.2f}s\n{leg.upper()} (front)\n{inner:.1f} deg\n{math.degrees(q):+.1f} deg\na + |q| = 180 deg\nK: knee pivot\n{geometric:.1f} deg (different definition)"
        else:
            whole_body=np.dot(W-H,forward)*forward+np.dot(W-H,body_down)*body_down
            whole_world=np.dot(W-H,world_forward)*world_forward+np.dot(W-H,world_down)*world_down
            thigh_body=np.dot(K-H,forward)*forward+np.dot(K-H,body_down)*body_down
            A=H+offset; B=A+thigh_body; C=A+whole_body
            for pos,name in [(A,'H'),(B,'K*'),(C,'W*')]:
                geom(scene,mujoco.mjtGeom.mjGEOM_SPHERE,pos,[.010,0,0],CYAN);text(scene,pos+side*.025,name,WHITE)
            line(scene,A,B,ORANGE);line(scene,B,C,ORANGE)
            line(scene,A,C,CYAN,.003)
            body_ref=A+body_down*.48; grav_ref=A+world_down*.48
            for direction,color in [(body_down,PINK),(world_down,WHITE)]:
                for start in np.arange(0,.48,.04):line(scene,A+direction*start,A+direction*(start+.025),color,.002)
            arc(scene,A,body_down,whole_body,.22,CYAN,f'b={beta:.1f}')
            arc(scene,A,world_down,whole_world,.30,WHITE,f'g={phi:.1f}')
            arc(scene,A,body_down,thigh_body,.12,ORANGE,f't={thigh:.1f}')
            text(scene,body_ref,'body -Z',PINK);text(scene,grav_ref,'gravity down',WHITE)
            # A forward arrow makes the sign convention visible on the same frame.
            origin=d.xpos[c.base_id]+side*.20+R[:,2]*.08
            line(scene,origin,origin+forward*.24,WHITE,.003);text(scene,origin+forward*.26,'body +X',WHITE)
            labels='Model / frame\nSelected leg\nCYAN b: H-W vs body -Z\nWHITE g: H-W vs gravity\nORANGE t: H-K vs body -Z\nBody pitch (nose down)\nH / K* / W*\nAngles measured in sagittal planes'
            values=f"196600 / t={d.time:.2f}s\n{leg.upper()} (rear)\n{beta:+.1f} deg\n{phi:+.1f} deg\n{thigh:+.1f} deg\n{captured_state['body_pitch_positive_nose_down_deg']:.1f} deg\nhip / projected knee / wheel\nforward tilt is positive"
    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,labels,values))
    viewer.sync();time.sleep(.9)
    assert own_window()==window
    subprocess.run(['import','-window',window,str(OUT/f'{tag}_angles.png')],check=True,timeout=15)
    np.savez_compressed(OUT/f'{tag}_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),time=d.time)
    (OUT/f'{tag}_angles.json').write_text(json.dumps(captured_state,indent=2)+'\n')
    print('CAPTURE',json.dumps(captured_state),flush=True)
    time.sleep(.15)

from gait_metrics import GaitMetrics
metrics=None; timers=np.zeros(4); snapshots=set(); rows=[]; high_water=-float('inf'); last_capture=-float('inf'); reason=None

def measure(viewer,c,_visible):
    global captured,metrics,timers,high_water,last_capture,reason
    if metrics is None: metrics=GaitMetrics(c)
    d=mujoco.MjData(c.model)
    mujoco.mj_copyData(d,c.model,c.data)
    if d.time<6.5: return
    mujoco.mj_kinematics(c.model,d)
    wheels=d.xpos[c.wheel_body_ids].copy()
    ground=np.array([c.ground_height(d,*position[:2]) for position in wheels])
    forces=metrics.forces(c.data);norm=np.linalg.norm(forces,axis=1)
    supported=(forces[:,2]>5)&(forces[:,2]>.5*norm)&(np.abs(wheels[:,2]-ground-.09)<=.05)
    timers=np.where(supported,timers+.02,0)
    stable=supported&(timers>=.20);air=norm<5
    q=np.degrees(np.abs(d.qpos[metrics.knees])).copy()
    start=c.terrain.parameters['start_x'][0];h=c.terrain.parameters['step_height'][0];depth=c.terrain.parameters['tread_depth'][0]
    phase=bool(wheels[:2,0].max()>=start-.1 and not (wheels[:,0]>=c.terrain.goal_x+.09).all())
    rows.append(dict(time=float(d.time),qpos=d.qpos.copy(),root=d.xpos[c.base_id].copy(),knee_deg=q,
        force=forces,wheel=wheels,ground=ground,support=stable,swing=air,phase=phase))
    support_front=stable[:2]&(ground[:2]>=h*.5)
    swing_front=air[:2]&(wheels[:2,0]>=start+depth*2)
    loaded_front=supported[:2]&(forces[:2,2]>50)&(ground[:2]>=h*.5)&(ground[:2]<h*(c.terrain.parameters['step_count'][0]-.5))&(q[:2]>=100)&(wheels[:2,0]>=start+depth*2)
    if phase and support_front.any() and 'first_support' not in snapshots:
        leg=('fl','fr')[int(np.where(support_front,q[:2],-np.inf).argmax())]
        annotate(viewer,c,'first_support',leg); snapshots.add('first_support')
    elif phase and loaded_front.any() and 'loaded_contact' not in snapshots:
        leg=('fl','fr')[int(np.where(loaded_front,q[:2],-np.inf).argmax())]
        annotate(viewer,c,'loaded_contact',leg);snapshots.add('loaded_contact')
        path=OUT/'loaded_contact_angles.json';record=json.loads(path.read_text())
        side=('fl','fr').index(leg);record['wheel_force_world_n']=forces[side].tolist();record['wheel_ground_m']=float(ground[side]);record['continuous_contact_s']=float(timers[side]);path.write_text(json.dumps(record,indent=2)+'\n')
    elif phase and support_front.any() and ground[:2].min()>=3*h-.01 and 'middle_support' not in snapshots:
        leg=('fl','fr')[int(np.where(support_front,q[:2],-np.inf).argmax())]
        annotate(viewer,c,'middle_support',leg);snapshots.add('middle_support')
    elif phase and swing_front.any() and 'later_swing' not in snapshots:
        leg=('fl','fr')[int(np.where(swing_front,q[:2],-np.inf).argmax())]
        annotate(viewer,c,'later_swing',leg);snapshots.add('later_swing')
    elif phase and swing_front.any():
        side=int(np.where(swing_front,q[:2],-np.inf).argmax())
        if q[side]>high_water+4 and d.time-last_capture>.25:
            annotate(viewer,c,'high_swing',('fl','fr')[side]);high_water=q[side];last_capture=d.time;snapshots.add('high_swing')
    if (wheels[:,0]>=c.terrain.goal_x+.09).all() and (np.abs(wheels[:,1])<=c.terrain.course_width/2).all():
        reason='all_wheels_completed';captured=True
    elif phase and (np.abs(wheels[:,1])>c.terrain.course_width/2).any():
        reason='left_course';captured=True

deploy.M20Contract=Contract;deploy.TerminalInput=Terminal;deploy.launch_viewer=launch;deploy.draw_height_scan=measure
# Preserve original keyboard velocities. Only the script's original Z/C/W press timing is reproduced.
run=argparse.Namespace(model=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml'),checkpoint=checkpoint,
    terrain=None,terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml',stair_height=None,tread_depth=None,stair_count=None,stair_start=None,
    viewer=True,autoplay=False,steps=2000,output=None,print_every=250,stop_on_fall=True)
result=deploy.interactive_rollout(run)
assert rows and snapshots,'No terrain-phase screenshot captured'
data={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
np.savez_compressed(OUT/'telemetry.npz',**data)
phase=data['phase'][:,None]
support=data['support'][:,:2]&phase&(data['ground'][:,:2]>=.075)
swing=data['swing'][:,:2]&phase&(data['wheel'][:,:2,0]>=.25+.4)
def stats(mask):
    values=data['knee_deg'][:,:2][mask]
    return {'count':int(mask.sum()),'knee_q_p95_deg':float(np.quantile(values,.95)) if len(values) else None,
            'knee_q_max_deg':float(values.max()) if len(values) else None,'knee_inner_min_deg':180-float(values.max()) if len(values) else None}
result.update(stop_reason=reason or ('fall' if result['fell_in_policy'] else 'time_cap'),
    terrain_source_sha256=hashlib.sha256(run.terrain_xml.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    command_x=deploy.TELEOP_VELOCITIES['w'][0],terrain_cli_overrides=None,screenshots=sorted(snapshots),
    support_stats=stats(support),swing_stats=stats(swing))
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('SUMMARY',json.dumps(result),flush=True)
