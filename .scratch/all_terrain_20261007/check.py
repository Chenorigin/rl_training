from pathlib import Path
import argparse, json, sys, math, xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deploy
from PIL import Image

torch.set_num_threads(1)
xml=ROOT/'deploy/deploy_mujoco/terrains/all_terrain.xml'
robot=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
args=argparse.Namespace(terrain=None,terrain_xml=xml)
model=deploy.load_model(robot,args); c=deploy.M20Contract(model,args.terrain_config); d=mujoco.MjData(model); c.reset(d)
original=mujoco.MjModel.from_xml_path(str(robot))
np.testing.assert_array_equal(model.body_mass,original.body_mass)
assert [model.joint(i).name for i in range(model.njnt)]==[original.joint(i).name for i in range(original.njnt)]
assert [model.actuator(i).name for i in range(model.nu)]==[original.actuator(i).name for i in range(original.nu)]
assert (model.nq,model.nv,model.nu)==(original.nq,original.nv,original.nu)
static=model.geom_bodyid==0
assert np.all(model.geom_group[static]==0)
assert np.sum(static & (model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE))==1
assert abs(c.ground_height(d,-1,0))<1e-9
p=ET.parse(xml).getroot(); native=mujoco.MjModel.from_xml_path(str(xml)); assert native.ngeom==107
# Independent analytic surface: route definitions for stairs/slopes; directly
# read nonrotated native obstacle rectangles, not compiled MuJoCo coordinates.
rects=[]
for g in p.findall('./worldbody/geom'):
    name=g.get('name'); pos=np.fromstring(g.get('pos','0 0 0'),sep=' '); size=np.fromstring(g.get('size'),sep=' ')
    if name.startswith(('rough_','offset_','hurdle_')) or name=='low_friction_patch': rects.append((pos,size))
def height(x,y):
    if abs(y)>2: return 0.
    if 1<=x<2.8: return min(6,math.floor((x-1)/.3)+1)*.15
    if 2.8<=x<4.8: return .9
    if 4.8<=x<6.6: return max(0,5-math.floor((x-4.8)/.3))*.15
    if 8<=x<11: return (x-8)*math.tan(math.radians(12))
    if 11<=x<12.5: return 3*math.tan(math.radians(12))
    if 12.5<=x<15.5: return (15.5-x)*math.tan(math.radians(12))
    if 29<=x<=32: return (y+2)*math.tan(math.radians(4))
    z=0.
    for pos,size in rects:
        if abs(x-pos[0])<=size[0] and abs(y-pos[1])<=size[1]: z=max(z,pos[2]+size[2])
    return z
samples=[(-1,0,0),(1.15,0,.15),(1.75,0,.45),(2.65,0,.9),(3.8,0,.9),
         (4.95,0,.75),(5.55,0,.45),(6.15,0,.15),(6.75,0,0),
         (9.5,0,1.5*math.tan(math.radians(12))),(11.75,0,3*math.tan(math.radians(12))),
         (14,0,1.5*math.tan(math.radians(12))),(16.2,0,0),
         (17.1,-1.9,height(17.1,-1.9)),(20.5,0,0),(21.3,-1,.08),(21.42,1,.12),
         (22.3,-1,.14),(23.52,1,.14),(25.09,0,.08),(26.1,0,.12),(27.11,0,.16),
         (30.5,-1.8,.2*math.tan(math.radians(4))),(30.5,0,2*math.tan(math.radians(4))),
         (30.5,1.8,3.8*math.tan(math.radians(4))),(33,0,0),(35,0,.02),(38,0,0),(2,2.5,0)]
for x,y,z in samples: np.testing.assert_allclose(c.ground_height(d,x,y),z,atol=2e-8)
# Negative floor-only control must not satisfy raised-obstacle samples.
raised=[z for _,_,z in samples if z>0]; assert raised and not np.allclose(raised,0,atol=2e-8)
scan_errors=[]; root=int(c.root_joint.qposadr[0])
for x in [.137,1.137,2.337,3.737,5.537,6.937,9.537,11.737,14.037,17.737,19.137,21.537,23.537,25.137,27.037,30.137,35.137,38.137]:
    for yaw in [0.,.43]:
        # Measurement-only pose: not a policy traversal or physically settled state.
        d.qpos[root:root+7]=[x,.037,1.6,math.cos(yaw/2),0,0,math.sin(yaw/2)]; mujoco.mj_forward(model,d)
        scan=c.height_scan(d); xy=c.grid@np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])+[x,.037]
        heights=np.array([height(*q) for q in xy]); err=float(np.max(np.abs(c.last_scan_hits[:,2]-heights)))
        np.testing.assert_allclose(c.last_scan_hits[:,:2],xy,atol=1e-8)
        np.testing.assert_allclose(c.last_scan_hits[:,2],heights,atol=2e-8)
        np.testing.assert_allclose(scan,np.clip(1.6-heights-.5,-1,1),atol=2e-7)
        assert scan.shape==(187,) and np.isfinite(scan).all(); scan_errors.append(err)
idx=model.geom('terrain_low_friction_patch').id
np.testing.assert_allclose(model.geom_friction[idx],[.5,.01,.01])
assert model.geom_priority[idx] > model.geom_priority[~static].max()
# Real robot contacts on the patch: verify the effective contact coefficient,
# not just the XML value (MuJoCo otherwise mixes equally ranked friction).
c.reset(d)
d.qpos[root:root+3] = [35,0,.60]
mujoco.mj_forward(model,d)
for _ in range(100):
    c.apply_pd(d,np.zeros(16)); mujoco.mj_step(model,d)
patch_contacts=[d.contact[i].friction.copy() for i in range(d.ncon)
                if idx in d.contact[i].geom]
assert patch_contacts, 'No actual robot contact on low-friction patch'
for coefficients in patch_contacts: np.testing.assert_allclose(coefficients[:2],[.5,.5])
# Verify only the new scene native geoms inherit the named default.
assert all(model.geom_group[i]==0 and model.geom_contype[i] for i in np.flatnonzero(static))
out=ROOT/'docs/all_terrain_20261007'; out.mkdir(exist_ok=True)
views=[('楼梯与平台',3.5,.35,11,125,-25),('上下坡',11.5,.3,11,125,-25),('粗糙路面与障碍',22,.12,14,125,-32),('横坡与低摩擦区',33.5,.1,14,125,-32)]
c.reset(d); images=[]
with mujoco.Renderer(model,height=480,width=640) as renderer:
    for i,(label,x,z,distance,azimuth,elevation) in enumerate(views):
        camera=mujoco.MjvCamera(); camera.lookat[:]=[x,0,z]; camera.distance=distance; camera.azimuth=azimuth; camera.elevation=elevation
        renderer.update_scene(d,camera=camera); renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        im=Image.fromarray(renderer.render()); im.save(out/f'section_{i+1}.png'); images.append(im)
contact=Image.new('RGB',(1280,960))
for i,im in enumerate(images): contact.paste(im,((i%2)*640,(i//2)*480))
contact.save(out/'terrain_preview.png')
result=dict(terrain=str(xml),model=str(robot),native_terrain_geoms=native.ngeom,robot_joint_order_unchanged=True,robot_mass_unchanged=True,
            samples=[[x,y,z,c.ground_height(d,x,y)] for x,y,z in samples],height_scan_points=187,scan_poses=len(scan_errors),
            max_height_error_m=max(scan_errors),low_friction=[.5,.01,.01],floor_negative_control_rejected=True,
            effective_patch_sliding_friction=float(patch_contacts[0][0]),patch_contact_count=len(patch_contacts),
            rendering='static geometry preview, not policy motion',performance='full course not evaluated')
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
