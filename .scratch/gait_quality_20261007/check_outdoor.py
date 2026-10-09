import argparse, json, math, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image, ImageDraw
root=Path.cwd(); sys.path.insert(0,str(root/'deploy/deploy_mujoco'))
import deploy_mujoco as deploy
path=root/'deploy/deploy_mujoco/terrains/realistic_outdoor.xml'
robot=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
out=root/'docs/gait_quality_20261007/outdoor'; out.mkdir(exist_ok=True)
standalone=mujoco.MjModel.from_xml_path(str(path))
args=argparse.Namespace(terrain=None,terrain_xml=path)
model=deploy.load_model(robot,args); original=mujoco.MjModel.from_xml_path(str(robot))
assert (model.nq,model.nv,model.nu)==(original.nq,original.nv,original.nu)
assert [model.joint(i).name for i in range(model.njnt)]==[original.joint(i).name for i in range(original.njnt)]
np.testing.assert_allclose(model.body_mass,original.body_mass)
assert int(((model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)&(model.geom_bodyid==0)).sum())==1
c=deploy.M20Contract(model,args.terrain_config); d=mujoco.MjData(model); c.reset(d)
checks=[]
for x,y,z in [(-1,0,0),(3,.2,.12),(4.75,.2,.06),(6.24,1,.025),(6.49,1,0),
              (7.16,0,.15),(7.48,0,.30),(7.8,0,.45),(8.12,0,.6),(9.2,0,.6),
              (10.44,0,.45),(10.76,0,.30),(11.08,0,.15),(11.4,0,0),
              (24.4,-.7,.18),(25.7,-.15,.22),(30,0,2*math.tan(math.radians(8))),
              (32.7,0,4*math.tan(math.radians(8))),(35.5,0,2*math.tan(math.radians(8))),
              (40,.4,.01),(44,0,0)]:
    actual=c.ground_height(d,x,y); np.testing.assert_allclose(actual,z,atol=2e-7)
    checks.append({'x':x,'y':y,'height':actual,'expected':z})
source=ET.parse(path).find('./asset/hfield')
heights=np.fromstring(source.get('elevation'),sep=' ').reshape(49,145)
assert model.nhfield==1 and (int(model.hfield_nrow[0]),int(model.hfield_ncol[0]))==(49,145)
# Native elevation is normalized by MuJoCo. Probe asymmetric vertices to verify ordering.
field=np.asarray(model.hfield_data).reshape(49,145)
np.testing.assert_allclose(field,heights[::-1],atol=1e-7)
for row,col in [(0,0),(24,72),(13,40),(37,101),(48,144)]:
    x=17+6*col/144; y=-2+4*row/48
    actual=c.ground_height(d,x,y); expected=float(field[row,col]*.12)
    np.testing.assert_allclose(actual,expected,atol=2e-7)
    checks.append({'x':x,'y':y,'height':actual,'expected':expected,'hfield_vertex':[row,col]})
poses=0
for x in [-1,3,7.2,9,10.5,13.8,18,20,22,25.7,30,34,40,44]:
    for yaw in [0,.55]:
        rootid=int(c.root_joint.qposadr[0]); ground=c.ground_height(d,x,0)
        d.qpos[rootid:rootid+7]=[x,0,ground+.58,math.cos(yaw/2),0,0,math.sin(yaw/2)]
        mujoco.mj_forward(model,d)
        scan=c.height_scan(d)
        assert scan.shape==(187,) and np.isfinite(scan).all() and np.isfinite(c.last_scan_hits).all()
        obs=c.observe(d,np.array([.5,0,0],dtype=np.float32),np.zeros(16,dtype=np.float32))
        assert obs.shape==(244,) and np.isfinite(obs).all()
        poses+=1
# Actual contacts in wet zone: enforce physical surface friction, not metadata only.
c.reset(d); rootid=int(c.root_joint.qposadr[0]); d.qpos[rootid:rootid+3]=[40,0,.59]; mujoco.mj_forward(model,d)
for _ in range(80):
    c.apply_pd(d,np.zeros(16)); mujoco.mj_step(model,d)
wet=model.geom('terrain_wet_road').id if 'terrain_wet_road' in [model.geom(i).name for i in range(model.ngeom)] else next(i for i in range(model.ngeom) if 'wet_road' in model.geom(i).name)
contacts=[d.contact[i] for i in range(d.ncon) if wet in (d.contact[i].geom1,d.contact[i].geom2)]
assert contacts, 'no wet surface contact'
for contact in contacts: np.testing.assert_allclose(contact.friction[0],.45,atol=1e-12)
actual_friction=[float(contact.friction[0]) for contact in contacts]
c.reset(d)
panels=[]
with mujoco.Renderer(model,height=480,width=640) as renderer:
    for title,x,distance in [('Curb / stairs',7,16),('Repairs / dirt track',18,12),('Stones / ramp',29,15),('Wet road / exit',40,11)]:
        cam=mujoco.MjvCamera(); cam.lookat[:]=[x,0,.2]; cam.distance=distance; cam.azimuth=125; cam.elevation=-35
        renderer.update_scene(d,camera=cam); renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        im=Image.fromarray(renderer.render()); ImageDraw.Draw(im).text((15,15),title,fill='white'); panels.append(im)
image=Image.new('RGB',(1280,960))
for i,im in enumerate(panels): image.paste(im,((i%2)*640,(i//2)*480))
image.save(out/'preview.png')
result={'standalone_geoms':standalone.ngeom,'attached_geoms':model.ngeom,'abi':[244,16],
        'independent_ray_checks':checks,'scan_poses':poses,'wet_actual_contact_friction':actual_friction,
        'one_floor':True,'unchanged_robot_abi_mass':True,'full_route_performance':'not tested'}
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='independent_ray_checks'},indent=2))
