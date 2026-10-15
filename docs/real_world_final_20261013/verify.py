"""Bounded terrain geometry/contact checks; never a policy traversal benchmark.

Run from any directory using m20_wzh and MUJOCO_GL=osmesa.
  python verify.py --model /absolute/path/to/M20.xml [--render]
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import resource
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / 'deploy/deploy_mujoco/terrains/real_world_final.xml'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'deploy/deploy_mujoco'))


def ray(model, data, x, y):
    gid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([x,y,4.]), np.array([0.,0.,-1.]),
                            np.array([1,0,0,0,0,0], dtype=np.uint8), True, -1, gid)
    assert distance >= 0 and gid[0] >= 0
    return 4-distance, model.geom(int(gid[0])).name


def probe(x,y,z, *, remove_priority=False, slide=False):
    spec = mujoco.MjSpec.from_file(str(XML))
    if remove_priority:
        for g in spec.worldbody.geoms:
            g.priority = 0
    body = spec.worldbody.add_body(name='probe', pos=[x,y,z+.065])
    body.add_freejoint()
    body.add_geom(name='probe_geom',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.035,.035,.035],
                  mass=1,friction=[1.2,.02,.002],condim=3,group=1,contype=1,conaffinity=1)
    model = spec.compile(); data = mujoco.MjData(model)
    for _ in range(180):
        mujoco.mj_step(model,data)
    contacts=[c for c in data.contact if model.geom('probe_geom').id in c.geom]
    assert contacts, (x,y,'no probe contacts')
    answer={'friction': contacts[0].friction.tolist(), 'dim': int(contacts[0].dim),
            'settled_z': float(data.qpos[2])}
    if slide:
        # Same cube, same velocity, equal-height neighboring dry/wet tile lanes.
        start=float(data.qpos[0]); data.qvel[0]=1.
        for _ in range(240):
            mujoco.mj_step(model,data)
        assert np.isfinite(data.qpos).all()
        answer['slide_distance_m']=float(data.qpos[0]-start)
    return answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--render',action='store_true')
    args=p.parse_args()
    native=mujoco.MjModel.from_xml_path(str(XML)); nd=mujoco.MjData(native)
    mujoco.mj_forward(native,nd)
    assert native.nq==0 and native.nbody==1
    assert np.all(native.geom_group==0) and np.all(native.geom_contype==1)
    planes=np.flatnonzero(native.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)
    assert len(planes)==1 and native.geom_pos[planes[0],2]<-.18
    assert native.ngeom < 500
    # Independent specification points: not reconstructed from the XML being tested.
    samples=[(-1,.2,0),(2.3,.2,0),(3.5,-1.25,0),(5.38,0,.045),(6.2,0,.012),
             (7.25,.4,.12),(8.05,-.715,.127),(10.25,0,.06),
             (11.66,0,.15),(12.30,0,.45),(13.4,0,.60),(14.38,0,.45),(15.4,0,0),
             (17.15,0,.15),(18.2,0,.30),(19.25,0,.15),
             (21.75,-1.5,.035),(21.75,1.5,.245),(24.2,0,.08),(25.4,-.75,.02),
             (26.4,-1.6,.002),(27.515,0,0),(27.5475,0,-.16),
             (41.5,0,0),(42.1,0,0),(42.2,0,-.18),(43,.95,-.18),(51,0,0)]
    heights=[]
    for x,y,z in samples:
        actual, name=ray(native,nd,x,y)
        np.testing.assert_allclose(actual,z,atol=3e-6,err_msg=f'{x},{y}: {name}')
        heights.append(dict(x=x,y=y,expected_z=z,actual_z=actual,geom=name))
    pit_z,pit_name=ray(native,nd,31.1,-.5)
    assert pit_z < -.07 and pit_name=='potholes'
    heights.append(dict(x=31.1,y=-.5,actual_z=pit_z,geom=pit_name))
    # Instrument controls: known plane and intentionally wrong floor-only terrain.
    fm=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="100 100 .1"/></worldbody></mujoco>')
    fd=mujoco.MjData(fm); mujoco.mj_forward(fm,fd)
    assert ray(fm,fd,0,0)[0]==0
    rejected=[abs(ray(fm,fd,x,y)[0]-z)>3e-6 for x,y,z in samples if z!=0]
    assert rejected and all(rejected)
    # Every contact class has an accessible flat spot; values are compared to compiled source.
    locations={'wood':(-1,.2,0),'tile':(3.5,1.25,0),'wet_tile':(3.5,-1.25,0),
        'carpet':(6.2,0,.012),'concrete':(13.4,0,.6),'tactile':(8.25,0,.127),
        'asphalt':(23.7,0,0),'metal':(25.4,-.75,.02),'paint':(26.4,-1.6,.002),
        'leaves':(41.5,0,0),'grass':(41.5,1.5,0),'wet_wood':(43.1,0,0),
        'rubber':(51,0,0),'brick':(45.9,-1,.65),'stone':(33.2,-1.75,.075),
        'dirt':(37.1,0,.01)}
    contacts={}
    for cls,(x,y,z) in locations.items():
        v=probe(x,y,z,slide=cls in ('tile','wet_tile'))
        gid=next(i for i in range(native.ngeom) if native.geom_matid[i]>=0 and
                 native.material(native.geom_matid[i]).name=='rw_mat_'+cls)
        f=native.geom_friction[gid]
        np.testing.assert_allclose(v['friction'],f[[0,0,1,2,2]],atol=1e-9)
        assert v['dim']==6
        contacts[cls]=v
    negative=probe(3.5,-1.25,0,remove_priority=True)
    assert abs(negative['friction'][0]-contacts['wet_tile']['friction'][0])>.5
    assert contacts['wet_tile']['slide_distance_m'] > 1.5*contacts['tile']['slide_distance_m']

    # Real project entry and exact documented robot, without selecting/loading a policy.
    import torch
    import deploy_mujoco as deploy
    torch.set_num_threads(1)
    da=argparse.Namespace(terrain=None,terrain_xml=XML)
    model=deploy.load_model(args.model.resolve(),da)
    contract=deploy.M20Contract(model,da.terrain_config)
    data=mujoco.MjData(model); contract.reset(data)
    original=mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    assert (model.nq,model.nv,model.nu)==(original.nq,original.nv,original.nu)
    np.testing.assert_array_equal(model.body_mass,original.body_mass)
    for getter,count in [('joint',model.njnt),('actuator',model.nu)]:
        assert [getattr(model,getter)(i).name for i in range(count)]==[getattr(original,getter)(i).name for i in range(count)]
    static=model.geom_bodyid==0
    assert sum(static & (model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE))==1
    scan_count=0; root=int(contract.root_joint.qposadr[0])
    for x in (-1,3.5,6,8,12,14.8,17.5,21.5,25.5,27.8,31,35,39,43.5,47,51,
              54.5,60.8,62.1,63.4,65.5,68.8,74,77,79,84,87,90,93,95,97,100):
        for yaw in (0,.65):
            contract.reset(data)
            data.qpos[root:root+7]=[x,0,1.7,math.cos(yaw/2),0,0,math.sin(yaw/2)]
            mujoco.mj_forward(model,data)
            scan=contract.height_scan(data)
            assert scan.shape==(187,) and np.isfinite(scan).all()
            expected=np.array([ray(native,nd,*xy)[0] for xy in contract.last_scan_hits[:,:2]])
            np.testing.assert_allclose(contract.last_scan_hits[:,2],expected,atol=3e-6)
            scan_count+=1
    # M20 physical settling on one dry and one wet lane with existing PD code.
    robot_contacts={}
    for label,y,expected in [('dry',1.15,.60),('wet',-1.15,.22)]:
        contract.reset(data); data.qpos[root:root+3]=[3.80,y,.58]
        mujoco.mj_forward(model,data)
        for _ in range(180):
            contract.apply_pd(data,np.zeros(16)); mujoco.mj_step(model,data)
        observed=[]
        for c in data.contact:
            for gid in c.geom:
                if model.geom(int(gid)).name.startswith('terrain_tile_'):
                    observed.append(float(c.friction[0])); assert c.dim==6
        assert observed, (label,'no M20 tile contacts')
        np.testing.assert_allclose(observed,expected,atol=1e-9)
        assert np.isfinite(data.qpos).all()
        robot_contacts[label]={'count':len(observed),'mu':observed,'physics_seconds':float(data.time)}

    result=dict(mujoco=mujoco.__version__,xml_sha256=hashlib.sha256(XML.read_bytes()).hexdigest(),
        robot=str(args.model.resolve()),native_geoms=native.ngeom,hfields=native.nhfield,textures=native.ntex,
        surface_contacts=contacts,robot_contacts=robot_contacts,height_samples=heights,
        floor_only_negative_control_rejected=True,priority_negative_control_mu=negative['friction'][0],
        scans=scan_count,scan_points=187,robot_mass_joints_actuators_unchanged=True,
        render='not_requested',policy_traversal='NOT_RUN',real_robot='NOT_RUN')
    if args.render:
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
        images=[]
        views=[('01–03  家门：木板 / 湿瓷砖 / 门槛 / 地毯',2.7,9.3),
               ('04–05  社区：路缘 / 盲道 / 四级楼梯',10.8,9.8),
               ('06–07  车道：纵坡 / 横坡 / 连续过渡',19.5,9.3),
               ('08–09  街道：减速带 / 井盖 / 排水格栅',26.6,8.8),
               ('10–11  老路：坑洼 / 车辙 / 卵石铺装',33.4,9.2),
               ('12–13  公园：起伏土路 / 树根 / 湿栈桥',40.6,9.6),
               ('14–15  中继：花坛绕行 / 停稳休息区',48.8,8.8),
               ('细节  格栅真缝与下沉沟底',27.8,2.8)]
        with mujoco.Renderer(native,height=450,width=760) as renderer:
            for title,x,dist in views:
                cam=mujoco.MjvCamera(); cam.lookat[:]=[x,0,.06]
                cam.distance=dist; cam.azimuth=105; cam.elevation=-55
                renderer.update_scene(nd,camera=cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                im=Image.new('RGB',(760,490),(24,32,39))
                im.paste(Image.fromarray(renderer.render()),(0,40))
                ImageDraw.Draw(im).text((14,5),title,font=font,fill=(235,239,242))
                images.append(im)
        sheet=Image.new('RGB',(1520,1960))
        for i,im in enumerate(images):
            sheet.paste(im,((i%2)*760,(i//2)*490))
        sheet.save(OUT/'preview.png')
        # Lower oblique close-ups reveal risers and curved/negative shapes that
        # were hard to see in the original high-angle overview.
        special_views=[('05  常规楼梯：上行 / 平台 / 下行',13.2,8.6),
                       ('16  变阶高、变踏深楼梯',55.9,8.2),
                       ('17  半圆扇形转弯楼梯',62.8,9.0),
                       ('18  左右错位踏步',69.5,7.8),
                       ('19  尖顶屋脊坡',74,7.4),
                       ('20  三段连续波浪坡',79,8.8),
                       ('21  真实 V 形下凹沟',84,7.4),
                       ('22  倾斜多边形踏石',89.1,8.5),
                       ('23  S 形起伏窄桥',95,8.8)]
        images=[]
        # Preview-only lighting; model collision/contact parameters unchanged.
        native.vis.headlight.ambient[:]=[.16,.16,.16]
        native.vis.headlight.diffuse[:]=[.35,.35,.35]
        native.light_dir[0]=[-.65,-.5,-.65]
        native.light_ambient[0]=[.12,.12,.12]
        native.light_diffuse[0]=[.8,.78,.74]
        mujoco.mj_forward(native,nd)
        with mujoco.Renderer(native,height=420,width=720) as renderer:
            for title,x,dist in special_views:
                cam=mujoco.MjvCamera(); cam.lookat[:]=[x,0,.20]
                cam.distance=dist; cam.azimuth=125; cam.elevation=-32
                if title.startswith('17 '):
                    cam.lookat[:]=[62.3,0,.30]
                    cam.distance=7.8; cam.azimuth=-70; cam.elevation=-48
                renderer.update_scene(nd,camera=cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                # Hide only perimeter walls to expose the geometry; collision
                # remains present. Door/planter/central plinth are NOT removed.
                for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
                        n=native.geom(g.objid).name
                        if 'boundary' in n or n in ('back_wall','finish_back_wall'):
                            g.rgba[3]=0
                im=Image.new('RGB',(720,460),(24,32,39))
                im.paste(Image.fromarray(renderer.render()),(0,40))
                ImageDraw.Draw(im).text((12,5),title,font=font,fill=(235,239,242))
                images.append(im)
        sheet=Image.new('RGB',(2160,1380))
        for i,im in enumerate(images):
            sheet.paste(im,((i%3)*720,(i//3)*460))
        sheet.save(OUT/'special_shapes.png')
        result['render']='static native scene, inspected separately; no policy animation'
    result['max_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (OUT/'checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('surface_contacts','height_samples')},indent=2))


if __name__=='__main__':
    main()
