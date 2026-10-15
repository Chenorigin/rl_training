"""Independent shape heights and actual convex-mesh contact checks."""
from pathlib import Path
import hashlib
import json
import math
import resource
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT/'deploy/deploy_mujoco/terrains/real_world_final.xml'
OUT = Path(__file__).resolve().parent


def ray(model,data,x,y):
    gid=np.array([-1],dtype=np.int32)
    distance=mujoco.mj_ray(model,data,np.array([x,y,4.]),np.array([0.,0.,-1.]),
                          np.array([1,0,0,0,0,0],dtype=np.uint8),True,-1,gid)
    assert distance>=0
    return 4-distance,model.geom(int(gid[0])).name


def check_shapes():
    model=mujoco.MjModel.from_xml_path(str(XML)); data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    portable=mujoco.MjModel.from_xml_string(XML.read_text())
    assert portable.ngeom==model.ngeom and portable.nmesh==38
    points=[(53.71,0,.10),(54.1,0,.24),(54.44,0,.42),(54.88,0,.54),
            (55.21,0,.70),(55.8,0,.70),(56.4,0,.54),(56.7,0,.42),(57.1,0,.24),
            (57.55,0,.10),(58.4,0,0),(60.8,-.65,0),(63.4,-.65,.80),
            (65.275,-.65,.40),(62.1,-1.6,-.18),
            (67.6,-1,.08),(67.6,1,0),(68,1,.16),(68.3,-1,.18),
            (69,-1,.30),(69,1,.28),(73.25,0,.225),(74,0,.45),(74.75,0,.225),
            (76,0,0),(76.5,0,.12),(77,0,.24),(77.5,0,.12),(78,0,0),
            (79,0,.24),(81,0,.24),(82.5,0,-.06),(83.5,0,-.18),(84,0,-.24),(85,0,-.12),
            (86.785,-1.3,-.14),(95,1.2,-.18),(100,0,0)]
    probes=[]
    for i in range(10):
        a=math.pi-(i+.5)*math.pi/10
        p=(62.1+1.3*math.cos(a),1.3*math.sin(a),.08*(i+1))
        points.append(p); probes.append((f'fan_step_{i+1:02d}',p))
    for i,z in enumerate([.02,.09,.04,.12,.06,.10,.03,.07]):
        p=(86.42+i*.73,.28*(-1 if i%2==0 else 1),z)
        points.append(p); probes.append((f'polygon_stone_{i+1:02d}',p))
    for i in range(20):
        t=(i+.5)/20
        # A linearly segmented sine bridge has the average endpoint elevation.
        z=.06*(math.sin(math.pi*i/20)**2+math.sin(math.pi*(i+1)/20)**2)
        p=(92+6*t,.45*math.sin(2*math.pi*t),z)
        points.append(p); probes.append((f's_bridge_segment_{i+1:02d}',p))
    results=[]
    for x,y,z in points:
        actual,name=ray(model,data,x,y)
        np.testing.assert_allclose(actual,z,atol=3e-6,err_msg=f'{x},{y}: {name}')
        results.append(dict(x=x,y=y,expected_z=z,z=actual,geom=name))

    # A ray exactly on a shared radial edge used to pass through ~1e-7m cracks.
    # Strict checks at every edge and several radii also catch false deep pits
    # that are too thin to affect the physical sphere probes above.
    seam_points=[]
    for j in range(1,10):
        a=math.pi-j*math.pi/10
        for radius in (.8,1.3,1.8):
            x=62.1+radius*math.cos(a); y=radius*math.sin(a)
            actual,name=ray(model,data,x,y)
            np.testing.assert_allclose(actual,.08*(j+1),atol=3e-6,err_msg=f'fan seam {j}, {radius}')
            seam_points.append(dict(edge=j,radius=radius,x=x,y=y,z=actual,geom=name))
    fan_arc_min=1.
    for a in np.linspace(.0001,math.pi-.0001,1801):
        z,name=ray(model,data,62.1+1.3*math.cos(a),1.3*math.sin(a))
        assert name.startswith('fan_step_') and z>0, (a,z,name)
        fan_arc_min=min(fan_arc_min,z)
    seam_negative='NOT_RUN: pre-fix backup unavailable'
    pre_fix=ROOT/'.scratch/real_world_shapes_20261013/fan_pre_seam_fix.xml'
    if pre_fix.exists():
        old_m=mujoco.MjModel.from_xml_path(str(pre_fix)); old_d=mujoco.MjData(old_m)
        mujoco.mj_forward(old_m,old_d)
        leaks=[p for p in seam_points if ray(old_m,old_d,p['x'],p['y'])[1]=='fan_recess']
        assert len(leaks)>=2, 'Regression control must detect the old shared-edge cracks'
        seam_negative=len(leaks)

    # Instrument self-check and negative shape control.
    control=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" pos="0 0 .1" size="1 1 .1"/></worldbody></mujoco>')
    cd=mujoco.MjData(control); mujoco.mj_forward(control,cd)
    assert abs(ray(control,cd,0,0)[0]-.2)<1e-12
    fill_spec=mujoco.MjSpec.from_file(str(XML))
    fill_spec.worldbody.add_geom(name='incorrect_bridge_fill',type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[95,0,-.04],size=[3,2,.04])
    filled=fill_spec.compile(); filled_data=mujoco.MjData(filled)
    mujoco.mj_forward(filled,filled_data)
    assert abs(ray(filled,filled_data,95,1.2)[0])<1e-8
    assert abs(ray(model,data,95,1.2)[0]+.18)<1e-8

    # New mesh polygons are separately convex; collision must match the top ray.
    spec=mujoco.MjSpec.from_file(str(XML))
    body=spec.worldbody.add_body(name='shape_contact_probe')
    body.add_freejoint()
    body.add_geom(name='shape_contact_probe_geom',type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[.02],mass=.1,group=1,friction=[1.4,.02,.002],condim=3)
    pm=spec.compile(); pd=mujoco.MjData(pm)
    contact_results=[]
    for name,(x,y,z) in probes:
        mujoco.mj_resetData(pm,pd); pd.qpos[:7]=[x,y,z+.0198,1,0,0,0]
        mujoco.mj_forward(pm,pd)
        gid=pm.geom(name).id
        assert any(gid in c.geom for c in pd.contact), f'No initial mesh contact: {name}'
        for _ in range(40):
            mujoco.mj_step(pm,pd)
        hits=[c for c in pd.contact if gid in c.geom]
        assert hits and np.isfinite(pd.qpos).all(), name
        for c in hits:
            np.testing.assert_allclose(c.friction[:2],np.repeat(pm.geom_friction[gid,0],2))
            assert c.dim==6
        h,_=ray(model,data,pd.qpos[0],pd.qpos[1])
        assert abs(pd.qpos[2]-.02-h)<.003, (name,pd.qpos[2]-.02,h)
        contact_results.append(dict(geom=name,contacts=len(hits),mu=float(hits[0].friction[0]),
                                    foot_bottom_z=float(pd.qpos[2]-.02),ray_z=h))

    # All earlier per-geom XML attributes must remain byte-for-byte equivalent,
    # except the explicitly moved end wall and visually enlarged catch plane.
    baseline=ROOT/'.scratch/real_world_shapes_20261013/before/real_world_final.xml'
    old_preserved='NOT_CHECKED: baseline backup unavailable'
    if baseline.exists():
        before=ET.parse(baseline).getroot(); after=ET.parse(XML).getroot()
        old={e.get('name'):e.attrib for e in before.findall('./worldbody/geom')}
        new={e.get('name'):e.attrib for e in after.findall('./worldbody/geom')}
        for name,attrs in old.items():
            if name not in ('catch_floor','finish_back_wall'):
                assert new[name]==attrs, name
        # Assets and defaults that determine the old contacts also stay identical.
        for e in before.findall('./asset/*'):
            n=after.find(f'./asset/{e.tag}[@name="{e.get("name")}"]')
            assert n is not None and n.attrib==e.attrib
        assert ET.tostring(before.find('default'))==ET.tostring(after.find('default'))
        old_preserved=len(old)-2

    result=dict(xml_sha256=hashlib.sha256(XML.read_bytes()).hexdigest(),mujoco=mujoco.__version__,
                geoms=model.ngeom,meshes=model.nmesh,hfields=model.nhfield,
                native_and_path_independent_compile=True,height_samples=results,
                max_height_error_m=max(abs(r['z']-r['expected_z']) for r in results),
                physical_mesh_probes=contact_results,filled_bridge_negative_control_rejected=True,
                fan_seam_points=seam_points,fan_arc_samples=1801,fan_arc_min_z=fan_arc_min,
                pre_fix_seam_leaks_detected=seam_negative,
                original_geom_attributes_preserved=old_preserved,policy_traversal='NOT_RUN',
                rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (OUT/'shape_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('height_samples','physical_mesh_probes','fan_seam_points')},indent=2))
    return result


if __name__=='__main__':
    check_shapes()
