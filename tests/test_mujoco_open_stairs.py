"""Real collision/ray tests for parameterized open stairs; no policy training."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"deploy/deploy_mujoco"))
sys.path.insert(0, str(ROOT/"source/rl_training"))
import deploy_mujoco as deployment
from terrain_loader import (PRESETS, TERRAIN_DIR, attach_terrain, static_terrain_flex_ids,
                            terrain_ray, terrain_contact_bodies, select_terrain)
from stair_trot_adapter import StairTrotMujocoController
from task_support.stair_trot_v1.signals import TrotSettings
from student_v0_adapter import visible_terrain
from gait_metrics import GaitMetrics
from quiet_landing_metrics import QuietMujocoMetrics

OPEN = TERRAIN_DIR/"stairs_ascent_open.xml"
SOLID = TERRAIN_DIR/"stairs_ascent.xml"
GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


def probe_model(path=OPEN, **overrides):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><option timestep=".001"/><worldbody>'
        '<body name="probe" pos=".5 0 .3"><freejoint/>'
        '<geom type="sphere" size=".03" mass="1" group="1"/></body>'
        '</worldbody></mujoco>')
    terrain = attach_terrain(spec, path, overrides or dict(stair_count=3, stair_height=.2, tread_depth=.4))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, terrain


def ray_height(model, data, x, y):
    distance, _, _ = terrain_ray(model, data, np.array([x,y,10.]), np.array([0.,0.,-1.]),
                                  GROUPS, static_terrain_flex_ids(model))
    return 10-distance


class GeometryTests(unittest.TestCase):
    def test_preset_and_compiled_collision_are_real_world_fixed_horizontal_triangles(self):
        self.assertEqual(PRESETS['ascent_open'], OPEN.name)
        self.assertEqual(select_terrain(SimpleNamespace(terrain='ascent_open', terrain_xml=None)), OPEN)
        model, _, terrain = probe_model()
        self.assertEqual(model.nflex, 1)  # Must survive attachment to the robot.
        self.assertTrue(model.flex_rigid[0])
        self.assertEqual(model.flex_radius[0], 0)
        self.assertTrue((model.flex_vertbodyid == 0).all())
        self.assertEqual(model.flex_elemnum[0], 2*(3+2))
        triangles = model.flex_vert[model.flex_elem.reshape(-1,3)]
        normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        np.testing.assert_allclose(normals[:,:2],0,atol=1.e-12)
        self.assertTrue((normals[:,2] > 0).all())
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == 0 and model.geom_contype[geom]:
                self.assertEqual(model.geom_type[geom], mujoco.mjtGeom.mjGEOM_PLANE)
        visuals=np.flatnonzero(model.site_group==2)
        self.assertEqual(len(visuals),3+2)
        self.assertEqual(np.count_nonzero(model.geom_group==2),0)
        self.assertTrue((model.site_size[visuals,2] < 1.e-5).all())
        self.assertTrue((model.site_type[visuals]==mujoco.mjtGeom.mjGEOM_BOX).all())
        self.assertTrue(terrain.summary()['open_risers'])

    def test_height_count_depth_overrides_change_actual_raycast_surfaces(self):
        for count,height,depth in ((1,.08,.25),(6,.15,.35),(10,.24,.50)):
            model,data,terrain = probe_model(stair_count=count,stair_height=height,tread_depth=depth)
            start=terrain.parameters['start_x'][0]
            self.assertAlmostEqual(terrain.goal_x,start+count*depth)
            self.assertEqual(model.flex_elemnum[0],2*(count+2))
            self.assertAlmostEqual(ray_height(model,data,-1,0),0)
            for i in range(count):
                self.assertAlmostEqual(ray_height(model,data,start+(i+.5)*depth,.3),(i+1)*height)
            self.assertAlmostEqual(ray_height(model,data,terrain.goal_x+.5,0),count*height)
            self.assertAlmostEqual(ray_height(model,data,start+.5*depth,3),-.5)

    def test_xml_edits_without_cli_overrides_also_take_effect(self):
        tree=ET.parse(OPEN)
        for key,value in (('step_count',4),('step_height',.13),('tread_depth',.45)):
            tree.find(f"./custom/numeric[@name='{key}']").set('data',str(value))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'open.xml';tree.write(path)
            spec=mujoco.MjSpec()
            terrain=attach_terrain(spec,path)
            model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            self.assertEqual(model.flex_elemnum[0],12)
            self.assertAlmostEqual(ray_height(model,data,.25+3.5*.45,0),4*.13)
            self.assertAlmostEqual(terrain.goal_x,.25+4*.45)

    def test_missing_riser_really_allows_passage_where_solid_wall_contacts(self):
        for path,expect_wall in ((OPEN,False),(SOLID,True)):
            model,data,terrain=probe_model(path)
            # A sphere overlaps the middle of the first riser, far from both
            # horizontal treads/edges. Thin boxes or a convex stair hull fail.
            data.qpos[:3]=[terrain.parameters['start_x'][0]-.015,0,.1]
            data.qvel[:]=0
            mujoco.mj_forward(model,data)
            wall=any(abs(float(c.frame[2])) < .1 for c in data.contact)
            self.assertEqual(wall,expect_wall)
            if not expect_wall:
                self.assertEqual(data.ncon,0)

    def test_invalid_customizations_fail_before_building_collision(self):
        for kwargs in (dict(stair_count=0),dict(stair_count=2.5),dict(stair_height=-.1),
                       dict(tread_depth=0),dict(stair_height=float('nan'))):
            with self.assertRaises(ValueError):
                probe_model(**kwargs)


class PhysicalAndSensorTests(unittest.TestCase):
    def test_sphere_drops_and_is_supported_on_a_tread(self):
        model,data,_=probe_model()
        data.qpos[:3]=[.45,0,.35]
        data.qvel[:]=0
        mujoco.mj_forward(model,data)
        for _ in range(700):
            mujoco.mj_step(model,data)
        self.assertAlmostEqual(float(data.qpos[2]),.23,delta=.003)
        upward=0.
        ids=static_terrain_flex_ids(model)
        for i,c in enumerate(data.contact):
            b1,b2=terrain_contact_bodies(model,c,ids)
            self.assertTrue(b1 == 0 or b2 == 0)
            wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
            upward+=float(wrench[0])
        self.assertGreater(upward,8)

    def test_height_sensor_and_all_wheel_force_readers_include_flex(self):
        # Minimal named M20 interface; actual robot physics is checked separately.
        bodies=[]
        for i,name in enumerate(deployment.JOINT_NAMES):
            link=name.removesuffix('_joint')
            x,y,z=((.3 if i in (12,13) else -.3),(.2 if i in (12,14) else -.2),-.3) if i>=12 else (0,0,.3)
            radius=.09 if i>=12 else .005
            bodies.append(f'<body name="{link}" pos="{x} {y} {z}"><joint name="{name}" axis="0 1 0"/>'
                          f'<geom type="sphere" size="{radius}" mass="1" group="1" '
                          f'contype="{1 if i>=12 else 0}" conaffinity="{1 if i>=12 else 0}"/></body>')
        motors=''.join(f'<motor name="{name}" joint="{name}"/>' for name in deployment.JOINT_NAMES)
        spec=mujoco.MjSpec.from_string('<mujoco><worldbody><body name="base_link"><freejoint name="floating_base"/>'
             '<geom type="box" size=".2 .1 .1" mass="10" group="1"/>'+''.join(bodies)+
             '</body></worldbody><actuator>'+motors+'</actuator></mujoco>')
        terrain=attach_terrain(spec,OPEN,dict(stair_count=2,stair_height=.2,tread_depth=1.))
        model=spec.compile();data=mujoco.MjData(model)
        contract=deployment.M20Contract(model,terrain)
        contract.reset(data)
        data.qpos[:3]=[.75,0,.585]
        mujoco.mj_forward(model,data)
        scan=contract.height_scan(data)
        self.assertEqual(scan.shape,(187,))
        self.assertTrue(np.isfinite(scan).all())
        self.assertAlmostEqual(contract.ground_height(data,.75,0),.2)
        actor=SimpleNamespace(m20_training_manifest={'trot_settings':asdict(TrotSettings())})
        controller=StairTrotMujocoController(contract,actor,.02)
        controller.sample_contacts(data)
        self.assertTrue((controller.forces[:,2] > 0).all())
        gait=GaitMetrics(contract)
        self.assertTrue((gait.forces(data)[:,2] > 0).all())
        quiet=QuietMujocoMetrics(model,contract)
        quiet.post_step(data,quiet.pre_step(data))
        self.assertTrue((quiet.current_force[:,2] > 0).all())

    def test_student_visibility_uses_actual_tread_occlusion(self):
        model,_,_=probe_model(stair_count=2,stair_height=.2,tread_depth=1.)
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        # Identity base frame above a tread, two targets along the same ray:
        # the upper tread is visible; the lower floor behind it is occluded.
        hits=np.array([[.75,0,.2],[.75,0,-.5]])
        # Raycast needs actual MjData, while sensor pose uses that data's base.
        data.qpos[:3]=[.5,0,1.];mujoco.mj_forward(model,data)
        mask=visible_terrain(model,data,model.body('probe').id,hits,(0,0,0))
        np.testing.assert_array_equal(mask,[True,False])


if __name__ == '__main__':
    unittest.main()
