"""Bounded passive-terrain tests: no checkpoint, optimizer, GPU or training."""
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy/deploy_mujoco'))
from terrain_loader import attach_terrain, is_terrain_geom, terrain_contact_bodies, terrain_ray

XML = ROOT / 'deploy/deploy_mujoco/terrains/sample_stairs.xml'
GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
NAMES = ('stairs_25_left', 'stairs_25_right', 'stairs_30', 'white_platform', 'dark_box')


def robot_spec():
    return mujoco.MjSpec.from_string(
        '<mujoco><option timestep=".005"/><worldbody>'
        '<body name="probe" pos="1.205 0 .6"><freejoint/>'
        '<geom name="probe_geom" type="sphere" group="1" size=".05" mass="2"/>'
        '</body></worldbody></mujoco>')


def attached(probe_radius=.05):
    spec = robot_spec()
    spec.geom('probe_geom').size[0] = probe_radius
    terrain = attach_terrain(spec, XML)
    deck = spec.body('terrain_white_platform').pos
    spec.body('probe').pos = [deck[0], deck[1], .6]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, terrain


def advance(model, data, seconds):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
        raise AssertionError('Unstable physics or MuJoCo warning')


def support_response(locked=False, record=False):
    """Fixed comparison from evaluation criteria; no policy or optimizer.

    Control removes internal hinges only: identical geometry, mass and contact.
    A steady downward preload is retained during lateral-pulse recovery.
    """
    tree = ET.parse(XML)
    if locked:
        for body in tree.findall('./worldbody/body/body'):
            for joint in body.findall('joint'):
                body.remove(joint)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))
    data = mujoco.MjData(model)
    body = model.body('white_platform').id
    deck = model.geom('white_deck').id
    dark = model.geom('dark_box_top').id
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    joint_positions = model.jnt_qposadr[hinges]
    def point(geom, offset):
        return data.geom_xpos[geom]+data.geom_xmat[geom].reshape(3, 3) @ offset
    top_offset = np.array([0., 0., model.geom_size[deck, 2]])
    white_edge = np.array([0., -model.geom_size[deck, 1], model.geom_size[deck, 2]])
    dark_edge = np.array([0., model.geom_size[dark, 1], model.geom_size[dark, 2]])
    advance(model, data, 2.)
    initial = point(deck, top_offset)
    peak = angle = 0.
    gap_min = float('inf')
    trajectory = []
    for step in range(round(6.25/model.opt.timestep)):
        t = step*model.opt.timestep
        force = np.array([0., -25. if 1. <= t < 1.25 else 0., -150.])
        data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(model, data, force, np.zeros(3), point(deck, top_offset), body, data.qfrc_applied)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        top = point(deck, top_offset)
        gap = point(deck, white_edge)[1]-point(dark, dark_edge)[1]
        peak = max(peak, float(np.linalg.norm(top[:2]-initial[:2])))
        gap_min = min(gap_min, float(gap))
        if len(hinges):
            angle = max(angle, float(np.max(np.abs(data.qpos[joint_positions]))))
        if record:
            trajectory.append([float(data.time), *top, gap, *data.qpos])
    if np.any(data.warning.number) or not np.isfinite(data.qpos).all():
        raise AssertionError('Support response became unstable')
    return {'peak_sway_m': peak, 'peak_leg_angle_deg': float(np.degrees(angle)),
            'minimum_gap_m': gap_min, 'final_max_abs_qvel': float(np.max(np.abs(data.qvel))),
            'qpos_trajectory': np.asarray(trajectory)}, model, data


class MovableStairsTests(unittest.TestCase):
    def test_confirmed_dimensions_and_no_initial_penetration(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dimensions = {'stairs_25_left': [.75, .5, .45],
                      'stairs_25_right': [.75, .5, .45],
                      'stairs_30': [.9, 1., .45],
                      'white_platform': [.55, .9, .45],
                      'dark_box': [.75, .4, .4]}
        for name, expected in dimensions.items():
            ids = np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.body(name).id)
            extents = np.einsum('nij,nj->ni', np.abs(data.geom_xmat[ids].reshape(-1, 3, 3)),
                                model.geom_size[ids])
            low = np.min(data.geom_xpos[ids]-extents, axis=0)
            high = np.max(data.geom_xpos[ids]+extents, axis=0)
            np.testing.assert_allclose(high-low, expected, atol=1.e-10)
            self.assertAlmostEqual(low[2], 0.)
        self.assertTrue(all(c.dist >= -1.e-10 for c in data.contact))

    def test_standalone_is_five_passive_free_assemblies(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        self.assertEqual((model.nq, model.nv, model.nu, model.neq), (43, 38, 0, 0))
        self.assertEqual(np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 5)
        hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self.assertEqual(len(hinges), 8)
        self.assertTrue(np.all(model.jnt_stiffness[hinges] > 0))
        self.assertTrue(np.all(model.dof_damping[model.jnt_dofadr[hinges]] > 0))
        self.assertTrue(np.all(model.body_mass[1:] > 0))
        for name in NAMES:
            self.assertEqual(model.body_parentid[model.body(name).id], 0)
            joint = model.joint(name+'_free')
            self.assertEqual(model.jnt_stiffness[joint.id], 0)
            np.testing.assert_array_equal(model.dof_damping[int(joint.dofadr[0]):int(joint.dofadr[0])+6], 0)

    def test_static_equilibrium_without_scripted_motion(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        roots = [model.body(name).id for name in NAMES]
        initial = data.xpos[roots].copy()
        advance(model, data, 3.)
        np.testing.assert_allclose(data.xpos[roots, :2], initial[:, :2], atol=.001)
        self.assertLess(np.max(np.abs(data.qvel)), .001)
        self.assertLess(np.max(np.abs(data.xpos[roots, 2]-initial[:, 2])), .002)

    def test_wood_modules_resist_small_horizontal_loads(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        for name in NAMES:
            if name == 'white_platform':
                continue
            mujoco.mj_resetData(model, data)
            advance(model, data, 1.)
            body = model.body(name).id
            before = data.xpos[body].copy()
            data.xfrc_applied[body, 0] = .3*model.body_mass[body]*9.81
            advance(model, data, .2)
            self.assertLess(np.linalg.norm(data.xpos[body, :2]-before[:2]), .002)

    def test_wood_modules_can_slide_under_external_force(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        for name in NAMES:
            # The elastic deck can absorb and recover from a short push;
            # permanent displacement is not a valid compliance criterion.
            if name == 'white_platform':
                continue
            with self.subTest(name=name):
                mujoco.mj_resetData(model, data)
                advance(model, data, 2.)
                body = model.body(name).id
                before = data.xpos[body].copy()
                data.xfrc_applied[body, 0] = 1.5 * model.body_mass[body] * 9.81
                advance(model, data, .2)
                data.xfrc_applied[:] = 0
                advance(model, data, .8)
                self.assertGreater(np.linalg.norm(data.xpos[body, :2]-before[:2]), .003)

    def test_compliant_platform_is_not_anchored_to_world(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        q = int(model.joint('white_platform_free').qposadr[0])
        data.qpos[q+2] += .5
        initial_z = data.qpos[q+2]
        advance(model, data, .15)
        self.assertGreater(initial_z-data.qpos[q+2], .08)

    def test_loaded_thin_legs_sway_more_than_identical_locked_control(self):
        locked, _, _ = support_response(locked=True)
        compliant, _, _ = support_response()
        self.assertGreater(compliant['peak_sway_m'], locked['peak_sway_m']+.001)
        self.assertGreater(compliant['peak_leg_angle_deg'], .5)
        self.assertLess(compliant['final_max_abs_qvel'], .02)
        self.assertLess(compliant['minimum_gap_m'], locked['minimum_gap_m']-.001)

    def test_platform_gap_and_height_step_have_real_collision(self):
        model, data, _ = attached(probe_radius=.09)
        deck = model.body('terrain_white_platform').pos
        dark = model.body('terrain_dark_box').pos
        white_y = deck[1]-.45
        dark_y = dark[1]+.2
        self.assertAlmostEqual(white_y-dark_y, .055)
        def height(y):
            distance, _, _ = terrain_ray(model, data, np.array([deck[0], y, 3.]),
                                         np.array([0., 0., -1.]), GROUPS)
            return 3.-distance
        self.assertAlmostEqual(height(deck[1]), .45)
        self.assertAlmostEqual(height(dark[1]), .40)
        self.assertAlmostEqual(height((white_y+dark_y)/2), 0.)
        # A wheel travelling +Y from the dark top hits the higher white lip.
        # It must not see a hidden ramp or collision-free painted edge.
        probe = model.geom('probe_geom').id
        data.qpos[:3] = [deck[0], dark_y-.015, .49]
        mujoco.mj_forward(model, data)
        lip = model.geom('terrain_white_deck').id
        resisted = False
        for i, c in enumerate(data.contact):
            if set(c.geom) != {probe, lip}:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            force = c.frame.reshape(3, 3).T @ wrench[:3]
            if c.geom[0] == probe:
                force = -force
            resisted |= force[1] < -1.
        self.assertTrue(resisted, 'White lip must resist forward wheel motion across the gap')

    def test_platform_rocks_and_settles_after_short_torque(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        advance(model, data, 2.)
        body = model.body('white_platform').id
        # Deliberate test input, not a calibrated real robot load.
        data.xfrc_applied[body, 4] = 65.
        advance(model, data, .1)
        angle = np.arccos(np.clip(data.xmat[body, 8], -1, 1))
        self.assertGreater(angle, np.deg2rad(.1))
        data.xfrc_applied[:] = 0
        advance(model, data, 3.)
        self.assertLess(np.arccos(np.clip(data.xmat[body, 8], -1, 1)), np.deg2rad(.1))

    def test_height_rays_follow_translation_and_rotation(self):
        model, data, _ = attached()
        center = model.body('terrain_white_platform').pos
        def height(x):
            distance, geom, _ = terrain_ray(model, data, np.array([x, center[1], 3.]),
                                            np.array([0., 0., -1.]), GROUPS)
            self.assertTrue(is_terrain_geom(model, geom))
            return 3.-distance
        self.assertAlmostEqual(height(center[0]), .45)
        joint = model.joint('terrain_white_platform_free')
        q = int(joint.qposadr[0])
        data.qpos[q+1] += 3.
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(height(center[0]), 0.)
        mujoco.mj_resetData(model, data)
        angle = .1
        data.qpos[q+3:q+7] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(height(center[0]), .45/np.cos(angle), places=6)

    def test_probe_load_has_real_contact_force_on_moving_platform(self):
        model, data, _ = attached()
        probe = model.body('probe').id
        advance(model, data, 1.)
        upward = 0.
        for i, contact in enumerate(data.contact):
            b1, b2 = terrain_contact_bodies(model, contact, ())
            if (b1, b2) not in ((probe, 0), (0, probe)):
                continue
            # Assert actual dynamic terrain, not the floor below it.
            terrain_geom = contact.geom[0 if b1 == 0 else 1]
            self.assertNotEqual(model.geom_bodyid[terrain_geom], 0)
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            world = contact.frame.reshape(3, 3).T @ force[:3]
            upward += world[2] if b2 == probe else -world[2]
        self.assertAlmostEqual(upward, 2*9.81, delta=1.)
        self.assertAlmostEqual(data.xpos[probe, 2], .5, delta=.003)
        self.assertFalse(is_terrain_geom(model, model.geom('probe_geom').id))

    def test_invalid_dynamic_fragments_are_rejected(self):
        for mutation in ('no_opt_in', 'wrong_count', 'fixed_component', 'collision_group', 'unnamed'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                tree = ET.parse(XML)
                flag = tree.find('./custom/numeric[@name="dynamic_components"]')
                body = tree.find('./worldbody/body')
                if mutation == 'no_opt_in':
                    tree.find('./custom').remove(flag)
                elif mutation == 'wrong_count':
                    flag.set('data', '4')
                elif mutation == 'fixed_component':
                    body.remove(body.find('freejoint'))
                elif mutation == 'collision_group':
                    body.find('geom').set('group', '1')
                else:
                    del body.find('geom').attrib['name']
                path = Path(tmp)/'invalid.xml'
                tree.write(path)
                with self.assertRaises(ValueError):
                    attach_terrain(robot_spec(), path)

    def test_robot_cannot_impersonate_terrain_by_name(self):
        spec = robot_spec()
        spec.geom('probe_geom').name = 'terrain_robot_pad'
        spec.geom('terrain_robot_pad').group = 0
        with self.assertRaisesRegex(ValueError, 'reserved terrain_'):
            attach_terrain(spec, XML)

    def test_undeclared_or_nonpassive_internal_motion_is_rejected(self):
        for mutation in ('no_declaration', 'wrong_count', 'zero_stiffness', 'zero_damping',
                         'unlimited', 'free_child', 'mocap', 'gravity_disabled', 'actuator',
                         'root_stiffness', 'root_damping', 'root_frictionloss'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                tree = ET.parse(XML)
                flag = tree.find('./custom/numeric[@name="passive_internal_joints"]')
                body = tree.find('./worldbody/body[@name="white_platform"]/body')
                joint = body.find('joint')
                if mutation == 'no_declaration':
                    tree.find('./custom').remove(flag)
                elif mutation == 'wrong_count':
                    flag.set('data', '7')
                elif mutation == 'zero_stiffness':
                    joint.set('stiffness', '0')
                elif mutation == 'zero_damping':
                    joint.set('damping', '0')
                elif mutation == 'unlimited':
                    joint.set('limited', 'false')
                elif mutation == 'free_child':
                    joint.set('type', 'free')
                elif mutation == 'mocap':
                    body.set('mocap', 'true')
                elif mutation == 'gravity_disabled':
                    body.set('gravcomp', '1')
                elif mutation == 'actuator':
                    ET.SubElement(ET.SubElement(tree.getroot(), 'actuator'), 'motor', joint=joint.get('name'))
                else:
                    root = tree.find('./worldbody/body[@name="white_platform"]')
                    free = root.find('freejoint')
                    root.remove(free)
                    ET.SubElement(root, 'joint', name=free.get('name'), type='free',
                                  **{mutation.removeprefix('root_'): '100'})
                path = Path(tmp)/'invalid.xml'
                tree.write(path)
                with self.assertRaises(ValueError):
                    attach_terrain(robot_spec(), path)

    def test_m20_scan_and_wheel_force_readers_use_movable_terrain(self):
        # A portable, minimal M20 naming/actuator contract, not a policy test.
        from types import SimpleNamespace
        from dataclasses import asdict
        import deploy_mujoco as deployment
        from gait_metrics import GaitMetrics
        from quiet_landing_metrics import QuietMujocoMetrics
        from stair_trot_adapter import StairTrotMujocoController
        from task_support.stair_trot_v1.signals import TrotSettings
        parts = []
        for i, name in enumerate(deployment.JOINT_NAMES):
            wheel = i >= 12
            x = .17 if i in (12, 13) else -.17
            y = .16 if i in (12, 14) else -.16
            pos = f'{x} {y} -.3' if wheel else '0 0 .3'
            parts.append(f'<body name="{name.removesuffix("_joint")}" pos="{pos}">'
                         f'<joint name="{name}" axis="0 1 0"/>'
                         f'<geom type="sphere" size="{.09 if wheel else .005}" mass="1" '
                         f'group="1" contype="{int(wheel)}" conaffinity="{int(wheel)}"/></body>')
        motors = ''.join(f'<motor name="{n}" joint="{n}"/>' for n in deployment.JOINT_NAMES)
        spec = mujoco.MjSpec.from_string(
            '<mujoco><option timestep=".005"/><worldbody><body name="base_link">'
            '<freejoint name="floating_base"/><geom type="box" size=".2 .1 .1" mass="10" group="1"/>'
            + ''.join(parts) + '</body></worldbody><actuator>'+motors+'</actuator></mujoco>')
        terrain = attach_terrain(spec, XML)
        model = spec.compile()
        data = mujoco.MjData(model)
        contract = deployment.M20Contract(model, terrain)
        contract.reset(data)
        deck = model.body('terrain_white_platform').pos
        data.qpos[:3] = [deck[0], deck[1], .835]
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(contract.ground_height(data, *deck[:2]), .45)
        self.assertTrue(np.isfinite(contract.height_scan(data)).all())
        self.assertEqual(contract.height_scan(data).shape, (187,))
        self.assertTrue((GaitMetrics(contract).forces(data)[:, 2] > 0).all())
        actor = SimpleNamespace(m20_training_manifest={'trot_settings': asdict(TrotSettings())})
        controller = StairTrotMujocoController(contract, actor, .02)
        controller.sample_contacts(data)
        self.assertTrue((controller.forces[:, 2] > 0).all())
        quiet = QuietMujocoMetrics(model, contract)
        quiet.post_step(data, quiet.pre_step(data))
        self.assertTrue((quiet.current_force[:, 2] > 0).all())
        joint = model.joint('terrain_white_platform_free')
        data.qpos[int(joint.qposadr[0])+1] += 3.
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(contract.ground_height(data, *deck[:2]), 0.)
        self.assertTrue(np.isfinite(contract.height_scan(data)).all())


if __name__ == '__main__':
    unittest.main()
