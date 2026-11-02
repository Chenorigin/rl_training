"""Wall hold tests use real course/geometry/accounting with fake physics.

These counterexamples validate gates and wiring, not a trained controller.
"""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.all_terrain_geometry import build_atlas
from task_support.cts.wall_practice import collision_document
from task_support.cts.wall_hold_practice import validate
from task_support.student_v1.perception import quaternion_rotation
import test_cts_all_terrain_course as helpers
from test_cts_objective_consistency import reward_env


@pytest.fixture(scope='module')
def source():
    settings = default_settings()
    c = settings['all_terrain']
    c['wall_hold_practice'].pop('handover', None)  # Legacy direct-hold regression fixture.
    for key in ('stair_recovery', 'stair_stop', 'motion_quality', 'objective_consistency',
                'stair_walls', 'hind_ascent', 'front_descent', 'wall_stability'):
        c.pop(key, None)
    c['stair_hold_practice']['enabled'] = False  # Keep the single verified FK ledger.
    c['geometry'].update(samples_per_level=3, geometry_levels=1, grid_resolution=.25)
    c['curriculum']['geometry_levels'] = 1
    return settings, build_atlas(c['geometry'], c['wall_practice'])


@pytest.fixture
def setup(source, monkeypatch):
    settings, atlas = source
    names = list(dict.fromkeys(g['body'] for g in collision_document(settings['all_terrain']['wall_practice'])['primitives']))
    wheels = [names.index(leg+'_wheel') for leg in ('fl', 'fr', 'hl', 'hr')]
    class Robot(helpers.Robot):
        def __init__(self, n):
            super().__init__(n)
            for key in ('body_link_pos_w', 'body_com_pos_w', 'body_lin_vel_w', 'body_ang_vel_w'):
                setattr(self.data, key, torch.zeros(n, len(names), 3))
            self.data.body_link_quat_w = torch.zeros(n, len(names), 4)
            self.data.body_link_quat_w[..., 0] = 1
            self.data.root_lin_vel_w = self.data.root_lin_vel_b
            self.data.default_joint_pos = torch.zeros(n, 16)
        def find_bodies(self, requested, preserve_order=False):
            return [names.index(name) for name in requested], requested
        def write_joint_state_to_sim(self, p, v, env_ids):
            pass
    def support(env, s):
        d = env.scene.robot.data
        d.body_link_pos_w[:] = d.root_pos_w[:, None]
        local = torch.tensor(settings['all_terrain']['stair_hold_practice']['wheel_centers_body_m'])
        rot = quaternion_rotation(d.root_quat_w)
        d.body_link_pos_w[:, wheels] = (rot@local.T).transpose(1, 2)+d.root_pos_w[:, None]
        d.body_com_pos_w.copy_(d.body_link_pos_w)
        d.body_link_quat_w[:] = d.root_quat_w[:, None]
        env.scene.sensors['contact_forces'].data.net_forces_w[:] = torch.tensor([0., 0., 100.])
        s.measure_tick = -1
    monkeypatch.setattr(helpers, 'Robot', Robot)
    monkeypatch.setattr(helpers, 'support', support)
    env, s = helpers.fake((settings, atlas), n=100)
    reward_env(env)
    env.scene.robot.find_joints = lambda requested, preserve_order=False: (list(range(12)), requested)
    h = s.wall_hold
    i = int((h.active & ~s.teacher).nonzero()[0])
    return env, s, h, i, wheels, support


def step(env, s, age=.4):
    helpers.frame(env, s, age, velocity=None)


def test_persistent_two_wall_zero_command_and_safe_spawn(setup):
    env, s, h, i, wheels, support = setup
    receipt = h.validate_initial()
    assert receipt['teacher'] > 0 and receipt['student'] > 0
    active = h.active.clone()
    for _ in range(8):
        assert s.raw_command()[active].abs().max() == 0
        assert all(s.atlas.records[t]['wall_kind']=='corridor' for t in s.tile[active].tolist())
        assert (s.mode[active] == 20).all()
        assert not s.tags()['cts_anchor'][active].any()
        s.reset(torch.arange(env.num_envs));support(env, s)
        torch.testing.assert_close(h.active, active)
    assert (s.walls.active & ~active).any()  # Ordinary traversal remains exposed.


def test_stationary_passes_continuous_gate_and_independent_ledger(setup):
    env, s, h, i, wheels, _ = setup
    original = s.episodes.clone()
    for k in range(151):step(env, s, .3+k*.02)
    assert h.ready[i] and s.success[i] and h.steps[i]>=h.required[i]
    assert not s.walls.done[i] and s.progress[i]==0
    env.termination_manager.terminated[i] = True
    s.record(torch.tensor([i]));s.record(torch.tensor([i]))
    assert h.totals['episodes'][1] == h.totals['successes'][1] == 1
    torch.testing.assert_close(s.episodes, original)
    assert s.curriculum.diagnostics()[int(s.group[i])]['student_episodes']==0
    saved = s.snapshot();s.restore(saved)
    torch.testing.assert_close(h.totals['episodes'], saved['wall_hold_practice']['totals']['episodes'])


@pytest.mark.parametrize('case', ['slow_back', 'body_shift', 'single_wheel_shift', 'two_supports',
    'front_only', 'slip', 'yaw_rate', 'roll_rate', 'nonzero', 'nonfinite', 'wall_collision'])
def test_failure_counterexamples(setup, case):
    env, s, h, i, wheels, _ = setup
    d = env.scene.robot.data
    step(env,s);assert h.ready[i]
    baseline=float(h.cost[i])
    if case=='slow_back':d.root_lin_vel_w[i,0]=-.04
    elif case=='body_shift':d.root_pos_w[i,0]-=.021
    elif case=='single_wheel_shift':d.body_link_pos_w[i,wheels[0],0]-=.021
    elif case=='two_supports':env.scene.sensors['contact_forces'].data.net_forces_w[i,[1,2]]=0
    elif case=='front_only':env.scene.sensors['contact_forces'].data.net_forces_w[i,2:]=0
    elif case=='slip':d.body_lin_vel_w[i,wheels,0]=.2
    elif case=='yaw_rate':d.root_ang_vel_b[i,2]=.2
    elif case=='roll_rate':d.root_ang_vel_b[i,0]=.5
    elif case=='nonzero':env.command_manager.get_command=lambda name: torch.ones(env.num_envs,3)*.2
    elif case=='nonfinite':d.root_pos_w[i,0]=float('nan')
    elif case=='wall_collision':
        box=s.atlas.records[int(s.tile[i])]['wall_rectangles'][0]
        d.body_link_pos_w[i,:,1]=s.origins[s.tile[i],1]+(box[2]+box[3])/2
    step(env,s,.42)
    assert not h.ready[i] and not s.success[i]
    if case in ('slow_back','body_shift','single_wheel_shift','two_supports','front_only','slip','roll_rate'):
        assert h.cost[i]>baseline
    if case in ('nonfinite','wall_collision'):assert s.failed[i]


def test_retreat_then_return_or_stop_elsewhere_never_passes(setup):
    env,s,h,i,wheels,_=setup;d=env.scene.robot.data
    anchor=h.anchor.clone();feet=h.foot_anchor.clone()
    d.root_pos_w[i,0]-=.03
    step(env,s,.1)  # Grace is not permission to move the acceptance anchor.
    assert h.escaped[i]
    d.root_pos_w[i]=anchor[i]
    for k in range(160):step(env,s,.3+k*.02)
    assert not s.success[i] and not h.ready[i]
    torch.testing.assert_close(h.anchor,anchor);torch.testing.assert_close(h.foot_anchor,feet)


def test_nonconsecutive_stable_frames_do_not_accumulate(setup):
    env,s,h,i,_,_=setup
    for k in range(160):
        env.scene.robot.data.root_lin_vel_w[i,0]=.04 if k%2 else 0.
        step(env,s,.3+k*.02)
    assert h.steps[i]<2 and not s.success[i]


def test_reward_uses_real_hold_cost_and_partial_reset_preserves_other_anchors(setup):
    from task_support.cts.all_terrain_terms import reward
    env,s,h,i,_,support=setup
    step(env,s);before=reward(env)[i].item()
    env.scene.robot.data.root_lin_vel_w[i,0]=-.05
    step(env,s,.42);after=reward(env)[i].item()
    assert after < before
    anchor=h.anchor.clone();s.reset(torch.tensor([i]));support(env,s)
    keep=torch.arange(env.num_envs)!=i
    torch.testing.assert_close(h.anchor[keep],anchor[keep])
    assert not h.escaped[i] and h.steps[i]==0


def test_atomic_resume_rejects_corrupt_hold_ledger(setup):
    env,s,h,i,_,_=setup
    saved=s.snapshot();bad=deepcopy(saved)
    bad['wall_hold_practice']['totals']['episodes'][0]=float('nan')
    with pytest.raises(ValueError,match='wall hold'):s.restore(bad)
    torch.testing.assert_close(h.totals['episodes'],saved['wall_hold_practice']['totals']['episodes'])
    torch.testing.assert_close(s.episodes,saved['episodes'])


@pytest.mark.parametrize('key,value',[('motion_environment_fraction',1.),('speed_m_s',float('nan')),
 ('position_m',.2),('timeout_s',3.),('hold_s_range',[0.,1.]),('position_scale_m',0.)])
def test_invalid_configuration_rejected(key,value):
    terrain=default_settings()['all_terrain'];c=deepcopy(terrain['wall_hold_practice']);c[key]=value
    with pytest.raises(ValueError):validate(c,terrain)


def test_legacy_v24_contract_remains_readable():
    c=default_settings();c['all_terrain'].pop('wall_hold_practice')
    validate_config(c,legacy=True)
