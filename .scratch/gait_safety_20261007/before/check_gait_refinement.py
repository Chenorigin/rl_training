#!/usr/bin/env python3
"""Adversarial CPU checks of real reward bodies (simulator imports stubbed).

These synthetic states verify credit and gates; they cannot establish that PPO
will discover the requested gait or that a policy is safe on physical hardware.
"""
import argparse
import json
from pathlib import Path

import torch
from check_stair_rewards import Entity, MDP, context, load_functions, make_env, quat_apply, yaw_quat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ns = dict(torch=torch, SceneEntityCfg=Entity, quat_apply=quat_apply, yaw_quat=yaw_quat,
              get_gait_level_tensor=lambda env: torch.ones(env.num_envs),
              joint_pos_penalty=lambda *a, **kw: torch.ones(a[0].num_envs))
    load_functions(MDP/'stair_teacher.py', ns)
    wheels, scanner, contact = Entity('robot'), Entity('height_scanner'), Entity('contact_forces')
    hips, knees = Entity('robot', body_ids=[4,5,6,7]), Entity('robot', joint_ids=[0,1,2,3])
    params = dict(asset_cfg=wheels, hip_cfg=hips, knee_cfg=knees, sensor_cfg=scanner, contact_sensor_cfg=contact)
    result = {}

    def env_for(down=True, folded=False, swing=False, invalid=False, command=(.5,0,0)):
        env = make_env()
        w = env.scene['robot'].data.body_pos_w
        hip = w.clone()
        hip[..., 2] += .30 if folded else .45
        env.scene['robot'].data.body_pos_w = torch.cat((w, hip), dim=1)
        env.scene['robot'].data.joint_pos = torch.full((1,4), 2.2 if folded else 1.5)
        env.scene['robot'].data.root_ang_vel_b = torch.tensor([[0.,0.,command[2]]])
        env.command_manager.get_command('base_velocity')[:] = torch.tensor([command])
        env.scene['height_scanner'].data.ray_hits_w[..., 2] = 0 if not invalid else torch.inf
        env.scene['contact_forces'].data.current_contact_time[:] = 0 if swing else 1
        env.scene['contact_forces'].data.last_air_time = torch.full((1,4), .2)
        env.scene['contact_forces'].compute_first_contact = lambda _: torch.ones(1,4,dtype=torch.bool)
        if swing:
            env.scene['contact_forces'].data.net_forces_w.zero_()
        context(env)
        c = env._m20_stair_scan_context['context']
        c['up_gate'].fill_(float(not down)); c['down_gate'].fill_(float(down))
        return env

    for name, kwargs, expected in (
        ('normal_down', {}, False), ('folded_down', dict(folded=True), True),
        ('folded_swing', dict(folded=True,swing=True), False),
        ('folded_up', dict(folded=True,down=False), False),
        ('invalid_scan', dict(folded=True,invalid=True), False),
        ('zero_command', dict(folded=True,command=(0,0,0)), False),
    ):
        cost = ns['descent_rear_fold_cost'](env_for(**kwargs), **params)
        assert torch.isfinite(cost).all() and bool(cost.item()>0) == expected, (name,cost)
        result[name+'_cost'] = float(cost)
    env = env_for(folded=True)
    env._m20_stair_scan_context['context']['down_gate'].zero_()
    env._m20_stair_step_state = {'active':torch.tensor([True]),'up':torch.tensor([False])}
    result['rear_follow_cost_after_scan_flat'] = float(ns['descent_rear_fold_cost'](env, **params))
    assert result['rear_follow_cost_after_scan_flat'] > 0
    def front_cost(env):
        cost = ns['ascent_front_fold_cost'](env, **params)
        assert torch.isfinite(cost).all() and 0 <= float(cost) <= 1
        return float(cost)
    for name, kwargs, positive in (
        ('front_normal_up', dict(down=False), False),
        ('front_folded_up', dict(down=False,folded=True), True),
        ('front_folded_swing', dict(down=False,folded=True,swing=True), False),
        ('front_folded_down', dict(down=True,folded=True), False),
        ('front_invalid_scan', dict(down=False,folded=True,invalid=True), False),
        ('front_zero_command', dict(down=False,folded=True,command=(0,0,0)), False),
        ('front_reverse_command', dict(down=False,folded=True,command=(-.5,0,0)), False),
        ('front_pure_yaw', dict(down=False,folded=True,command=(0,0,.6)), False),
    ):
        env = env_for(**kwargs)
        result[name+'_cost'] = front_cost(env)
        assert (result[name+'_cost'] > 0) == positive, (name,result[name+'_cost'])
    env = env_for(down=False,folded=True)
    env._m20_stair_scan_context['context']['up_gate'].zero_()
    result['front_flat_cost'] = front_cost(env)
    assert result['front_flat_cost'] == 0
    env = env_for(down=False,folded=True)
    env.scene['contact_forces'].data.current_contact_time.fill_(.15)
    result['front_landing_buffer_cost'] = front_cost(env)
    assert result['front_landing_buffer_cost'] == 0
    for name, command, invalid, positive in (
        ('extreme_front_up_swing',(.5,0,0),False,True),
        ('extreme_front_zero_swing',(0,0,0),False,False),
        ('extreme_front_invalid_swing',(.5,0,0),True,False),
    ):
        env = env_for(down=False,swing=True,invalid=invalid,command=command)
        env.scene['robot'].data.body_pos_w[:,4:6] = env.scene['robot'].data.body_pos_w[:,:2]
        env.scene['robot'].data.body_pos_w[:,4:6,2] += .18
        env.scene['robot'].data.joint_pos[:,:2] = 2.7
        result[name+'_cost'] = front_cost(env)
        assert (result[name+'_cost']>0) == positive
    env = env_for(down=False,folded=True)
    env.scene['contact_forces'].data.net_forces_w[:,:,:] = torch.tensor([100.,0.,1.])
    result['front_riser_face_only_cost'] = front_cost(env)
    assert result['front_riser_face_only_cost'] == 0
    env = env_for(down=False,folded=True)
    env._m20_stair_scan_context['context']['up_gate'].zero_()
    env._m20_ascent_state = {'active':torch.tensor([True])}
    result['front_active_ascent_scan_transition_cost'] = front_cost(env)
    assert result['front_active_ascent_scan_transition_cost'] > 0
    env = env_for(down=False,folded=True)
    env._m20_stair_scan_context['context']['up_gate'].zero_()
    env._m20_stair_scan_context['context']['ascending_edges_valid'] = torch.tensor([[True]])
    result['front_recent_riser_cost'] = front_cost(env)
    assert result['front_recent_riser_cost'] > 0
    env = env_for(down=False,folded=True)
    env.episode_length_buf.fill_(0)
    env._m20_stair_scan_context['context']['up_gate'].zero_()
    env._m20_ascent_state = {'active':torch.tensor([True])}
    result['front_reset_stale_target_cost'] = front_cost(env)
    assert result['front_reset_stale_target_cost'] == 0
    # One supported front leg folds, then mirror FL/FR: equal total cost.
    env = env_for(down=False,folded=True)
    env.scene['contact_forces'].data.current_contact_time[0,1] = 0
    left_cost = front_cost(env)
    env = env_for(down=False,folded=True)
    env.scene['contact_forces'].data.current_contact_time[0,0] = 0
    right_cost = front_cost(env)
    result['front_single_leg_cost'] = left_cost
    assert left_cost == right_cost and left_cost > 0
    # A whole-body rotation preserves hip-wheel length and cost.
    env = env_for(down=False,folded=True)
    points = env.scene['robot'].data.body_pos_w
    points[:,:,:2] = torch.stack((-points[:,:,1],points[:,:,0]),dim=-1)
    hits = env.scene['height_scanner'].data.ray_hits_w
    hits[:,:,:2] = torch.stack((-hits[:,:,1],hits[:,:,0]),dim=-1)
    result['front_rotated_cost'] = front_cost(env)
    assert abs(result['front_rotated_cost']-result['front_folded_up_cost']) < 1e-6
    for length, knee, expected in ((.4,1.5,False),(.3,1.5,True),(.4,2.2,True)):
        env = env_for(down=False)
        env.scene['robot'].data.body_pos_w[:,4:6] = env.scene['robot'].data.body_pos_w[:,:2]
        env.scene['robot'].data.body_pos_w[:,4:6,2] += length
        env.scene['robot'].data.joint_pos[:,:2] = knee
        assert (front_cost(env)>0) == expected
    air = torch.tensor([0., .10, .20, .35, 1., 5.])
    result['air_scores'] = ns['bounded_air_score'](air, .10, .20, .35).tolist()
    assert result['air_scores'][2] > 0 and max(result['air_scores'][3:]) == 0
    for name, cmd in [('turn',(0,0,.6)), ('traverse',(.3,0,.2)), ('side',(0,.3,.2))]:
        env = env_for(down=False, swing=True, command=cmd)
        env.scene['robot'].data.body_pos_w[:,:4,2] = .29
        result[name+'_swing_cost'] = float(ns['turn_swing_size_cost'](env, **params))
        result[name+'_air_credit'] = float(ns['compact_turn_air_time'](
            env,'base_velocity',contact,wheels,.2,scanner))
    assert result['turn_swing_cost'] > 0 and result['turn_air_credit'] > 0
    assert result['traverse_swing_cost'] == result['traverse_air_credit'] == 0
    assert result['side_swing_cost'] == result['side_air_credit'] == 0
    result['static_turn_quality'] = float(ns['turn_tracking_quality'](torch.tensor([0.]),torch.tensor([.6])))
    result['opposite_turn_quality'] = float(ns['turn_tracking_quality'](torch.tensor([-.6]),torch.tensor([.6])))
    assert result['static_turn_quality'] == result['opposite_turn_quality'] == 0
    env = env_for(down=False, command=(0,0,.6))
    # Two groups on different world-height treads, all physically grounded.
    result['all_contact_turn_status'] = float(ns['compact_turn_status'](env, **params))
    assert result['all_contact_turn_status'] == 0
    result['weak_turn_posture_cost'] = float(ns['stair_joint_pos_cost'](
        env,'base_velocity',wheels,5.,.1,.1,scanner,.1,True,turn_penalty_scale=.2))
    assert result['weak_turn_posture_cost'] > 0

    # Exercise the real ledger on a delayed same-tread landing, then a platform.
    env = make_env()
    state = ns['_new_state'](env)
    state['radius'].fill_(.09)
    state['front_count'].fill_(1)
    state['front_history_heading'][0,0,0] = 1
    state['front_history_target_z'][0,0] = .1
    state['front_expected'].fill_(1)
    env._m20_ascent_state = state
    env.scene['robot'].data.body_pos_w[0,:2,0] = .1
    env.scene['robot'].data.body_pos_w[0,:2,2] = .19
    def advance(up=True, upper=.2):
        env.common_step_counter += 1
        context(env,edge=.3,height=upper)
        c = env._m20_stair_scan_context['context']
        c['up_gate'].fill_(float(up)); c['gate'].fill_(float(up))
        return ns['ascent_state'](env,wheels,scanner,contact)
    s = advance()
    result['intermediate_pair_event'] = bool(s['front_same_tread_event'])
    result['early_dwell'] = float(s['same_tread_dwell'][0,0])
    for _ in range(10): s = advance()
    result['persistent_dwell'] = float(s['same_tread_dwell'][0,0])
    result['one_shot_event_total'] = float(s['same_tread_events_total'][0,0])
    assert result['intermediate_pair_event'] and result['early_dwell'] == 0
    assert result['persistent_dwell'] == 1 and result['one_shot_event_total'] == 1
    env.scene['contact_forces'].data.current_contact_time[0,1] = 0
    env.scene['contact_forces'].data.net_forces_w[0,1].zero_()
    advance()
    env.scene['contact_forces'].data.current_contact_time[0,1] = 1
    env.scene['contact_forces'].data.net_forces_w[0,1,2] = 100
    s = advance()
    result['bounce_does_not_renew_grace'] = float(s['same_tread_dwell'][0,0])
    assert result['bounce_does_not_renew_grace'] == 1
    env.scene['height_scanner'].data.quat_w[0] = torch.tensor([.9396926,0.,0.,.3420201])
    advance()
    env.scene['height_scanner'].data.quat_w[0] = torch.tensor([1.,0.,0.,0.])
    s = advance()
    result['turn_does_not_renew_grace'] = float(s['same_tread_dwell'][0,0])
    assert result['turn_does_not_renew_grace'] == 1
    env.command_manager.get_command('base_velocity').zero_()
    env.common_step_counter += 1; context(env,edge=.3,height=.2)
    env._m20_stair_scan_context['context']['command_x'].zero_()
    ns['ascent_state'](env,wheels,scanner,contact)
    env.command_manager.get_command('base_velocity')[0,0] = .5
    s = advance()
    result['pause_does_not_renew_grace'] = float(s['same_tread_dwell'][0,0])
    assert result['pause_does_not_renew_grace'] == 1
    s = advance(up=False,upper=.1)
    result['platform_event'] = bool(s['front_same_tread_event'])
    result['platform_dwell'] = float(s['same_tread_dwell'].sum())
    assert not result['platform_event'] and result['platform_dwell'] == 0
    # Rear still on tread1, front physically recorded tread2, scanner on flat.
    s['front_count'].fill_(2); s['rear_count'].fill_(1)
    s['front_history_target_z'][0,1] = .2
    s['front_history_heading'][0,1,0] = 1
    s['front_history_xy'][0,1,0] = .3
    env.scene['robot'].data.body_pos_w[0,2:,0] = .1
    env.scene['robot'].data.body_pos_w[0,2:,2] = .19
    for _ in range(10): s = advance(up=False,upper=.1)
    result['rear_dwell_front_on_platform'] = float(s['same_tread_dwell'][0,1])
    assert result['rear_dwell_front_on_platform'] == 1
    # Clean final platform with a fresh ledger must not incur the event either.
    s['rear_count'].fill_(2)
    env.scene['robot'].data.body_pos_w[0,2:,0] = .5
    env.scene['robot'].data.body_pos_w[0,2:,2] = .29
    s = advance(up=False,upper=.2)
    result['rear_final_platform_event'] = bool(s['rear_same_tread_event'])
    assert not result['rear_final_platform_event']
    env.episode_length_buf.fill_(0)
    s = advance(up=False,upper=.2)
    result['reset_dwell'] = float(s['same_tread_time'].sum())
    assert result['reset_dwell'] == 0
    # Real scan counterexample: the trunk still sees the first edge as nearest
    # while both front wheels are already on tread1 and tread2 is visible.
    for second_riser in (True, False):
        env = make_env()
        env.scene['height_scanner'].data.pos_w[0,0] = -.25
        xy = env.scene['height_scanner'].ray_starts[:, :2]
        hits = env.scene['height_scanner'].data.ray_hits_w
        hits[0,:,:2] = xy + torch.tensor([-.25,0.])
        hits[...,2] = torch.where(hits[...,0] >= 0, .1, 0.)
        if second_riser:
            hits[...,2] = torch.where(hits[...,0] >= .3, .2, hits[...,2])
        env.scene['robot'].data.body_pos_w[0,:2,0] = .1
        env.scene['robot'].data.body_pos_w[0,:2,2] = .19
        state = ns['_new_state'](env)
        state['radius'].fill_(.09); state['front_count'].fill_(1)
        state['front_history_heading'][0,0,0] = 1
        state['front_history_target_z'][0,0] = .1
        env._m20_ascent_state = state
        s = ns['ascent_state'](env,wheels,scanner,contact)
        name = 'real_scan_next_step' if second_riser else 'real_scan_terminal_platform'
        result[name] = bool(s['front_same_tread_event'])
        assert result[name] == second_riser
    result['passed'] = True
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
