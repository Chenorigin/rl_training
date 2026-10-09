#!/usr/bin/env python3
"""Independent CPU regression of production rewards; no training/physics claim."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/tools'))
from check_stair_rewards import Entity, MDP, context, load_functions, make_env, quat_apply, yaw_quat

torch.set_num_threads(1)
ns = dict(torch=torch, np=np, SceneEntityCfg=Entity, quat_apply=quat_apply,
          yaw_quat=yaw_quat,
          quat_apply_inverse=lambda q, v: quat_apply(q * torch.tensor([1., -1., -1., -1.]), v))
load_functions(MDP / 'stair_teacher.py', ns)
robot, scanner, contacts = Entity('robot'), Entity('height_scanner'), Entity('contact_forces')
result = {}


def track():
    env = make_env()

    def tick(leg=None, x=None, z=None, touch=True, edge=0., height=.1):
        env.common_step_counter += 1
        env.episode_length_buf += 1
        if leg is not None:
            env.scene['robot'].data.body_pos_w[0, leg, 0] = x
            env.scene['robot'].data.body_pos_w[0, leg, 2] = z
            env.scene['contact_forces'].data.current_contact_time[0, leg] = float(touch)
            env.scene['contact_forces'].data.net_forces_w[0, leg] = torch.tensor([0., 0., 100. if touch else 0.])
        context(env, edge, height)
        return ns['ascent_state'](env, robot, scanner, contacts)

    tick()
    return env, tick


# Exactly two .02s landing frames: retirement must preserve the first arrival.
# Lead leaves immediately; the follower arrives only afterwards.
for lead in (0, 1):
    env, tick = track()
    tick(lead, -.04, .25, False)
    tick(lead, .08, .25, False)
    tick(lead, .1, .19, True)
    st = tick(lead, .1, .19, True)
    assert bool(st['front_event'].any())
    assert bool(st['tread_seen'][0, 0, 0, lead])
    tick(lead, .4, .35, False, .3, .2)
    events = []
    for _ in range(8):
        st = tick(1-lead, .1, .19, True, .3, .2)
        events.append(bool(st['front_same_tread_event']))
    assert sum(events) == 1, events
    result[f'minimal_contact_delayed_duplicate_lead_{lead}'] = events


# Completed long platform, 90-degree turn, next independent stair flight.
env = make_env()
state = ns['_new_state'](env)
env._m20_ascent_state = state
state['front_count'].fill_(1)
state['rear_count'].fill_(1)
state['radius'].fill_(.09)
state['front_history_target_z'][0, 0] = .1
state['front_history_source_z'][0, 0] = 0.
state['front_history_heading'][0, 0] = torch.tensor([1., 0.])
state['front_expected'].fill_(1)
state['rear_expected'].fill_(1)
env.scene['robot'].data.body_pos_w[0, :, 0] = torch.tensor([1.4, 1.4, 1., 1.])
env.scene['robot'].data.body_pos_w[..., 2] = .19
env.scene['robot'].data.root_pos_w[0] = torch.tensor([1.1, 0., .6])
env.scene['height_scanner'].data.pos_w[0, :2] = torch.tensor([1.1, 0.])
env.scene['height_scanner'].data.ray_hits_w[..., 2] = .1
env.scene['height_scanner'].data.quat_w[0] = torch.tensor([2**-.5, 0., 0., 2**-.5])


def turned_tick(y=None, z=None, touch=True):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    if y is not None:
        env.scene['robot'].data.body_pos_w[0, 0, 1] = y
        env.scene['robot'].data.body_pos_w[0, 0, 2] = z
        env.scene['contact_forces'].data.current_contact_time[0, 0] = float(touch)
        env.scene['contact_forces'].data.net_forces_w[0, 0] = torch.tensor([0., 0., 100. if touch else 0.])
    context(env, edge=2.1, height=.2)
    return ns['ascent_state'](env, robot, scanner, contacts)


st = turned_tick()
assert bool(st['active']) and int(st['current_flight']) == 1
assert int(st['front_expected']) == int(st['rear_expected']) == -1
turned_tick(.96, .35, False)
turned_tick(1.08, .35, False)
turned_tick(1.1, .29, True)
st = turned_tick(1.1, .29, True)
assert bool(st['front_event'].any()) and int(st['front_count']) == 2
assert st['front_history_flight'][0, :2].tolist() == [0, 1]
assert not bool(st['tread_intermediate'][0, 0])
result['new_flight_after_long_platform'] = True
result['old_top_not_intermediate'] = True


# Metrics must distinguish a single wrong-side landing from correct unique
# support, and must exclude the terminal platform from their denominator.
env = make_env()
st = ns['_new_state'](env)
env._m20_ascent_state = st
st['front_count'].fill_(3)
st['tread_intermediate'][0, :2] = True
st['tread_seen'][0, :, :3, 0] = True
st['tread_order_valid'][0, :, :3] = True
st['tread_order_valid'][0, :, 1] = False
metrics = ns['stair_step_completion_metrics'](env, torch.tensor([0]))
for axle in ('front', 'rear'):
    assert float(metrics[f'up_{axle}_unique_tread_fraction']) == 1.
    assert float(metrics[f'up_{axle}_strict_riser_fraction']) == .5
    assert float(metrics[f'up_{axle}_recorded_intermediate_treads']) == 2.
result['wrong_side_unique_fraction'] = 1.
result['wrong_side_strict_fraction'] = .5
st['tread_seen'][0, :, 0, 1] = True
metrics = ns['stair_step_completion_metrics'](env, torch.tensor([0]))
for axle in ('front', 'rear'):
    assert float(metrics[f'up_{axle}_unique_tread_fraction']) == .5
    assert float(metrics[f'up_{axle}_strict_riser_fraction']) == 0.
result['duplicate_unique_fraction'] = .5
result['duplicate_plus_wrong_order_strict_fraction'] = 0.


# Known boundary: flying over target1 and landing on2 gives neither a target1
# completion reward nor a dedicated skip penalty. Keep this explicit.
env, tick = track()
tick(0, -.04, .35, False)
tick(0, .38, .35, False)
for _ in range(8):
    st = tick(0, .4, .29, True)
assert float(st['front_successes']) == float(st['front_violations']) == 0.
assert float(st['same_tread_events_total'].sum()) == 0.
result['skip_riser_no_credit_no_explicit_cost'] = True

env = make_env()
assert torch.isfinite(ns['stair_forward_progress'](env, robot, scanner)).all()
result['production_progress_callable'] = True
print(json.dumps(result, indent=2))
