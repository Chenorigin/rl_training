#!/usr/bin/env python3
"""Independent CPU event-payment and discounted same-tread counterexamples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/tools'))
import torch
from check_gait_safety import load
from check_stair_rewards import Entity, make_env

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py')
parser.add_argument('--assert-new', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
ns = load(args.source)
params = dict(asset_cfg=Entity('robot'), sensor_cfg=Entity('height_scanner'), contact_sensor_cfg=Entity('contact_forces'))


def make(front_ahead=True):
    env = make_env()
    robot = env.scene['robot'].data
    if front_ahead:
        robot.body_pos_w[0, :2, 0] = .6
        robot.body_pos_w[0, :2, 2] = .69
    trace = []

    def tick(side=None, x=None, z=None, touch=True, face=False):
        env.common_step_counter += 1
        env.episode_length_buf += 1
        if side is not None:
            robot.body_pos_w[0, side, 0] = x
            robot.body_pos_w[0, side, 2] = z
            env.scene['contact_forces'].data.current_contact_time[0, side] = .1 if touch else 0
            env.scene['contact_forces'].data.net_forces_w[0, side] = torch.tensor(
                [100., 0., 1.] if face else [0., 0., 100. if touch else 0.])
        scanner = env.scene['height_scanner'].data
        robot.root_pos_w[0, 0] = robot.body_pos_w[0, :2, 0].mean() - .35
        scanner.pos_w[0, 0] = robot.root_pos_w[0, 0]
        local_edges = torch.arange(8) * .3 - scanner.pos_w[0, 0]
        visible = (local_edges >= -.45) & (local_edges <= .70)
        future = visible & (local_edges >= .05)
        nearest = int(torch.nonzero(future)[0]) if future.any() else 0
        ctx = dict(edge_x=local_edges[nearest:nearest + 1],
                   upper_z=torch.tensor([.1 * (nearest + 1)]), lower_z=torch.tensor([.1 * nearest]),
                   up_gate=torch.tensor([float(future.any())]), command_x=torch.tensor([.5]),
                   ascending_edges_x=local_edges, ascending_edges_valid=visible[None],
                   ascending_edges_upper_z=(torch.arange(8) + 1)[None] * .1,
                   ascending_edges_lower_z=torch.arange(8)[None] * .1)
        env._m20_stair_scan_context = dict(counter=env.common_step_counter, sensor_name='height_scanner', context=ctx)
        state = ns['ascent_state'](env, **params)
        trace.append(dict(front=int(state['front_event'].sum()), rear=int(state['rear_event'].sum()),
                          prep_front=float(state['front_lift_gain']), prep_rear=float(state['rear_lift_gain']),
                          duplicates=state['duplicate_event_count'][0].tolist()))
        return state

    def land(side, stair):
        edge = stair * .3
        height = (stair + 1) * .1
        tick(side, edge - .12, float(robot.body_pos_w[0, side, 2]), True)
        tick(side, edge - .04, height + .15, False)
        tick(side, edge + .08, height + .15, False)
        tick(side, edge + .10, height + .09, True)
        return tick(side, edge + .10, height + .09, True)

    tick()
    return env, tick, land, trace


def counts(state, trace):
    return dict(front_main_events=sum(t['front'] for t in trace), rear_main_events=sum(t['rear'] for t in trace),
                front_first_geometry=float(state['front_successes']), rear_first_geometry=float(state['rear_successes']),
                duplicates=state['same_tread_events_total'][0].tolist(),
                transition_counters=[float(state.get(name + '_transition_successes', torch.tensor([-1.]))[0])
                                     for name in ('front', 'rear')])


out = {}
# Correct rear alternation must remain rewardable without any front completion.
for first in (0, 1):
    env, tick, land, trace = make()
    for i in range(4):
        state = land(2 + (first if i % 2 == 0 else 1 - first), i)
        if i == 0:
            out[f'rear_lead{first}_first_seed'] = dict(main_events=sum(t['rear'] for t in trace),
                prep_units=sum(t['prep_rear'] for t in trace))
            if args.assert_new:
                assert out[f'rear_lead{first}_first_seed']['main_events'] == 0
                assert 0 < out[f'rear_lead{first}_first_seed']['prep_units'] <= 1
    out[f'pure_rear_lead{first}'] = counts(state, trace)
    if args.assert_new:
        assert out[f'pure_rear_lead{first}']['rear_main_events'] == 3
        assert state['front_successes'] == 0 and state['rear_successes'] == 4

for first in (0, 1):
    env, tick, land, trace = make(front_ahead=False)
    for i in range(4):
        state = land(first if i % 2 == 0 else 1 - first, i)
    out[f'pure_front_lead{first}'] = counts(state, trace)
    if args.assert_new:
        assert out[f'pure_front_lead{first}']['front_main_events'] == 3
        assert state['front_successes'] == 4 and state['rear_successes'] == 0

# The exact user failure, with delayed opposing-foot catch-up on every tread.
for hold_seconds in (.6, 1.):
    env, tick, land, trace = make()
    for i in range(3):
        land(2, i)
        for _ in range(round(hold_seconds / env.step_dt)):
            tick()
        state = land(3, i)
    result = counts(state, trace)
    gamma = .99
    style = [3 * t['rear'] + .5 * t['prep_rear'] - 4.5 * t['duplicates'][1] for t in trace]
    result['discounted_rear_completion_prep_duplicate'] = sum(gamma ** k * r for k, r in enumerate(style))
    result['bounded_rear_prep_units'] = sum(t['prep_rear'] for t in trace)
    out[f'rear_same_every_tread_hold_{hold_seconds}s'] = result
    if args.assert_new:
        assert result['rear_main_events'] == 0
        assert result['duplicates'][1] == 3
        assert result['discounted_rear_completion_prep_duplicate'] < 0

# Duplicate previous tread removes transition pay on the immediately following
# single landing; the next genuinely opposite adjacent landing may recover.
env, tick, land, trace = make()
land(2, 0)
land(3, 0)
state = land(2, 1)
out['first_after_duplicate'] = counts(state, trace)
state = land(3, 2)
out['second_after_duplicate'] = counts(state, trace)
if args.assert_new:
    assert out['first_after_duplicate']['rear_main_events'] == 0
    assert out['second_after_duplicate']['rear_main_events'] == 1

# Adjacent safe alternation after an earlier skip can recover; it must not be
# dependent on the previous step having received a correct/completion reward.
env, tick, land, trace = make()
land(2, 0)
state = land(3, 2)
out['skip_without_pay'] = counts(state, trace)
state = land(2, 3)
out['adjacent_after_skip_recovers'] = counts(state, trace)
if args.assert_new:
    assert out['skip_without_pay']['rear_main_events'] == 0
    assert out['adjacent_after_skip_recovers']['rear_main_events'] == 1

# During a zero command geometry/contact history continues, while payments
# pause. On resumption a previously uncharged duplicate and the next candidate
# can be processed in the same frame; current previous_pair must veto payment.
env, tick, land, trace = make()
land(2, 0)
env.command_manager.get_command('base_velocity').zero_()
land(3, 0)
state = land(3, 1)
out['before_same_frame_duplicate_resume'] = counts(state, trace)
env.command_manager.get_command('base_velocity')[0, 0] = .5
state = tick()
out['same_frame_previous_duplicate'] = counts(state, trace)
out['same_frame_previous_duplicate']['resume_duplicate_events'] = float(state['duplicate_event_count'][0, 1])
if args.assert_new:
    assert out['same_frame_previous_duplicate']['resume_duplicate_events'] == 1
    assert out['same_frame_previous_duplicate']['rear_main_events'] == 0
    assert state['rear_successes'] == 2

out['source_sha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
args.output.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
