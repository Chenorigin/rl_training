#!/usr/bin/env python3
"""Execute production forward-position cost and entry context on CPU."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from check_descent_entry import check as check_entry
from check_gait_safety import load
from check_stair_rewards import MDP


def check(ns):
    upper = torch.full((7,), .22)
    x = torch.tensor([-.20, .10, .22, .23, .27, .32, .39], requires_grad=True)
    cost = ns['descent_entry_forward_position_cost'](x, upper)
    assert not cost[:3].any()
    assert (cost[4:] > cost[3:-1]).all()
    cost.sum().backward()
    assert not x.grad[:3].any() and (x.grad[3:] > 0).all()
    huge = ns['descent_entry_forward_position_cost'](torch.tensor([1e10]), torch.tensor([.22]))
    ceiling = ns['TUNING']['entry_forward_position_gain']*ns['TUNING']['entry_forward_position_cap']
    assert torch.isfinite(huge).all() and huge.item() <= ceiling
    # Deliberately bad positive-X pose must fail the zero-cost criterion.
    try:
        assert cost[-1] == 0
    except AssertionError:
        red_control = True
    else:
        raise AssertionError('forward penalty probe cannot detect bad posture')
    legacy_swing = ns['descent_forward_excess']((x.detach()-upper)/ns['TUNING']['entry_x_scale'])*ns['TUNING']['rear_swing_mix']
    assert (cost[3:] > legacy_swing[3:]).all()
    # Existing context test covers airborne cost, non-descent zero, both lower
    # rear supports ending the phase, and compact-leg escape prevention.
    context = check_entry(ns)
    # Check the actual integration, not just the standalone shaping function:
    # both rear wheels airborne on the original top must pay the full new cost.
    env = SimpleNamespace(num_envs=1, device='cpu', step_dt=.02,
        episode_length_buf=torch.tensor([2]), scene={'height_scanner':
            SimpleNamespace(data=SimpleNamespace(pos_w=torch.zeros(1, 3)))})
    wheels = torch.zeros(1, 4, 3)
    wheels[:, :, 2] = .84
    delta = torch.zeros_like(wheels)
    delta[:, :, 2] = -.40
    delta[:, 2:, 0] = .39
    valid = torch.ones(1, 4, dtype=torch.bool)
    loaded = valid.clone()
    contact_time = torch.full((1, 4), .2)
    command = torch.tensor([[.5, 0., 0.]])
    def tick():
        env.episode_length_buf += 1
        return ns['_descent_entry_retract_context'](env,
            dict(down_gate=torch.ones(1), edge_x=torch.tensor([.4])), torch.tensor([True]),
            wheels, delta, torch.full((1, 4), .75), valid, loaded, contact_time,
            torch.zeros_like(wheels), command, torch.tensor([[1., 0., 0., 0.]]),
            command, torch.zeros(1))['cost']
    tick()  # Establish the entry from actual top support before take-off.
    loaded[:, 2:] = False
    contact_time[:, 2:] = 0
    actual = tick()
    expected = ns['descent_entry_forward_position_cost'](
        delta[:, 2:, 0], torch.full((1, 2), ns['TUNING']['entry_swing_x_upper'])).mean(-1)
    assert env._m20_descent_entry_state['active'].all()
    assert torch.allclose(actual, expected), (actual, expected)
    return dict(forward_x_m=x.detach().tolist(), new_cost=cost.detach().tolist(),
                old_swing_cost=legacy_swing.tolist(), positive_x_gradient=x.grad.tolist(),
                cost_ceiling=ceiling, bad_pose_red_control=red_control, entry_context=context,
                integrated_airborne_full_cost=actual.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = check(load(MDP/'stair_teacher.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
