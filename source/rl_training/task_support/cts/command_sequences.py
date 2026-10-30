"""Training command sequences and geometric bounds; no simulator dependency.

Modes 0..8 and curriculum groups 0..4 keep their legacy identities. The
optional v14 course adds actual left/right translation and moving turns.
"""
import math
import torch


MODE_NAMES = ('forward', 'backward', 'yaw_left', 'yaw_right',
              'forward_left', 'forward_right', 'backward_left', 'backward_right',
              'idle', 'left', 'right',
              'forward_side_left', 'forward_side_right', 'backward_side_left', 'backward_side_right',
              'side_left_yaw_left', 'side_left_yaw_right', 'side_right_yaw_left', 'side_right_yaw_right',
              'wall_single', 'wall_corridor', 'wall_doorway', 'wall_front_stop', 'wall_corner')
GROUP_NAMES = ('forward', 'backward', 'turn', 'arc', 'idle', 'left', 'right', 'diagonal', 'lateral_turn',
               'wall_single', 'wall_corridor', 'wall_doorway', 'wall_front_stop', 'wall_corner')
MODE_GROUPS = (0, 1, 2, 2, 3, 3, 1, 1, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 10, 11, 12, 13)


def _step(time, ramp):
    fraction = (time/ramp.clamp_min(torch.finfo(time.dtype).eps)).clamp(0., 1.)
    smooth = fraction.square()*(3-2*fraction)
    return torch.where(ramp > 0, smooth, (time >= 0).to(time.dtype))


def command_sequence(command, elapsed, duration, ramp, transition):
    """Four equal phases: straight, original arc, reverse arc, straight.

    The longitudinal velocity remains unchanged across the turning phases.
    Positive ramps smoothly blend each target; zero ramps preserve deliberate
    abrupt commands. Rise/fall windows are inside the motion phase, so stop
    semantics and the allotted stop hold are unchanged.
    """
    elapsed = elapsed.to(command.dtype)
    duration, ramp = duration.to(command.dtype), ramp.to(command.dtype)
    envelope = torch.minimum(_step(elapsed, ramp), _step(duration-elapsed, ramp))
    envelope *= ((elapsed >= 0) & (elapsed < duration)).to(command.dtype)
    yaw = (_step(elapsed-duration/4, ramp)-2*_step(elapsed-duration/2, ramp)
           +_step(elapsed-3*duration/4, ramp))
    result = command*envelope[:, None]
    result[:, 2] *= torch.where(transition, yaw, torch.ones_like(yaw))
    return result


def lateral_duration(speed, travel, ramp):
    """The two smoothstep edge ramps lose one ramp interval of travel."""
    if (speed <= 0).any():
        raise ValueError('Lateral motion requires positive speed')
    duration = travel/speed+ramp
    if (duration < 2*ramp).any():
        raise ValueError('Lateral travel cannot accommodate its command ramps')
    return duration


def lateral_spawn(xy, bounds, mode, travel, half_length, half_width,
                  wheel_radius, wheel_half_length, boundary_margin):
    """Keep the whole nominal lateral path in the actual collision field.

    The yaw-invariant support radius reserves body and wheel footprint. The
    full travel is reserved along both axes, which also covers the small
    randomized initial yaw without relying on a point-body boundary test.
    Surface height is deliberately left to the caller's actual terrain query.
    """
    if not len(xy):
        return xy
    footprint = math.hypot(half_length, half_width)+wheel_radius+wheel_half_length
    margin = footprint+boundary_margin
    low = torch.stack((bounds[:, 0]+margin+travel, bounds[:, 2]+margin), -1)
    high = torch.stack((bounds[:, 1]-margin-travel, bounds[:, 3]-margin), -1)
    low[:, 1] += torch.where(mode == 10, travel, 0.)
    high[:, 1] -= torch.where(mode == 9, travel, 0.)
    if (low > high).any():
        raise ValueError('Terrain field cannot contain the lateral path and support footprint')
    return torch.maximum(low, torch.minimum(high, xy))


def transition_spawn(xy, bounds, command, duration, ramp, yaw, half_length,
                     half_width, wheel_radius, wheel_half_length, boundary_margin):
    """Conservative corridor for the straight/arc/reverse/straight sequence.

    The turn angle is bounded by |wz|*duration/4. Curvature changes lateral
    position over at most duration/2+ramp; initial yaw contributes drift over
    the whole path. A circular support envelope covers the rotating robot.
    """
    if not len(xy):
        return xy
    footprint = math.hypot(half_length, half_width)+wheel_radius+wheel_half_length
    margin = footprint+boundary_margin
    low = torch.stack((bounds[:, 0]+margin, bounds[:, 2]+margin), -1)
    high = torch.stack((bounds[:, 1]-margin, bounds[:, 3]-margin), -1)
    speed = command[:, 0].abs()
    travel = speed*duration
    angle = command[:, 2].abs()*duration/4
    can_reverse = yaw.abs()+angle >= math.pi/2
    low[:, 0] += torch.where((command[:, 0] < 0) | can_reverse, travel, 0.)
    high[:, 0] -= torch.where((command[:, 0] > 0) | can_reverse, travel, 0.)
    drift = travel*yaw.sin().abs()
    turn = speed*(duration/2+ramp)*angle.clamp(max=math.pi/2).sin()
    direction = command[:, 0]*command[:, 2]
    low[:, 1] += drift+torch.where(direction < 0, turn, 0.)
    high[:, 1] -= drift+torch.where(direction > 0, turn, 0.)
    if (low > high).any():
        raise ValueError('Terrain field cannot contain the turning sequence and support footprint')
    return torch.maximum(low, torch.minimum(high, xy))
