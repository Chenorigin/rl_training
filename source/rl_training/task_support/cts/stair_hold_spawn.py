"""Default-joint support poses on actual ascending treads, with no high drop."""
import math
import numpy as np


def candidates(record, cfg, radius):
    if record['course'] not in ('up', 'round_trip'):
        return []
    wheels = np.asarray(cfg['wheel_centers_body_m'])
    front, rear = wheels[:2].mean(0), wheels[2:].mean(0)
    length = front[0]-rear[0]
    height, depth, count = record['height'], record['depth'], record['steps']
    start = record['route_start']
    result = []
    for k in range(1, count-1):
        rise = k*height
        if rise >= length:
            continue
        pitch = math.asin(rise/length)
        if pitch > cfg['max_pitch_rad']:
            continue
        co, si = math.cos(pitch), math.sin(pitch)
        # R_y(-pitch): local +X points up the stair route.
        fx, rx = co*front[0]-si*front[2], co*rear[0]-si*rear[2]
        rz = si*rear[0]+co*rear[2]
        for j in range(min(cfg['rear_step_max'], count-k-2)+1):
            back = start+j*depth
            ahead = start+(j+k)*depth
            # Behind a wheel is a lower tread; in front is a higher riser.
            lo = max(back+cfg['back_edge_margin_m']-rx, ahead+cfg['back_edge_margin_m']-fx)
            hi = min(back+depth-radius-cfg['riser_margin_m']-rx,
                     ahead+depth-radius-cfg['riser_margin_m']-fx)
            if hi-lo < cfg['minimum_spawn_interval_m']:
                continue
            z = (j+1)*height+radius+cfg['spawn_clearance_m']-rz
            result.append(dict(x_min=lo,x_max=hi,z=z,pitch=pitch,rear_step=j,step_gap=k))
    return result


def sample_pose(record, options, cfg, rng):
    item = options[int(rng.integers(len(options)))]
    x = rng.uniform(item['x_min'], item['x_max'])
    y = rng.uniform(-cfg['lateral_spawn_m'],cfg['lateral_spawn_m'])
    pitch = item['pitch']
    return np.array([x,y,item['z']]), np.array([math.cos(pitch/2),0.,-math.sin(pitch/2),0.])
