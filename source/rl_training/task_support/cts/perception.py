"""CTS timing mixture, independent of terrain, command and sensor-noise draws."""
import math


def validate_capture(cfg):
    if not isinstance(cfg,dict) or set(cfg) != {'padding_m'}:
        raise ValueError('Unknown or missing CTS map capture setting')
    p = cfg['padding_m']
    if type(p) not in (float,int) or not math.isfinite(p) or not 0 <= p <= 2.:
        raise ValueError('CTS map capture padding must be finite in [0,2]')
    return cfg


def validate_mixture(cfg):
    if not isinstance(cfg,dict) or set(cfg) != {'zero_delay_probability'}:
        raise ValueError('Unknown or missing CTS map delay mixture setting')
    p = cfg['zero_delay_probability']
    if type(p) not in (float,int) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('CTS zero-delay probability must be finite in [0,1]')
    return cfg


def validate_perception_contract(mixture, perception):
    from task_support.student_v1.configuration import StudentV1PerceptionCfg
    validate_mixture(mixture)
    StudentV1PerceptionCfg(**perception).validate()
    if perception.get('zero_delay_probability') != mixture['zero_delay_probability']:
        raise ValueError('CTS map delay mixture and effective sensor configuration differ')


def delay_diagnostics(state, teacher, *, scalar_only=False):
    """Count actual delivered-packet delays separately from episode assignments."""
    result = {'configured_zero_delay_probability':state.cfg.zero_delay_probability}
    for name,cohort in (('teacher',teacher),('student',~teacher)):
        zero = int(state.zero_delivery[cohort].sum())
        assigned = int(state.zero_delivery_resets[cohort].sum())
        delayed = int(state.delayed_delivery_resets[cohort].sum())
        packets = int(state.zero_delivery_packets[cohort].sum())
        delayed_packets = int(state.delayed_delivery_packets[cohort].sum())
        result.update({name+'_current_zero_delay_fraction':zero/max(1,int(cohort.sum())),
                       name+'_zero_delay_episode_assignments':assigned,
                       name+'_delayed_episode_assignments':delayed,
                       name+'_zero_delay_packets':packets,name+'_delayed_packets':delayed_packets})
    if not scalar_only:
        capture,arrival = state.packet_capture,state.packet_arrival
        valid = capture >= 0
        actual = arrival-capture
        zero = state.zero_delivery[None,:].expand_as(valid)
        if (actual[valid & zero] != 0).any():
            raise RuntimeError('CTS zero-delay episode contains delayed map packets')
        # This verifies the actual queue, including captured but not yet delivered packets.
        lo = math.ceil(state.cfg.min_delay_s/state.step_dt-1.e-8)
        hi = math.ceil(state.cfg.max_delay_s/state.step_dt)
        normal = actual[valid & ~zero]
        if ((normal < lo) | (normal > hi)).any():
            raise RuntimeError('CTS delayed episode contains out-of-range map packets')
        result.update(packet_queue_checked=int(valid.sum()),
                      zero_delay_queue_packets=int((valid & zero).sum()),
                      delayed_queue_packets=int((valid & ~zero).sum()),
                      delayed_queue_range_steps=[int(normal.min()),int(normal.max())] if len(normal) else [],
                      episode_sampling='independent Bernoulli at reset; counts include initialization resets',
                      zero_delivery_preserves='capture period, map age, FOV, dropout, noise and action delay')
    return result
