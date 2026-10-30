"""Teacher-only finite-course safety geometry; actor still receives sensor observations."""
import torch
from task_support.stair_quiet_v1_safe.observations import physical_sample as legacy_sample
from task_support.stair_quiet_v1_width.geometry import capsule_contact_velocity, wheel_margins
from .terrain_geometry import regions
from .terrain_course import course_state


def physical_sample(env):
    state=course_state(env)
    d=legacy_sample(env)
    d.update({key:state[key] for key in ('spans','round_trip','goal','goal_height')})
    return d


def safety_geometry(d,s):
    pos,axis=d['position'],d['axis']
    forward,side,top=regions(pos,axis,d,s)
    future=pos+d['velocity']*s.prediction_s
    predicted_forward,_,_=regions(future,axis,d,s)
    _,margin=wheel_margins(s,pos)
    extent=s.wheel_radius+s.wheel_half_length*axis[...,1].abs()+margin
    half=d['width'][:,None]/2
    left,right=half-pos[...,1]-extent,half+pos[...,1]-extent
    future_left,future_right=half-future[...,1]-extent,half+future[...,1]-extent
    bottom=pos[...,2]-s.wheel_radius-s.wheel_half_length*axis[...,2].abs()
    f=d['forces']
    supported=(f[...,2]>s.support_force_N) & (f[...,2]>torch.linalg.vector_norm(f[...,:2],dim=-1)) & ((bottom-top).abs()<.05)
    slip_speed=torch.linalg.vector_norm(capsule_contact_velocity(d['velocity'],d['omega'],axis,s.wheel_radius,s.wheel_half_length)[...,:2],dim=-1)
    return dict(forward=forward,side=side,left=left,right=right,top=top,clearance=bottom-top,
                predicted_forward=predicted_forward,predicted_left=future_left,predicted_right=future_right,
                supported=supported,slip_speed=slip_speed)


def teacher_safety(env):
    d=physical_sample(env);g=safety_geometry(d,env.cfg.stair_tail)
    feet=torch.stack((g['forward']/.1,g['left']/.1,g['right']/.1,g['predicted_forward']/.1,
        g['predicted_left']/.1,g['predicted_right']/.1,g['supported'].float(),g['slip_speed']/.3,
        d['forces'][...,2]/500.,d['velocity'][...,0],d['velocity'][...,1]),-1).flatten(1)
    body=torch.stack((d['root'][:,1]/(d['width']/2),d['root_velocity'][:,1],d['yaw'].sin(),d['yaw'].cos()),-1)
    result=torch.cat((feet,body),-1).clamp(-10,10)
    if result.shape[-1]!=48: raise ValueError('Teacher safety ABI changed')
    return result
