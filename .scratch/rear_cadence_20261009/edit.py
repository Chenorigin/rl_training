from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py');s=p.read_text()
s=s.replace('"rear_target_x_scale": 0.30, "rear_target_z_scale": 0.20,','"rear_target_x_scale": 0.30, "rear_target_z_scale": 0.20,\n    "rear_swing_clearance": 0.04, "rear_swing_target_ceiling": 0.15,\n    "rear_recovery_weight": 1.0,')
s=s.replace('"rear_prep_initialized": flags(64),','"rear_prep_initialized": flags(64), "rear_prep_max": zeros(64,2),\n        "rear_prep_paid": zeros(64), "rear_prep_blocked_gain": zeros(),\n        "rear_recovery_event": flags(), "rear_recoveries": zeros(),\n        "rear_style_refund": zeros(), "rear_style_refund_total": zeros(),\n        "map_rear_style_credit": zeros(64),')
s=s.replace('state["front_lift_settlement"].zero_()','state["front_lift_settlement"].zero_()\n    state["rear_recovery_event"].zero_(); state["rear_style_refund"].zero_()')
needle='        state[name+"_event"] = landed & transition[:, None]'
add='''        if axle == 1:
            # A safe single landing beyond a duplicated rear tread starts a
            # recovery. Its partner must still support that previous tread.
            # This is not counted as a strict alternating transition.
            recovery = correct & (last >= 0) & state["map_duplicate_paid"][row,previous,axle] & (
                stable[row,previous,3-side]) & state["map_intermediate"][row,candidate]
            state["rear_recovery_event"] = recovery
            state["rear_recoveries"] += recovery.float()
            style_credit = (transition.float()*TUNING["rear_completion_weight"]
                            + recovery.float()*TUNING["rear_recovery_weight"])
            state["map_rear_style_credit"][row,candidate] += style_credit
'''
assert needle in s;s=s.replace(needle,add+needle)
needle='        state["map_duplicate_paid"][:, :, axle] |= new_duplicate'
add='''        if axle == 1:
            # A later catch-up on the same tread invalidates the earlier style
            # credit. Return exactly that credit, once, in addition to the
            # existing same-tread cost; task crossing credit stays independent.
            refund = (new_duplicate*state["map_rear_style_credit"]).sum(-1)
            state["rear_style_refund"] = refund
            state["rear_style_refund_total"] += refund
'''
s=s.replace(needle,add+needle)
a=s.index('            score = rear_target_score(tx,');b=s.index('    # Task credit is independent',a)
old=s[a:b]
# Keep the original front preparation code, replacing only rear branch.
front=old[old.index('        else:\n            progress = torch.where(eligible, progress, 0)'):]
front=front.replace('        else:\n            progress = torch.where(eligible, progress, 0)\n','')
init_a=front.index('        if axle == 1:');init_b=front.index('        gain =',init_a)
front=front[:init_a]+front[init_b:]
front=front.replace('(gain if axle == 1 else torch.minimum(gain,budget))','torch.minimum(gain,budget)')
s=s[:a]+'''            bottom = wheels[:,sl,2]-state["radius"][:,sl]
            target_z = state["map_z"][row,target,None]
            source_z = state["map_lower_z"][row,target,None]
            rise = (target_z-source_z).clamp_min(0.06)
            raw = rear_target_score(tx,bottom,target_x[:,None],target_z)
            raw *= ((bottom-source_z)/rise).clamp(0,1)
            eligible = has_target[:,None] & enabled[:,None] & ~loaded[:,sl] & (other_support>=2) & (
                bottom > source_z+TUNING["rear_swing_clearance"]) & (
                bottom <= target_z+TUNING["rear_swing_target_ceiling"]) & (
                tx > -TUNING["prep_distance"]) & (
                tx < state["map_next_x"][row,target,None]-TUNING["tread_inset"])
            gain = rear_preparation_gain(state,row,target,has_target,raw,eligible,expected)
            state["rear_lift_gain"] = gain
            state["rear_lift_credit"] += gain
            continue
        progress = torch.where(eligible, progress, 0)
'''+front+s[b:]
needle='def rear_target_score(x, bottom_z, target_x, target_z):'
helper='''def rear_preparation_gain(state, row, target, has_target, raw, eligible, expected):
    """Per-foot geometric high-water marks, independent of support eligibility.

    Initial pose, unloading, changes of supporting partners and changing the
    selected leg never unlock previously observed scores. Total credit is at
    most one unit per physical target, across both rear feet and all retries.
    """
    previous = state["rear_prep_max"][row,target]
    initialized = state["rear_prep_initialized"][row,target]
    delta = (raw-previous).clamp_min(0)*initialized[:,None]*has_target[:,None]
    potential = delta*eligible
    side = torch.where(expected>=0,expected,potential.argmax(-1))
    gain = potential.gather(1,side[:,None]).squeeze(-1)
    gain = torch.minimum(gain,(1-state["rear_prep_paid"][row,target]).clamp_min(0))
    state["rear_prep_blocked_gain"] += (delta.sum(-1)-gain).clamp_min(0)
    state["rear_prep_max"][row,target] = torch.where(has_target[:,None],
        torch.maximum(previous,raw),previous)
    state["rear_prep_initialized"][row,target] |= has_target
    state["rear_prep_paid"][row,target] += gain
    state["map_lead"][row,target,1] = torch.where(gain>0,side,state["map_lead"][row,target,1])
    return gain


'''
s=s.replace(needle,helper+needle)
needle='def stair_up_order_violation_cost('
func='''def stair_rear_transition_recovery(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    """Small once-only signal for a safe first step out of rear same-tread gait."""
    return ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg)["rear_recovery_event"].float()/env.step_dt


def stair_rear_style_refund(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    """Withdraw earlier rear style credit when that tread later becomes shared."""
    return ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg)["rear_style_refund"]/env.step_dt


'''
s=s.replace(needle,func+needle)
s=s.replace('    result["up_rear_lift_credit"] = state["rear_lift_credit"][ids].mean()', '''    result["up_rear_lift_credit"] = state["rear_lift_credit"][ids].mean()
    result["up_rear_prep_blocked_gain"] = state["rear_prep_blocked_gain"][ids].mean()
    result["up_rear_recovery_events"] = state["rear_recoveries"][ids].mean()
    result["up_rear_style_refund"] = state["rear_style_refund_total"][ids].mean()
    result["up_rear_recovery_per_landing"] = state["rear_recoveries"][ids].sum()/state["physical_total"][ids,1].sum().clamp_min(1)''')
p.write_text(s)
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py');s=p.read_text()
needle='    stair_descent_rear_forward_cost = RewTerm('
s=s.replace(needle,'''    stair_rear_transition_recovery = RewTerm(
        func=stair_teacher.stair_rear_transition_recovery,
        weight=stair_teacher.TUNING["rear_recovery_weight"],
        params={k:v for k,v in _gait_entities().items() if k not in ("hip_cfg","knee_cfg")},
    )
    stair_rear_style_refund = RewTerm(
        func=stair_teacher.stair_rear_style_refund, weight=-1.0,
        params={k:v for k,v in _gait_entities().items() if k not in ("hip_cfg","knee_cfg")},
    )
'''+needle)
p.write_text(s)
