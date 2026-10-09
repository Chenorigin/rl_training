from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py');s=p.read_text()
s=s.replace('"rear_swing_b_limit": np.deg2rad(45.0), "rear_swing_g_limit": np.deg2rad(35.0),','"rear_swing_b_limit": np.deg2rad(35.0), "rear_swing_g_limit": np.deg2rad(25.0),')
s=s.replace('"rear_swing_mix": 0.4,','"rear_swing_mix": 0.75,')
s=s.replace('"entry_range_cost_weight": -0.25,','"entry_range_cost_weight": -1.0,\n    "entry_swing_x_lower": -0.10, "entry_swing_x_upper": 0.22,\n    "descent_cost_soft_cap": 4.0,')
s=s.replace('rear_support_cost = bounded(torch.maximum(excess_b,excess_g))','rear_support_cost = descent_forward_excess(torch.maximum(excess_b,excess_g))')
s=s.replace('rear_swing_cost = bounded(torch.maximum(swing_b,swing_g))','rear_swing_cost = descent_forward_excess(torch.maximum(swing_b,swing_g))')
needle='def descent_entry_position_band(depth,relax):'
s=s.replace(needle,'''def descent_forward_excess(value):
    """Monotonic forward-posture cost with useful severe-error sensitivity."""
    huber = 2*(torch.sqrt(1+value.clamp_min(0).square())-1)
    return huber/(1+huber/TUNING["descent_cost_soft_cap"])


'''+needle)
# New state fields: locate exact initialization.
a=s.index('def _descent_entry_retract_context');b=s.index('def stair_descent_entry_retract',a)
part=s[a:b]
print(part[:1400])
part=part.replace('"active":', '"active":',1)
# initialize settled alongside ready; actual dict uses keyword args.
part=part.replace('state=dict(active=', 'state=dict(settled=torch.zeros(n,2,dtype=torch.bool,device=device),active=')
# Support alternative spacing verified below.
part=part.replace('state = dict(active=', 'state = dict(settled=torch.zeros(n,2,dtype=torch.bool,device=device),active=')
part=part.replace('"credit","blocked")','"credit","blocked","transfer_samples","transfer_excess","transfer_swing_samples","transfer_cost_sum")')
part=part.replace('state["best"][start]=score[start]', 'state["best"][start]=score[start];state["settled"][start]=False')
part=part.replace('    state["active"]&=near&aligned&~gone.all(-1)&down', '''    # A lower ray hit means the wheel has crossed the edge, not that it has
    # landed. Keep the physical entry phase through the airborne transfer and
    # across command pauses; reward gates never erase its one-shot ledger.
    lower_support = gone & stable[:,2:] & (
        (wheels[:,2:,2]-TUNING["wheel_radius"]-ground[:,2:]).abs()<TUNING["entry_ground_tolerance"])
    state["settled"] |= lower_support & state["active"][:,None]
    state["active"] &= near & aligned & ~state["settled"].all(-1)''')
part=part.replace('    support=state["active"][:,None]&on_top&stable[:,2:]&(other>=2)', '    support=state["active"][:,None]&down[:,None]&on_top&stable[:,2:]&(other>=2)')
part=part.replace('    cost=(normalized.square()/(1+normalized.square()))*support', '''    # Continuous guard, including the airborne gap. Swing has a wider band
    # so reaching the lower tread remains possible; it cannot evade all cost
    # by briefly unloading a forward-leaning rear leg.
    depth=(-body_delta[:,2:,2]).clamp_min(0)
    swing_upper=torch.minimum(depth*torch.tan(depth.new_tensor(TUNING["rear_swing_b_limit"])+
        relax[:,None]*TUNING["rear_forward_relax"]),
        depth.new_full(depth.shape,TUNING["entry_swing_x_upper"])+
        relax[:,None]*TUNING["entry_x_high_rise_relax"])
    guard_lower=torch.where(loaded[:,2:],lower,TUNING["entry_swing_x_lower"])
    guard_upper=torch.where(loaded[:,2:],upper,swing_upper)
    guard_error=(guard_lower-rear_x).clamp_min(0)+(rear_x-guard_upper).clamp_min(0)
    guard=state["active"][:,None]&down[:,None]
    guard_gain=torch.where(loaded[:,2:],1.0,TUNING["rear_swing_mix"])
    cost=descent_forward_excess(guard_error/TUNING["entry_x_scale"])*guard*guard_gain''')
part=part.replace('    return dict(gain=gain.mean(-1),cost=cost.mean(-1))','''    stats["transfer_samples"]+=guard
    stats["transfer_excess"]+=guard&(guard_error>1e-6)
    stats["transfer_swing_samples"]+=guard&~loaded[:,2:]
    stats["transfer_cost_sum"]+=cost
    return dict(gain=gain.mean(-1),cost=cost.mean(-1))''')
s=s[:a]+part+s[b:]
s=s.replace('"Soft guard against forward/rearward overshoot, only at descent entry."','"Continuous entry guard until both rear wheels actually settle below."')
needle='            result[f"{leg}_entry_blocked_fraction"]=ss["blocked"]/count'
s=s.replace(needle,needle+'''
            transfer=ss["transfer_samples"].clamp_min(1)
            result[f"{leg}_entry_transfer_samples"]=ss["transfer_samples"]
            result[f"{leg}_entry_transfer_excess_fraction"]=ss["transfer_excess"]/transfer
            result[f"{leg}_entry_transfer_swing_fraction"]=ss["transfer_swing_samples"]/transfer
            result[f"{leg}_entry_transfer_cost_mean"]=ss["transfer_cost_sum"]/transfer''')
p.write_text(s)
