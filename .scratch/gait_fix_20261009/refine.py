from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py');s=p.read_text()
s=s.replace('"map_duplicate_paid": flags(64,2), "map_prep_max": zeros(64,2),','"map_duplicate_paid": flags(64,2), "map_prep_max": zeros(64,2),\n        "rear_prep_initialized": flags(64),')
s=s.replace('''            state["map_prep_max"][row[changed],target[changed],axle] = current[changed]''','''            initialize = has_target & ~state["rear_prep_initialized"][row,target]
            state["map_prep_max"][row[initialize],target[initialize],axle] = current[initialize]
            state["rear_prep_initialized"][row[initialize],target[initialize]] = True''')
s=s.replace('''    phase["active"] &= (dist < TUNING["pose_phase_radius"]) & ~(finished & ~seen_down)''','''    aligned = (heading*phase["heading"]).sum(-1) > 0.5
    phase["active"] &= (dist < TUNING["pose_phase_radius"]) & aligned & ~(finished & ~seen_down)''')
# Loaded positive means & swing/loaded per-leg tail histograms, no unbounded history.
s=s.replace('''    _update_rear_swing_stats(env,down,body_angle[:,2:],gravity_angle[:,2:],rear_swing,rear_swing_cost)''','''    _update_rear_swing_stats(env,down,body_angle[:,2:],gravity_angle[:,2:],rear_swing,rear_swing_cost)
    _update_rear_angle_histograms(env,body_angle[:,2:],gravity_angle[:,2:],rear_load,rear_swing)''')
idx=s.index('\ndef stair_rear_wheel_lift(')
s=s[:idx]+'''
def _update_rear_angle_histograms(env,b,g,loaded,swing):
    hist = getattr(env,"_m20_rear_angle_histograms",None)
    if hist is None:
        hist = {f"{phase}_{coord}":torch.zeros(env.num_envs,2,19,device=env.device)
                for phase in ("loaded","swing") for coord in ("b","g")}
        env._m20_rear_angle_histograms = hist
    reset = env.episode_length_buf <= 1
    for phase,mask in (("loaded",loaded),("swing",swing)):
        for coord,angle in (("b",b),("g",g)):
            value = hist[f"{phase}_{coord}"]; value[reset] = 0
            bins = (torch.rad2deg(angle).clamp(0,90)/5).long().clamp_max(18)
            value.scatter_add_(2,bins[:,:,None],mask.float()[:,:,None])

''' +s[idx:]
s=s.replace('''    return result


# =============================================================================
# 7. Terrain curriculum''','''    hist = getattr(env,"_m20_rear_angle_histograms",None)
    if hist is not None:
        for key,value in hist.items():
            counts = value[env_ids].sum(0)
            for i,leg in enumerate(("hl","hr")):
                total = counts[i].sum()
                # Upper edge of a 5-degree bin; negative angles enter zero bin.
                q = ((counts[i].cumsum(-1) < .95*total).sum()+1).clamp_max(18)*5
                result[f"{leg}_{key}_p95_upper_deg"] = torch.where(total>0,q,0).float()
    return result


# =============================================================================
# 7. Terrain curriculum''')
p.write_text(s)
p=Path('scripts/tools/check_ascent_direction.py');s=p.read_text();s=s.replace('value(c)<2.001','value(c)<3*ns["TUNING"]["direction_soft_cap"]+.001');s=s.replace('assert ns[\'TUNING\'][\'front_fold_weight\']==-.5 and ns[\'TUNING\'][\'same_tread_event_weight\']==-1','assert ns[\'TUNING\'][\'front_fold_weight\']==-1 and ns[\'TUNING\'][\'same_tread_event_weight\']==-1.5');s=s.replace("results['oct07_weights_reverted']=True","results['task_preservation_weights_retained']=True");p.write_text(s)
