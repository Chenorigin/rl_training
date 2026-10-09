from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py')
s=p.read_text()
s=s.replace('from isaaclab.utils.math import quat_apply, yaw_quat','from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat')
s=s.replace('"front_support_grace": 0.20, "front_fold_weight": -0.5,','"front_support_grace": 0.0, "front_fold_weight": -3.0,')
s=s.replace('"same_tread_event_weight": -1.0, "same_tread_dwell_weight": -1.0,','"same_tread_event_weight": -4.5, "same_tread_dwell_weight": -1.0,')
a=s.index('\n}\n',s.index('TUNING ='))
s=s[:a]+'''
    # Persistent physical tread ledger; these are perception/style tolerances.
    "riser_identity_distance": 0.12, "riser_height_tolerance": 0.04,
    "riser_lateral_corridor": 0.45, "riser_min_spacing": 0.15,
    "riser_max_spacing": 0.50, "riser_cross_clearance": 0.01,
    "tread_inset": 0.025, "tread_height_tolerance": 0.04,
    "landing_time": 0.04, "prep_distance": 0.60,
    "prep_weight": 0.5, "rear_completion_weight": 3.0,
    "front_touchdown_weight": 0.25,
    "front_a_support_low_deg": 90.0, "front_a_support_high_deg": 80.0,
    "front_a_swing_low_deg": 75.0, "front_a_swing_high_deg": 55.0,
    "pose_low_rise": 0.15, "pose_high_rise": 0.25,
    "pose_angle_scale": np.deg2rad(20.0), "pose_length_scale": 0.10,
    "pose_load_force_scale": 60.0,
    "rear_b_limit": np.deg2rad(35.0), "rear_g_limit": np.deg2rad(25.0),
    "rear_forward_relax": np.deg2rad(5.0), "rear_forward_weight": -2.0,
    "body_gap_target": 0.22, "body_gap_scale": 0.10,
    "body_bottom_z": -0.07, "body_gap_weight": -1.0,
    "pose_tracking_discount": 0.25,
''' +s[a:]
s=s.replace('"ascending_edges_upper_z": profile[:, 1:],','"ascending_edges_upper_z": profile[:, 1:],\n        "ascending_edges_lower_z": profile[:, :-1],')
a=s.index('def _new_state(');b=s.index('\n\ndef _reset(',a)
part=s[a:b].replace('    return {','    state = {',1)
part+='''
    state.update({
        "map_count": torch.zeros(n,device=device,dtype=torch.long),
        "map_xy": zeros(64,2), "map_heading": zeros(64,2),
        "map_z": zeros(64), "map_lower_z": zeros(64),
        "map_next_x": torch.full((n,64),torch.inf,device=device),
        "map_intermediate": flags(64), "map_landed": flags(64,4),
        "map_safe": flags(64,4), "map_failed": flags(64,4),
        "map_stable_time": zeros(64,4), "map_paid": flags(64,2),
        "map_correct": flags(64,2), "map_side": torch.full((n,64,2),-1,device=device,dtype=torch.long),
        "map_duplicate_paid": flags(64,2), "map_prep_max": zeros(64,2),
        "map_lead": torch.full((n,64,2),-1,device=device,dtype=torch.long),
        "map_dwell": zeros(64,2), "map_face_paid": flags(64,2),
        "map_flight": torch.full((n,64),-1,device=device,dtype=torch.long),
        "flight_count": torch.zeros(n,device=device,dtype=torch.long),
        "axis_last_slot": torch.full((n,2),-1,device=device,dtype=torch.long),
        "axis_target_slot": torch.full((n,2),-1,device=device,dtype=torch.long),
        "previous_wheel": zeros(4,3), "wheel_history_valid": flags(),
        "radius_calibrated": flags(4), "rear_lift_gain": zeros(),
        "rear_lift_credit": zeros(), "skip_total": zeros(2),
        "physical_total": zeros(2), "unsafe_total": zeros(2),
        "axis_run": zeros(2), "axis_max_run": zeros(2),
        "duplicate_event_count": zeros(2),
    })
    state["radius"].fill_(TUNING["wheel_radius"])
    return state
'''
s=s[:a]+part+s[b:]
s=s.replace('value[reset] = float("nan")','value[reset] = TUNING["wheel_radius"]',1)
s=s.replace('"front_same_tread_paid", "rear_same_tread_paid", "same_tread_riser"}', '"front_same_tread_paid", "rear_same_tread_paid", "same_tread_riser",\n                     "map_side", "map_lead", "axis_last_slot", "axis_target_slot", "map_flight"}')
s=s.replace('        else:\n            value[reset] = 0\n\n\ndef _contact_on_tread','        elif key == "map_next_x":\n            value[reset] = torch.inf\n        else:\n            value[reset] = 0\n\n\ndef _contact_on_tread',1)
a=s.index('def ascent_state(');b=s.index('# =============================================================================\n# 5.',a)
s=s[:a]+Path('.scratch/gait_v3_20261008/new_ledger.txt').read_text()+'\n\n'+s[b:]
s=s.replace('return (state["front_violation_event"] | state["rear_violation_event"]).float() / env.step_dt','return (state["front_violation_event"].float() + state["rear_violation_event"].float()) / env.step_dt')
s=s.replace('return (state["front_same_tread_event"] | state["rear_same_tread_event"]).float() / env.step_dt','return state["duplicate_event_count"].sum(-1) / env.step_dt')
p.write_text(s)
