from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py')
s=p.read_text().replace('        "tread_contact_time": zeros(2, 64, 2),','''        "current_flight": torch.zeros(n, device=device, dtype=torch.long),
        "front_history_flight": torch.zeros((n, 64), device=device, dtype=torch.long),
        "tread_contact_time": zeros(2, 64, 2),''')
s=s.replace('    state["tread_intermediate"] |= slots < (count - 1)[:, None]', '''    next_slot = (slots + 1).clamp_max(63).expand(env.num_envs, -1)
    same_flight = state["front_history_flight"] == state["front_history_flight"].gather(1, next_slot)
    state["tread_intermediate"] |= (slots < (count - 1)[:, None]) & same_flight''')
s=s.replace('    has_next = slots + 1 < count[:, None]', '    has_next = (slots + 1 < count[:, None]) & same_flight')
s=s.replace('''    adjacent = (count == 0) | (
        (along >= 0.15)''','''    adjacent = (count == 0) | (
        ((heading * last_heading).sum(-1) > 0.8)
        & (along >= 0.15)''')
s=s.replace('''    retired_distance = torch.linalg.vector_norm(candidate_xy - state["retired_xy"], dim=1)''','''    # A completed top platform can connect to another flight in the same
    # episode. Preserve all credit/landing history; only release expected side.
    wheel_past_last = ((wheel[..., :2] - last_xy[:, None]) * last_heading[:, None]).sum(-1)
    clear_platform = (contact.all(-1) & (wheel_past_last > 0.06).all(-1)
        & ((wheel[..., 2] - state["front_history_target_z"][row, last_i, None]
            - state["radius"]).abs() <= 0.05).all(-1))
    new_flight = ((count > 0) & ~adjacent & (state["rear_count"] >= count)
                  & clear_platform & (torch.linalg.vector_norm(delta_xy, dim=-1) > 0.80))
    retired_distance = torch.linalg.vector_norm(candidate_xy - state["retired_xy"], dim=1)''')
s=s.replace('        & adjacent & ~repeated & (count < 64)', '        & (adjacent | new_flight) & ~repeated & (count < 64)')
s=s.replace('    state["active"][start] = True', '''    flight_start = start & new_flight
    state["current_flight"][flight_start] += 1
    state["front_expected"][flight_start] = -1
    state["rear_expected"][flight_start] = -1
    state["active"][start] = True''')
s=s.replace('    state["front_history_xy"][ids, slots] = state["edge_xy"][ids]', '''    state["front_history_flight"][ids, slots] = state["current_flight"][ids]
    state["tread_seen"][ids, 0, slots] |= candidates[ids]
    state["front_history_xy"][ids, slots] = state["edge_xy"][ids]''')
s=s.replace('    state["rear_count"][rear_retired] += 1', '''    rear_ids = torch.nonzero(rear_retired, as_tuple=False).squeeze(-1)
    state["tread_seen"][rear_ids, 1, rear_i[rear_ids]] |= rear_candidates[rear_ids]
    state["rear_count"][rear_retired] += 1''')
p.write_text(s)
# Existing baseline check must wait the real .04s confirmation instead of
# mislabelling a single new contact frame as a completed stable landing.
p=Path('scripts/tools/check_stair_rewards.py')
s=p.read_text().replace('''    s=step(env,1,.10,.19,True,edge=.3,height=.2)
    results['same_tread_cost']''','''    step(env,1,.10,.19,True,edge=.3,height=.2)
    s=step(env,1,.10,.19,True,edge=.3,height=.2)
    results['same_tread_cost']''')
p.write_text(s)
