from pathlib import Path
p=Path('scripts/tools/check_descent_entry.py');s=p.read_text();needle='    # Nonfinite rays do not start a new detector.'
add='''    # A good initial entry posture must still be penalized if it later leans
    # forward. Keep the transfer until actual lower support, not a lower ray.
    e.episode_length_buf[:]=0;valid.fill_(True);loaded.fill_(True);time.fill_(.2)
    ground.fill_(.75);wheels[:,:,2]=.84;delta[:,:,2]=-.40;delta[:,2:,0]=.10
    scan['down_gate'].fill_(1);root_velocity[:,0]=.5
    tick();delta[:,2:,0]=.32;r=tick()
    assert not r['gain'].any() and (r['cost']>0).all();out['good_entry_then_bad_is_penalized']=r['cost'].tolist()
    ground[:,2:]=.60;loaded[:,2:]=False;time[:,2:]=0
    r=tick();assert e._m20_descent_entry_state['active'].all() and (r['cost']>0).all()
    out['lower_ray_airborne_guard_retained']=True
    assert not tick(False)['cost'].any() and e._m20_descent_entry_state['active'].all()
    assert (tick(True)['cost']>0).all();out['command_pause_resumes_guard']=True
    # One stable lower rear wheel is insufficient; both must be truly settled.
    loaded[:,2]=True;time[:,2]=.2;wheels[:,2,2]=.69
    tick();assert e._m20_descent_entry_state['active'].all()
    loaded[:,3]=True;time[:,3]=.2;wheels[:,3,2]=.84
    tick();assert e._m20_descent_entry_state['active'].all()  # ray lower but foot above it
    wheels[:,3,2]=.69;r=tick();assert not e._m20_descent_entry_state['active'].any() and not r['cost'].any()
    out['both_true_lower_support_clear_guard']=True
    assert e._m20_descent_entry_state['settled'].all()
    # New cost stays finite, zero for negative excess, stronger for large errors.
    values=ns['descent_forward_excess'](torch.tensor([-2.,0.,.5,1.,2.,3.,100.]))
    assert values[0]==values[1]==0 and torch.isfinite(values).all()
    assert (values[2:]<ns['TUNING']['descent_cost_soft_cap']).all()
    assert (values[3:]>values[2:-1]).all()
    out['descent_huber_cost']=values.tolist()
'''
assert needle in s;s=s.replace(needle,add+needle);p.write_text(s)
